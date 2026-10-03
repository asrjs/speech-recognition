import { normalizePcmInput } from '../../audio/index.js';
import { CtcDecoder } from '../../ctc/index.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import { mapLasrCtcNativeToCanonical } from '../lasr-ctc/mapping.js';
import type {
  AudioInputLike,
  SpeechModel,
  SpeechModelFactory,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import {
  createOrtSession,
  disposeOrtOutputs,
  initOrt,
  releaseOrtSession,
  type OrtModuleLike,
  type OrtTensorLike,
} from '../lasr-ctc/ort.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';
import { ParakeetCtcJsPreprocessor } from './frontend.js';
import { ParakeetCtcTokenizer } from './tokenizer.js';
import type { ParakeetCtcModelConfig, ParakeetCtcModelOptions } from './types.js';

export const DEFAULT_PARAKEET_CTC_CONFIG: ParakeetCtcModelConfig = {
  ecosystem: 'parakeet',
  architecture: 'parakeet-ctc',
  processorArchitecture: 'parakeet-mel-80',
  encoderArchitecture: 'parakeet-ctc-encoder',
  decoderArchitecture: 'ctc',
  sampleRate: 16000,
  featureHopSeconds: 0.01,
  rawStride: 8,
  nMels: 80,
  nFft: 512,
  winLength: 400,
  hopLength: 160,
  preemphasis: 0.97,
  vocabularySize: 1025,
  languages: ['en'],
  tokenizer: { kind: 'sentencepiece', blankTokenId: 1024 },
  featureLayout: 'mel-major',
};

function readFloat32(tensor: OrtTensorLike): Float32Array {
  const data = tensor.data;
  if (data instanceof Float32Array) {
    return data;
  }

  return Float32Array.from(Array.from(data as unknown as ArrayLikeLike<number>));
}

type ArrayLikeLike<T> = { readonly length: number; readonly [index: number]: T };

/** The exported graph declares `attention_mask` as rank 2, `[batch, frames]`. */
function toFrameMaskTensor(ort: OrtModuleLike, frames: number): OrtTensorLike {
  const mask = new BigInt64Array(frames);
  mask.fill(1n);

  return new ort.Tensor('int64', mask, [1, frames]);
}

/**
 * Loads a ParakeetForCTC graph plus its sentencepiece vocabulary and decodes
 * with the shared CTC decoder. The graph takes `input_features` and an
 * `attention_mask` and returns `[batch, frames, vocab]` logits.
 */
export async function loadParakeetCtcFromOptions(
  options: ParakeetCtcModelOptions,
): Promise<ParakeetCtcLoadedModel> {
  const source = options.source;
  if (!source || source.kind !== 'direct') {
    throw createExperimentalArtifactMissingError('parakeet-ctc', 'local artifact URLs');
  }

  const config: ParakeetCtcModelConfig = { ...DEFAULT_PARAKEET_CTC_CONFIG, ...options.config };
  const backendId = options.encoderBackend ?? 'wasm';
  const ort = await initOrt(backendId, {
    wasmPaths: options.wasmPaths,
    cpuThreads: options.cpuThreads,
    signal: options.signal ?? null,
  });
  const session = await createOrtSession(ort, source.artifacts.encoderUrl, {
    backendId,
    externalDataUrl: source.artifacts.encoderDataUrl,
    externalDataPath: source.artifacts.encoderFilename,
    signal: options.signal ?? null,
  });
  const tokenizer = await ParakeetCtcTokenizer.fromUrl(
    source.artifacts.tokenizerUrl,
    options.signal ?? null,
  );

  const secondsPerFrame = (config.hopLength / config.sampleRate) * config.rawStride;

  return {
    config,
    backendId,
    async transcribe(audio: Float32Array): Promise<LasrCtcNativeTranscript> {
      const started = nowMs();
      const pcm = normalizePcmInput(audio, { sampleRate: config.sampleRate });
      const mono = pcm.channels[0];
      if (!mono) {
        throw new Error('Parakeet CTC transcription requires mono audio.');
      }
      const batch = new ParakeetCtcJsPreprocessor({
        nMels: config.nMels,
        nFft: config.nFft,
        winLength: config.winLength,
        hopLength: config.hopLength,
        preemphasis: config.preemphasis,
      }).process(mono);

      const frames = batch.frameCount;
      const featureSize = batch.featureSize;
      // The shared frontend emits mel-major `[mel, frame]`; the Parakeet graph
      // declares `[batch, frames, mels]`, so transpose before feeding it.
      const frameMajor = new Float32Array(featureSize * frames);
      for (let mel = 0; mel < featureSize; mel += 1) {
        const source = mel * frames;
        for (let frame = 0; frame < frames; frame += 1) {
          frameMajor[frame * featureSize + mel] = batch.features[source + frame] ?? 0;
        }
      }
      const featureTensor = new ort.Tensor('float32', frameMajor, [1, frames, featureSize]);
      const maskTensor = toFrameMaskTensor(ort, frames);

      let outputs: Record<string, OrtTensorLike> | undefined;
      try {
        outputs = await session.run({
          input_features: featureTensor,
          attention_mask: maskTensor,
        });
        const logits = (outputs.logits ?? Object.values(outputs)[0]) as OrtTensorLike;
        const outFrames = Math.max(1, Math.floor(logits.dims[1] ?? frames));
        const logitsData = readFloat32(logits);
        const vocabSize = config.vocabularySize;

        const decoder = new CtcDecoder({
          blankId: config.tokenizer.blankTokenId,
          vocabSize,
          tokenizer,
        });
        const decoded = decoder.decodeFromLogits(
          logitsData.subarray(0, outFrames * vocabSize),
          outFrames,
          { outFrames },
        );

        return {
          utteranceText: decoded.text,
          isFinal: true,
          ctc: {
            frameIds: decoded.frameIds,
            collapsedIds: decoded.collapsedIds,
            secondsPerFrame,
            tokenSpans: decoded.tokenSpans,
            sentences: decoded.sentences,
          },
          metrics: {
            totalMs: roundMetric(nowMs() - started),
            encoderFrameCount: outFrames,
            decodeIterations: outFrames,
          },
        };
      } finally {
        featureTensor.dispose?.();
        maskTensor.dispose?.();
        disposeOrtOutputs(outputs);
      }
    },
    dispose(): void {
      releaseOrtSession(session);
    },
  };
}

export interface ParakeetCtcLoadedModel {
  readonly config: ParakeetCtcModelConfig;
  readonly backendId: string;
  transcribe(audio: Float32Array): Promise<LasrCtcNativeTranscript>;
  dispose(): void;
}

const CLASSIFICATION = {
  family: 'parakeet-ctc',
  ecosystem: 'nemo',
  processor: 'parakeet-mel-80',
  encoder: 'parakeet-ctc-encoder',
  decoder: 'ctc',
  topology: 'ctc',
  task: 'asr',
} as const;

/** Registers the family so built-in loads resolve `parakeet-ctc` requests. */
export function createParakeetCtcModelFamily(): SpeechModelFactory<
  ParakeetCtcModelOptions,
  LasrCtcTranscriptionOptions,
  LasrCtcNativeTranscript
> {
  return {
    family: 'parakeet-ctc',
    classification: CLASSIFICATION,
    supportedBackendIds: ['wasm', 'webgpu', 'native-cpu'],
    supports(modelId: string): boolean {
      const value = modelId.toLowerCase();
      return value.includes('parakeet-ctc') || value.includes('ctc-110m');
    },
    async createModel(
      request,
      context,
    ): Promise<
      SpeechModel<ParakeetCtcModelOptions, LasrCtcTranscriptionOptions, LasrCtcNativeTranscript>
    > {
      let loaded: ParakeetCtcLoadedModel | undefined;
      let disposed = false;

      return {
        info: {
          modelId: request.modelId,
          family: 'parakeet-ctc',
          displayName: request.modelId,
          languages: DEFAULT_PARAKEET_CTC_CONFIG.languages,
          backendId: context.backend.id,
        } as never,
        backend: context.backend,
        loadOptions: request.options,
        async createSession(): Promise<
          SpeechSession<LasrCtcTranscriptionOptions, LasrCtcNativeTranscript>
        > {
          loaded ??= await loadParakeetCtcFromOptions(
            (request.options ?? {}) as ParakeetCtcModelOptions,
          );

          return {
            async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
              input: AudioInputLike,
              options?: LasrCtcTranscriptionOptions & { readonly responseFlavor?: TFlavor },
            ): Promise<TranscriptResponse<LasrCtcNativeTranscript, TFlavor>> {
              const pcm = normalizePcmInput(input, {
                sampleRate: DEFAULT_PARAKEET_CTC_CONFIG.sampleRate,
              });
              const mono = pcm.channels[0];
              if (!mono) {
                throw new Error('Parakeet CTC transcription requires mono audio.');
              }
              const native = await loaded!.transcribe(mono);
              const canonical = mapLasrCtcNativeToCanonical(native, CLASSIFICATION, {
                detail: options?.detail ?? 'segments',
                detailLevel: options?.detail ?? 'segments',
                isFinal: native.isFinal,
              } as never);

              if (options?.responseFlavor === 'canonical+native') {
                return { canonical, native } as TranscriptResponse<
                  LasrCtcNativeTranscript,
                  TFlavor
                >;
              }

              return canonical as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
            },
            dispose(): void {
              if (!disposed) {
                disposed = true;
                loaded?.dispose();
                loaded = undefined;
              }
            },
          };
        },
        async dispose(): Promise<void> {
          if (!disposed) {
            disposed = true;
            loaded?.dispose();
            loaded = undefined;
          }
        },
      };
    },
  };
}
