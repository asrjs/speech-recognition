import { CtcDecoder } from '../../ctc/index.js';
import { normalizePcmInput } from '../../audio/index.js';
import { normalizeWav2Vec2Waveform } from '../wav2vec2/executor.js';
import {
  createOrtSession,
  disposeOrtOutputs,
  initOrt,
  releaseOrtSession,
  type OrtModuleLike,
  type OrtTensorLike,
} from '../lasr-ctc/ort.js';
import type { LasrCtcNativeTranscript } from '../lasr-ctc/types.js';
import { OmnilingualCtcTokenizer } from './tokenizer.js';

/** Meta Omnilingual's wav2vec2 waveform LayerNorm contract. */
export const DEFAULT_OMNILINGUAL_CTC_NORMALIZATION_EPSILON = 1e-5;

export interface OmnilingualCtcOptions {
  readonly modelUrl: string;
  readonly tokensUrl: string;
  readonly externalDataUrl?: string;
  readonly externalDataFilename?: string;
  readonly sampleRate?: number;
  /** Population-variance epsilon used by the raw-waveform LayerNorm frontend. */
  readonly normalizationEpsilon?: number;
  readonly vocabularySize?: number;
  readonly blankId?: number;
  readonly backendId?: 'wasm' | 'webgpu';
  readonly cpuThreads?: number;
  readonly wasmPaths?: string;
  readonly signal?: AbortSignal | null;
}

export interface OmnilingualCtcLoadedModel {
  readonly vocabularySize: number;
  transcribe(pcm: Float32Array): Promise<LasrCtcNativeTranscript>;
  dispose(): void;
}

function readFloat32(tensor: OrtTensorLike): Float32Array {
  const data = tensor.data;
  if (data instanceof Float32Array) {
    return data;
  }

  return Float32Array.from(
    Array.from(data as unknown as { length: number; [index: number]: number }),
  );
}

/**
 * Loads an omnilingual wav2vec2-line CTC graph.
 *
 * The graph takes raw 16 kHz samples as `x [batch, num_samples]` with the
 * convolutional feature extractor inside, and returns
 * `logits [batch, frames, vocab]`. Per-utterance zero-mean unit-variance
 * normalization is a caller responsibility, as in the wav2vec2 family, and the
 * graph exposes no length input so long audio is handled by the caller through
 * windowing.
 */
export async function loadOmnilingualCtc(
  options: OmnilingualCtcOptions,
): Promise<OmnilingualCtcLoadedModel> {
  const backendId = options.backendId ?? 'wasm';
  const ort: OrtModuleLike = await initOrt(backendId, {
    wasmPaths: options.wasmPaths,
    cpuThreads: options.cpuThreads,
    signal: options.signal ?? null,
  });
  const session = await createOrtSession(ort, options.modelUrl, {
    backendId,
    externalDataUrl: options.externalDataUrl,
    externalDataPath: options.externalDataFilename,
    signal: options.signal ?? null,
  });
  const tokenizer = await OmnilingualCtcTokenizer.fromUrl(options.tokensUrl, options.signal);
  const vocabularySize = options.vocabularySize ?? tokenizer.vocabSize;
  const blankId = options.blankId ?? tokenizer.blankId;
  const decoder = new CtcDecoder({ blankId, vocabSize: vocabularySize, tokenizer });

  return {
    vocabularySize,
    async transcribe(pcm: Float32Array): Promise<LasrCtcNativeTranscript> {
      const normalized = normalizeWav2Vec2Waveform(
        pcm,
        options.normalizationEpsilon ?? DEFAULT_OMNILINGUAL_CTC_NORMALIZATION_EPSILON,
      );
      const input = new ort.Tensor('float32', normalized, [1, normalized.length]);
      let outputs: Record<string, OrtTensorLike> | undefined;
      try {
        outputs = await session.run({ x: input });
        const logits = (outputs.logits ?? Object.values(outputs)[0]) as OrtTensorLike;
        const frames = Math.max(1, Number(logits.dims[1] ?? 0));
        const data = readFloat32(logits);
        const decoded = decoder.decodeFromLogits(
          data.subarray(0, frames * vocabularySize),
          frames,
          { outFrames: frames },
        );

        return {
          utteranceText: decoded.text,
          isFinal: true,
          ctc: {
            frameIds: decoded.frameIds,
            collapsedIds: decoded.collapsedIds,
            secondsPerFrame: decoded.secondsPerFrame ?? 0,
            tokenSpans: decoded.tokenSpans,
            sentences: decoded.sentences,
          },
        };
      } finally {
        input.dispose?.();
        disposeOrtOutputs(outputs);
      }
    },
    dispose(): void {
      releaseOrtSession(session);
    },
  };
}

export interface OmnilingualCtcTranscribeInput {
  readonly audio: Float32Array;
  readonly sampleRate?: number;
}

export async function transcribeOmnilingualCtc(
  model: OmnilingualCtcLoadedModel,
  input: OmnilingualCtcTranscribeInput,
): Promise<LasrCtcNativeTranscript> {
  const normalized = normalizePcmInput(input.audio, { sampleRate: input.sampleRate ?? 16000 });
  const mono = normalized.channels[0];
  if (!mono) {
    throw new Error('Omnilingual CTC transcription requires mono audio.');
  }

  return await model.transcribe(mono);
}
