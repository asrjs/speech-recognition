import { CtcDecoder } from '../../ctc/index.js';
import { normalizePcmInput } from '../../audio/index.js';
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

export interface OmnilingualCtcOptions {
  readonly modelUrl: string;
  readonly tokensUrl: string;
  readonly externalDataUrl?: string;
  readonly externalDataFilename?: string;
  readonly sampleRate?: number;
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
 * `logits [batch, frames, vocab]`. This exported graph consumes unnormalized
 * PCM samples. Applying the checkpoint's separate waveform normalization here
 * changes its predictions. Long audio is windowed by the caller.
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
      const input = new ort.Tensor('float32', pcm, [1, pcm.length]);
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
