import { describe, expect, it } from 'vitest';
import { PipelineAbortedError } from '../src/pipeline/composition.js';
import {
  whisperGreedyDecode,
  type WhisperCoreSession,
} from '../src/models/whisper-seq2seq/core.js';
import {
  splitGraphDecodeLoop,
  WhisperOnnxExecutor,
} from '../src/models/whisper-seq2seq/executor.js';
import { DEFAULT_WHISPER_CLASSIFICATION, parseWhisperSeq2SeqConfig } from '../src/models/whisper-seq2seq/config.js';
import type { OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';
import type { WhisperModelConfig } from '../src/models/whisper-seq2seq/generation-config.js';

const tinyConfig: WhisperModelConfig = {
  decoderLayers: 4,
  decoderAttentionHeads: 6,
  dModel: 384,
  headDim: 64,
  medianFilterWidth: 7,
};

class TrackingTensor implements OrtTensorLike<Float32Array> {
  disposed = 0;

  constructor(
    readonly type: string,
    readonly data: Float32Array | Uint16Array | BigInt64Array,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed += 1;
  }
}

function neverEosLogits(vocabSize: number, tokenId: number): Float32Array {
  const logits = new Float32Array(vocabSize);
  logits[tokenId] = 10;
  return logits;
}

describe('Whisper in-flight decode abort', () => {
  it('stops greedy decoder steps on abort and can decode again', async () => {
    const vocabSize = 8;
    const initLogits = new Float32Array(4 * vocabSize);
    initLogits[(4 - 1) * vocabSize + 2] = 8;
    let stepCount = 0;
    const session: WhisperCoreSession = {
      async runInit() {
        return { logits: initLogits, vocabSize, presentKv: {} };
      },
      async runStep() {
        stepCount += 1;
        const logits = neverEosLogits(vocabSize, 3);
        return { logits, vocabSize, presentKv: {} };
      },
    };
    const signal = { aborted: false };
    const options = {
      promptTokens: [50258, 50259, 50359, 50363],
      encoderOutput: new Float32Array(1),
      encoderDims: [1, 1, 1] as const,
      eosTokenId: 5,
      maxNewTokens: 8,
      signal,
    };

    const pending = whisperGreedyDecode(session, {
      ...options,
      onTokenLogits() {
        if (stepCount >= 1) signal.aborted = true;
      },
    });
    await expect(pending).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(stepCount).toBe(1);

    const completed = await whisperGreedyDecode(session, {
      ...options,
      signal: { aborted: false },
      maxNewTokens: 2,
    });
    expect(completed.tokens.length).toBeGreaterThan(0);
    expect(stepCount).toBe(2);
  });

  it('stops split-graph decode on abort and can decode again', async () => {
    let stepCount = 0;
    const signal = { aborted: false };
    const runInit = () => {
      const logits = neverEosLogits(10, 7);
      return {
        logits,
        vocabSize: 10,
        presentKv: { 'present.0.decoder.key': new Float32Array([1]) },
      };
    };
    const runStep = () => {
      stepCount += 1;
      if (stepCount >= 1) signal.aborted = true;
      return {
        logits: neverEosLogits(10, 7),
        vocabSize: 10,
        presentKv: { 'present.0.decoder.key': new Float32Array([stepCount]) },
      };
    };

    await expect(
      splitGraphDecodeLoop({
        promptTokens: [50258, 50259, 50359],
        encoderHiddenStates: new Float32Array(100),
        eosTokenId: 5,
        maxNewTokens: 20,
        modelConfig: tinyConfig,
        signal,
        runInit,
        runStep,
      }),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(stepCount).toBe(1);

    const completed = await splitGraphDecodeLoop({
      promptTokens: [50258, 50259, 50359],
      encoderHiddenStates: new Float32Array(100),
      eosTokenId: 5,
      maxNewTokens: 2,
      modelConfig: tinyConfig,
      signal: { aborted: false },
      runInit,
      runStep: () => ({
        logits: (() => {
          const logits = new Float32Array(10);
          logits[5] = 10;
          return logits;
        })(),
        vocabSize: 10,
        presentKv: {},
      }),
    });
    expect(completed.tokens.at(-1)).toBe(5);
  });

  it('disposes GPU-KV from an aborted greedy decode and remains usable', async () => {
    class Tensor extends TrackingTensor {
      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        super(type, data as Float32Array, dims);
      }
    }

    let initRuns = 0;
    let stepRuns = 0;
    let latestPresent: TrackingTensor | undefined;
    const encoderHidden = new TrackingTensor('float32', new Float32Array(8), [1, 2, 4]);
    const executor = new WhisperOnnxExecutor(
      'whisper-decode-abort',
      DEFAULT_WHISPER_CLASSIFICATION,
      parseWhisperSeq2SeqConfig('whisper-decode-abort', {
        maxSourcePositions: 4,
        maxTargetPositions: 8,
      }),
      'wasm',
      undefined,
    );
    const loaded = {
      ort: { Tensor },
      decoderInitSession: {
        async run() {
          initRuns += 1;
          const logits = new TrackingTensor(
            'float32',
            initRuns === 1 ? new Float32Array([0, 5, 1]) : new Float32Array([0, 0, 9]),
            [1, 1, 3],
          );
          latestPresent = new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]);
          return {
            logits,
            'present.0.decoder.key': latestPresent,
          };
        },
      },
      decoderStepSession: {
        inputNames: ['input_ids'],
        async run() {
          stepRuns += 1;
          return {
            logits: new TrackingTensor('float32', new Float32Array([0, 0, 9]), [1, 1, 3]),
            'present.0.decoder.key': new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]),
          };
        },
      },
    };
    const decode = (signal: { aborted: boolean }) =>
      (
        executor as unknown as {
          runGreedyGpuKvDecode: (params: Record<string, unknown>) => Promise<{ tokens: readonly number[] }>;
        }
      ).runGreedyGpuKvDecode({
        loaded,
        encoderHiddenStates: encoderHidden,
        promptTokens: [50258],
        eosTokenId: 2,
        maxNewTokens: 8,
        signal,
        onDecoderInitLogits() {
          if (initRuns === 1) signal.aborted = true;
        },
      });

    await expect(decode({ aborted: false })).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(stepRuns).toBe(0);
    expect(latestPresent?.disposed).toBeGreaterThan(0);
    expect(encoderHidden.disposed).toBe(0);

    const completed = await decode({ aborted: false });
    expect(initRuns).toBe(2);
    expect(stepRuns).toBe(0);
    expect(completed.tokens).toEqual([2]);
  });
});
