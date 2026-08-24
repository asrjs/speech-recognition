import { describe, expect, it } from 'vitest';
import { whisperGreedyDecode, type WhisperCoreSession } from '../src/models/whisper-seq2seq/core.js';

function makeSession(): WhisperCoreSession {
  const vocabSize = 8;
  const initLogits = new Float32Array(4 * vocabSize);
  initLogits[(4 - 1) * vocabSize + 2] = 8;

  let stepCount = 0;
  return {
    async runInit() {
      return {
        logits: initLogits,
        vocabSize,
        presentKv: {},
      };
    },
    async runStep() {
      const logits = new Float32Array(vocabSize);
      stepCount += 1;
      logits[stepCount >= 2 ? 5 : 3] = 8;
      return {
        logits,
        vocabSize,
        presentKv: {},
      };
    },
  };
}

const baseOptions = {
  promptTokens: [50258, 50259, 50359, 50363],
  encoderOutput: new Float32Array(1),
  encoderDims: [1, 1, 1],
  eosTokenId: 5,
  maxNewTokens: 8,
} as const;

describe('whisperGreedyDecode scoring', () => {
  it('does not compute or return a score unless requested', async () => {
    const result = await whisperGreedyDecode(makeSession(), baseOptions);

    expect(result.tokens).toEqual([2, 3, 5]);
    expect(result.score).toBeUndefined();
  });

  it('tracks cumulative log probability when best-of scoring needs it', async () => {
    const result = await whisperGreedyDecode(makeSession(), {
      ...baseOptions,
      trackScore: true,
    });

    expect(result.tokens).toEqual([2, 3, 5]);
    expect(result.score).toBeTypeOf('number');
    expect(result.score).toBeLessThanOrEqual(0);
  });

  it('collects selected-sequence quality traces only when requested', async () => {
    const plain = await whisperGreedyDecode(makeSession(), baseOptions);
    const tracked = await whisperGreedyDecode(makeSession(), {
      ...baseOptions,
      trackQuality: true,
    });

    expect(plain.tokenTraces).toBeUndefined();
    expect(tracked.tokenTraces?.map((trace) => trace.tokenId)).toEqual(tracked.tokens);
    expect(tracked.tokenTraces?.every((trace) => trace.logProb <= 0)).toBe(true);
    expect(tracked.tokenTraces?.every((trace) => trace.entropy >= 0)).toBe(true);
  });
});
