import { describe, expect, it } from 'vitest';
import { whisperBeamDecode, type WhisperCoreSession } from '../src/models/whisper-seq2seq/core.js';

const VOCAB_SIZE = 32;
const EOS_TOKEN_ID = 31;
const baseOptions = {
  promptTokens: [50258, 50259, 50359, 50363],
  encoderOutput: new Float32Array(1),
  encoderDims: [1, 1, 1],
  eosTokenId: EOS_TOKEN_ID,
  maxNewTokens: 12,
} as const;

interface BeamBenchmarkStats {
  readonly singleCalls: number;
  readonly batchCalls: number;
  readonly batchWidths: readonly number[];
}

function createBeamBenchmarkSession(): {
  readonly session: WhisperCoreSession;
  readonly stats: BeamBenchmarkStats;
} {
  let singleCalls = 0;
  let batchCalls = 0;
  const batchWidths: number[] = [];

  const logitsForToken = (tokenId: number): Float32Array => {
    const logits = new Float32Array(VOCAB_SIZE);
    logits.fill(-20);
    const primary = (tokenId * 3 + 1) % (EOS_TOKEN_ID - 1);
    const secondary = (primary + 1) % (EOS_TOKEN_ID - 1);
    logits[primary] = 7;
    logits[secondary] = 6.8;
    // Make EOS available on several paths without forcing every beam to finish
    // at the same step. This exercises active/finished candidate separation.
    if (tokenId % 3 === 0) logits[EOS_TOKEN_ID] = 6.5;
    return logits;
  };

  const makeStepResult = (tokenId: number) => ({
    logits: logitsForToken(tokenId),
    vocabSize: VOCAB_SIZE,
    presentKv: { marker: new Float32Array([tokenId]) },
  });

  const session: WhisperCoreSession = {
    async runInit() {
      const logits = new Float32Array(VOCAB_SIZE);
      logits.fill(-20);
      for (let tokenId = 2; tokenId <= 8; tokenId++) {
        logits[tokenId] = 8 - (tokenId - 2) * 0.2;
      }
      return {
        logits,
        vocabSize: VOCAB_SIZE,
        presentKv: { marker: new Float32Array([0]) },
      };
    },
    async runStep(tokenId) {
      singleCalls += 1;
      return makeStepResult(tokenId);
    },
    async runStepBatch(tokenIds) {
      batchCalls += 1;
      batchWidths.push(tokenIds.length);
      return tokenIds.map((tokenId) => makeStepResult(tokenId));
    },
  };

  return {
    session,
    get stats() {
      return { singleCalls, batchCalls, batchWidths: [...batchWidths] };
    },
  };
}

describe('Whisper beam batching benchmark contract', () => {
  for (const beamSize of [2, 3, 5]) {
    it(`preserves stable tokens and reduces decoder calls at beam size ${beamSize}`, async () => {
      const stable = createBeamBenchmarkSession();
      const stableResult = await whisperBeamDecode(stable.session, {
        ...baseOptions,
        beamSize,
        patience: 1,
      });

      const batched = createBeamBenchmarkSession();
      const batchedResult = await whisperBeamDecode(batched.session, {
        ...baseOptions,
        beamSize,
        patience: 1,
        experimentalBatchedBeam: true,
      });

      // This is a deterministic benchmark guard: wall-clock thresholds are
      // intentionally avoided because CI and browser GPU clocks vary.
      expect(batchedResult.tokens).toEqual(stableResult.tokens);
      expect(stable.stats.singleCalls).toBeGreaterThan(0);
      expect(batched.stats.singleCalls).toBe(0);
      expect(batched.stats.batchCalls).toBeGreaterThan(0);
      expect(batched.stats.batchWidths.some((width) => width > 1)).toBe(true);
      expect(batched.stats.batchCalls).toBeLessThan(stable.stats.singleCalls);
    });
  }
});
