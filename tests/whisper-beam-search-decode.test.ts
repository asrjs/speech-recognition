import { describe, expect, it, vi } from 'vitest';
import {
  whisperBeamDecode,
  whisperDecode,
  whisperGreedyDecode,
  type WhisperCoreSession,
} from '../src/models/whisper-seq2seq/core.js';

// Pure integration test for the beam-search decode loop.
// No ONNX runtime needed — the mock session exercises the full init→step logic.

const VOCAB_SIZE = 8;
const PROMPT_TOKENS = [50258, 50259, 50359, 50363];
const EOS_TOKEN_ID = 5;

function makeMockSession(): WhisperCoreSession {
  // Deterministic session that gives every beam the same logits shape and
  // always emits token 2 from init, then token 3, then EOS.
  return {
    async runInit() {
      const logits = new Float32Array(VOCAB_SIZE);
      logits[2] = 8.0;
      return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
    },
    async runStep() {
      const logits = new Float32Array(VOCAB_SIZE);
      logits[3] = 8.0;
      return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
    },
  };
}

function makeEosSession(eosAfter: number): WhisperCoreSession {
  // Emits token 2 from init, then token 3 for `eosAfter` steps, then EOS.
  let step = 0;
  return {
    async runInit() {
      const logits = new Float32Array(VOCAB_SIZE);
      logits[2] = 8.0;
      return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
    },
    async runStep() {
      step += 1;
      const logits = new Float32Array(VOCAB_SIZE);
      logits[step >= eosAfter ? EOS_TOKEN_ID : 3] = 8.0;
      return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
    },
  };
}

const baseOptions = {
  promptTokens: PROMPT_TOKENS,
  encoderOutput: new Float32Array(1),
  encoderDims: [1, 1, 1],
  eosTokenId: EOS_TOKEN_ID,
  maxNewTokens: 8,
} as const;

describe('whisperBeamDecode integration', () => {
  it('produces a token sequence ending in EOS', async () => {
    const result = await whisperBeamDecode(makeEosSession(2), {
      ...baseOptions,
      beamSize: 2,
      patience: 1,
    });

    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens[result.tokens.length - 1]).toBe(EOS_TOKEN_ID);
    // With eosAfter=2: init→2, step→3, step→EOS = [2, 3, 5]
    expect(result.tokens).toEqual([2, 3, EOS_TOKEN_ID]);
    expect(result.score).toBeTypeOf('number');
    expect(result.score).toBeLessThanOrEqual(0);
  });

  it('returns the same greedy path when beamSize is 1', async () => {
    const greedy = await whisperGreedyDecode(makeMockSession(), baseOptions);
    const beam = await whisperBeamDecode(makeMockSession(), {
      ...baseOptions,
      beamSize: 1,
      patience: 1,
    });

    expect(beam.tokens).toEqual(greedy.tokens);
  });

  it('whisperDecode dispatches to beam search when strategy is beam', async () => {
    const result = await whisperDecode(makeEosSession(1), {
      ...baseOptions,
      strategy: 'beam',
      beamSize: 2,
      patience: 1,
    });

    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens[result.tokens.length - 1]).toBe(EOS_TOKEN_ID);
  });

  it('whisperDecode dispatches to greedy by default', async () => {
    const result = await whisperDecode(makeMockSession(), baseOptions);

    expect(result.tokens).toEqual([2, 3, 3, 3, 3, 3, 3, 3]);
  });

  it('ignores bestOf when temperature is zero', async () => {
    let initCalls = 0;
    const session: WhisperCoreSession = {
      async runInit() {
        initCalls += 1;
        const logits = new Float32Array(VOCAB_SIZE);
        logits[2] = 8.0;
        return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
      },
      async runStep() {
        const logits = new Float32Array(VOCAB_SIZE);
        logits[EOS_TOKEN_ID] = 8.0;
        return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
      },
    };

    const result = await whisperDecode(session, {
      ...baseOptions,
      temperature: 0,
      bestOf: 3,
      maxNewTokens: 2,
    });

    expect(result.tokens).toEqual([2, EOS_TOKEN_ID]);
    expect(initCalls).toBe(1);
  });

  it('uses sampling instead of beam search when temperature is nonzero', async () => {
    const randomSpy = vi.spyOn(Math, 'random').mockReturnValue(0.5);
    let stepCalls = 0;
    const session: WhisperCoreSession = {
      async runInit() {
        const logits = new Float32Array(VOCAB_SIZE);
        logits.fill(-100);
        logits[2] = 8.0;
        logits[3] = 7.0;
        return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
      },
      async runStep() {
        stepCalls += 1;
        const logits = new Float32Array(VOCAB_SIZE);
        logits.fill(-100);
        logits[EOS_TOKEN_ID] = 8.0;
        return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
      },
    };

    try {
      const result = await whisperDecode(session, {
        ...baseOptions,
        strategy: 'beam',
        beamSize: 2,
        temperature: 0.4,
        maxNewTokens: 2,
      });

      expect(result.tokens).toEqual([2, EOS_TOKEN_ID]);
      expect(stepCalls).toBe(1);
    } finally {
      randomSpy.mockRestore();
    }
  });

  it('keeps KV caches aligned when completed beams are retained with patience', async () => {
    const seenActiveStepKv: Array<number | undefined> = [];
    const session: WhisperCoreSession = {
      async runInit() {
        const logits = new Float32Array(VOCAB_SIZE);
        logits[2] = 8.0;
        logits[3] = 7.0;
        return {
          logits,
          vocabSize: VOCAB_SIZE,
          presentKv: { marker: new Float32Array([0]) },
        };
      },
      async runStep(tokenId, pastKv) {
        const marker = pastKv.marker?.[0];
        const logits = new Float32Array(VOCAB_SIZE);
        if (tokenId === 2) {
          logits[EOS_TOKEN_ID] = 8.0;
          return { logits, vocabSize: VOCAB_SIZE, presentKv: { marker: new Float32Array([2]) } };
        }
        if (tokenId === 3) {
          logits[4] = 8.0;
          return { logits, vocabSize: VOCAB_SIZE, presentKv: { marker: new Float32Array([3]) } };
        }
        if (tokenId === 4) {
          seenActiveStepKv.push(marker);
          logits[EOS_TOKEN_ID] = 8.0;
          return { logits, vocabSize: VOCAB_SIZE, presentKv: { marker: new Float32Array([4]) } };
        }
        logits[EOS_TOKEN_ID] = 8.0;
        return { logits, vocabSize: VOCAB_SIZE, presentKv: { marker: new Float32Array([tokenId]) } };
      },
    };

    const result = await whisperBeamDecode(session, {
      ...baseOptions,
      beamSize: 2,
      patience: 2,
      maxNewTokens: 4,
    });

    expect(result.tokens[result.tokens.length - 1]).toBe(EOS_TOKEN_ID);
    expect(seenActiveStepKv).toEqual([3]);
  });

  it('uses batched beam step hook only when explicitly enabled', async () => {
    const makeSession = () => {
      let singleCalls = 0;
      let batchCalls = 0;
      const session: WhisperCoreSession = {
        async runInit() {
          const logits = new Float32Array(VOCAB_SIZE);
          logits[2] = 8.0;
          logits[3] = 7.5;
          return {
            logits,
            vocabSize: VOCAB_SIZE,
            presentKv: { marker: new Float32Array([0]) },
          };
        },
        async runStep(tokenId) {
          singleCalls += 1;
          const logits = new Float32Array(VOCAB_SIZE);
          logits[tokenId === 2 ? 4 : EOS_TOKEN_ID] = 8.0;
          return {
            logits,
            vocabSize: VOCAB_SIZE,
            presentKv: { marker: new Float32Array([tokenId]) },
          };
        },
        async runStepBatch(tokenIds) {
          batchCalls += 1;
          return tokenIds.map((tokenId) => {
            const logits = new Float32Array(VOCAB_SIZE);
            logits[tokenId === 2 ? 4 : EOS_TOKEN_ID] = 8.0;
            return {
              logits,
              vocabSize: VOCAB_SIZE,
              presentKv: { marker: new Float32Array([tokenId]) },
            };
          });
        },
      };
      return {
        session,
        get singleCalls() {
          return singleCalls;
        },
        get batchCalls() {
          return batchCalls;
        },
      };
    };

    const stable = makeSession();
    const stableResult = await whisperBeamDecode(stable.session, {
      ...baseOptions,
      beamSize: 2,
      patience: 1,
      maxNewTokens: 4,
    });

    const batched = makeSession();
    const batchedResult = await whisperBeamDecode(batched.session, {
      ...baseOptions,
      beamSize: 2,
      patience: 1,
      maxNewTokens: 4,
      experimentalBatchedBeam: true,
    });

    expect(batchedResult.tokens).toEqual(stableResult.tokens);
    expect(stable.batchCalls).toBe(0);
    expect(stable.singleCalls).toBeGreaterThan(1);
    expect(batched.batchCalls).toBeGreaterThan(0);
    expect(batched.singleCalls).toBeLessThan(stable.singleCalls);
  });

  it('throws if a batched beam step returns the wrong result count', async () => {
    const session: WhisperCoreSession = {
      async runInit() {
        const logits = new Float32Array(VOCAB_SIZE);
        logits[2] = 8.0;
        logits[3] = 7.5;
        return { logits, vocabSize: VOCAB_SIZE, presentKv: {} };
      },
      async runStep() {
        throw new Error('sequential fallback should not be used');
      },
      async runStepBatch() {
        return [];
      },
    };

    await expect(whisperBeamDecode(session, {
      ...baseOptions,
      beamSize: 2,
      patience: 1,
      maxNewTokens: 3,
      experimentalBatchedBeam: true,
    })).rejects.toThrow(/Batched Whisper beam step returned 0 results/);
  });
});
