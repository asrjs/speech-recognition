import { describe, expect, it } from 'vitest';
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
});
