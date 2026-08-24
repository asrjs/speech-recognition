/**
 * Tests for standalone quality/ module.
 * Phase A: relocate tests from whisper-quality-gates + whisper-temperature-fallback.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  compressionRatioGate,
  logProbGate,
  entropyGate,
  noSpeechGate,
  evaluateGates,
  withTemperatureFallback,
  type QualityGate,
  type QualityGateResult,
} from '../src/quality/index.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function peakedLogits(vocabSize: number, targetId: number, peak: number = 10.0): Float32Array {
  const logits = new Float32Array(vocabSize);
  logits[targetId] = peak;
  return logits;
}

function uniformLogits(vocabSize: number): Float32Array {
  const logits = new Float32Array(vocabSize);
  logits.fill(1.0);
  return logits;
}

// ---------------------------------------------------------------------------
// Compression Ratio
// ---------------------------------------------------------------------------

describe('compressionRatioGate', () => {
  const gate = compressionRatioGate(2.4);

  it('accepts normal text', () => {
    const result = gate('hello world this is a normal sentence', [], [], 0);
    expect(result.verdict).toBe('accept');
    expect(result.compressionRatio!).toBeLessThan(2.4);
  });

  it('rejects repetitive text', () => {
    const repeated = 'the the the the the the the the the the '.repeat(3);
    const result = gate(repeated, [], [], 0);
    expect(result.verdict).toBe('reject');
  });
});

// ---------------------------------------------------------------------------
// Log Probability
// ---------------------------------------------------------------------------

describe('logProbGate', () => {
  const gate = logProbGate(-1.0);

  it('accepts high-confidence', () => {
    const result = gate('', [1], [peakedLogits(100, 1, 10.0)], 100);
    expect(result.verdict).toBe('accept');
  });

  it('rejects low-confidence', () => {
    const result = gate('', [0], [uniformLogits(100)], 100);
    expect(result.verdict).toBe('reject');
  });

  it('evaluates selected-sequence traces without full-vocabulary logits', () => {
    const rejected = gate('', [1, 2], [], 100, {
      tokenTraces: [
        { tokenId: 1, logProb: -2.5, entropy: 0.2 },
        { tokenId: 2, logProb: -2.5, entropy: 0.2 },
      ],
    });
    expect(rejected.verdict).toBe('reject');
    expect(rejected.avgLogProb).toBeCloseTo(-2.5);

    const accepted = gate('', [1, 2], [], 100, {
      tokenTraces: [
        { tokenId: 1, logProb: -0.1, entropy: 0.2 },
        { tokenId: 2, logProb: -0.2, entropy: 0.2 },
      ],
    });
    expect(accepted.verdict).toBe('accept');
  });
});

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

describe('entropyGate', () => {
  const gate = entropyGate(2.4);

  it('accepts focused logits', () => {
    const result = gate('', [], [peakedLogits(100, 1, 10.0)], 100);
    expect(result.verdict).toBe('accept');
  });

  it('rejects uniform logits', () => {
    const result = gate('', [], [uniformLogits(100)], 100);
    expect(result.verdict).toBe('reject');
    expect(result.entropy!).toBeGreaterThan(2.4);
  });

  it('evaluates selected-sequence entropy traces without logits', () => {
    const rejected = gate('', [1], [], 100, {
      tokenTraces: [{ tokenId: 1, logProb: -0.1, entropy: 3.1 }],
    });
    expect(rejected.verdict).toBe('reject');

    const accepted = gate('', [1], [], 100, {
      tokenTraces: [{ tokenId: 1, logProb: -0.1, entropy: 0.4 }],
    });
    expect(accepted.verdict).toBe('accept');
  });
});

// ---------------------------------------------------------------------------
// No-Speech
// ---------------------------------------------------------------------------

describe('noSpeechGate', () => {
  const gate = noSpeechGate(0.6, -1.0);

  it('accepts low no-speech probability', () => {
    const logits = new Float32Array(52000);
    logits[42] = 10.0;
    const result = gate('', [42], [logits], 52000);
    expect(result.verdict).toBe('accept');
  });

  it('flags no_speech with dual condition', () => {
    const vocabSize = 52000;
    const firstLogits = new Float32Array(vocabSize);
    firstLogits.fill(0.0);
    firstLogits[50362] = 12.0; // ~75% prob
    const genLogits = uniformLogits(vocabSize);
    const result = gate('', [50362, 42], [firstLogits, genLogits], vocabSize);
    expect(result.verdict).toBe('no_speech');
  });

  it('uses raw decoder-init logits and a model-provided token ID', () => {
    const processedFirst = new Float32Array(16);
    processedFirst[1] = 10.0;
    const rawInit = new Float32Array(16);
    rawInit[7] = 12.0;
    const result = gate('', [1, 2], [processedFirst, uniformLogits(16)], 16, {
      noSpeechLogits: rawInit,
      noSpeechTokenId: 7,
    });

    expect(result.verdict).toBe('no_speech');
    expect(result.noSpeechProb!).toBeGreaterThan(0.6);
  });
});

// ---------------------------------------------------------------------------
// Temperature Fallback
// ---------------------------------------------------------------------------

describe('temperature fallback', () => {
  it('returns first result when gate accepts', async () => {
    const fn = vi.fn(async () => ({
      result: { text: 'hello' },
      text: 'hello', tokens: [1], logits: [] as Float32Array[], vocabSize: 100,
    }));
    const acceptGate: QualityGate = () => ({ verdict: 'accept' });
    const result = await withTemperatureFallback(fn, [acceptGate]);
    expect(result.attempts).toBe(1);
  });

  it('counts decode attempts, not gate evaluations', async () => {
    const fn = vi.fn(async () => ({
      result: { text: 'hello' },
      text: 'hello', tokens: [1], logits: [] as Float32Array[], vocabSize: 100,
    }));
    const acceptGate: QualityGate = () => ({ verdict: 'accept' });
    const result = await withTemperatureFallback(fn, [acceptGate, acceptGate]);
    expect(result.attempts).toBe(1);
    expect(result.gateResults).toHaveLength(2);
  });

  it('retries on reject', async () => {
    let call = 0;
    const fn = vi.fn(async () => {
      call++;
      return {
        result: { text: `attempt${call}` },
        text: call === 2 ? 'good text' : 'bad',
        tokens: [call],
        logits: [] as Float32Array[],
        vocabSize: 100,
      };
    });
    const goodGate: QualityGate = (text: string): QualityGateResult =>
      text.includes('good') ? { verdict: 'accept' } : { verdict: 'reject', reason: 'bad' };
    const result = await withTemperatureFallback(fn, [goodGate], [0.0, 0.2]);
    expect(result.attempts).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Composite evaluateGates
// ---------------------------------------------------------------------------

describe('evaluateGates', () => {
  it('returns accept when all pass', () => {
    const gate: QualityGate = () => ({ verdict: 'accept' });
    expect(evaluateGates('', [], [], 100, [gate, gate]).verdict).toBe('accept');
  });

  it('short-circuits on first reject', () => {
    let secondCalled = false;
    const gate1: QualityGate = () => ({ verdict: 'reject', reason: 'fail' });
    const gate2: QualityGate = () => { secondCalled = true; return { verdict: 'accept' }; };
    const result = evaluateGates('', [], [], 100, [gate1, gate2]);
    expect(result.verdict).toBe('reject');
    expect(secondCalled).toBe(false);
  });

  it('accepts empty gates', () => {
    expect(evaluateGates('', [], [], 100, []).verdict).toBe('accept');
  });
});

describe('selected-sequence token traces', () => {
  it('rejects low logprob from scalar traces without logits', () => {
    const result = logProbGate(-1.0)('', [1, 2], [], 100, {
      tokenTraces: [
        { tokenId: 1, logProb: -2.4, entropy: 0.3 },
        { tokenId: 2, logProb: -2.1, entropy: 0.4 },
      ],
    });
    expect(result.verdict).toBe('reject');
    expect(result.avgLogProb).toBeCloseTo(-2.25, 5);
  });

  it('rejects high entropy from scalar traces without logits', () => {
    const result = entropyGate(2.4)('', [1], [], 100, {
      tokenTraces: [
        { tokenId: 1, logProb: -0.2, entropy: 3.1 },
        { tokenId: 2, logProb: -0.3, entropy: 2.8 },
      ],
    });
    expect(result.verdict).toBe('reject');
    expect(result.entropy).toBeCloseTo(2.95, 5);
  });

  it('prefers traces over full-vocabulary logits', () => {
    const misleading = [uniformLogits(16)];
    const result = logProbGate(-1.0)('', [0], misleading, 16, {
      tokenTraces: [{ tokenId: 0, logProb: -0.05, entropy: 0.1 }],
    });
    expect(result.verdict).toBe('accept');
    expect(result.avgLogProb).toBeCloseTo(-0.05, 5);
  });
});

describe('withTemperatureFallback quality context', () => {
  it('forwards raw-logit context to each gate', async () => {
    const context = {
      noSpeechLogits: new Float32Array([0, 1]),
      noSpeechTokenId: 1,
    };
    let received: unknown;
    const gate: QualityGate = (_text, _tokens, _logits, _vocabSize, qualityContext) => {
      received = qualityContext;
      return { verdict: 'accept' };
    };

    await withTemperatureFallback(
      async () => ({
        result: { text: 'hello' },
        text: 'hello',
        tokens: [1],
        logits: [new Float32Array([0, 1])],
        vocabSize: 2,
        qualityContext: context,
      }),
      [gate],
    );

    expect(received).toBe(context);
  });
});
