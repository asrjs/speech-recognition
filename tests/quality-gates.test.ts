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
