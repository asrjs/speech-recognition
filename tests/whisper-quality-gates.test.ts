/**
 * Tests for Whisper quality gates.
 * Phase 2: compression ratio, log probability, no-speech, entropy.
 *
 * All gates are pure functions — no ONNX dependency.
 */

import { describe, it, expect } from 'vitest';
import {
  compressionRatioGate,
  logProbGate,
  noSpeechGate,
  entropyGate,
  evaluateGates,
} from '../src/models/whisper-seq2seq/quality-gates.js';
import type { QualityGate } from '../src/models/whisper-seq2seq/enhanced-types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Make a Float32Array logit vector where token `targetId` has the highest value. */
function peakedLogits(vocabSize: number, targetId: number, peak: number = 10.0): Float32Array {
  const logits = new Float32Array(vocabSize);
  logits[targetId] = peak;
  return logits;
}

/** Make a Float32Array logit vector with uniform random-ish values. */
function uniformLogits(vocabSize: number): Float32Array {
  const logits = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    logits[i] = 1.0; // all equal → uniform softmax
  }
  return logits;
}

// ---------------------------------------------------------------------------
// Compression Ratio Gate
// ---------------------------------------------------------------------------

describe('compressionRatioGate', () => {
  const threshold = 2.4;
  const gate = compressionRatioGate(threshold);

  it('accepts normal text (low compression ratio)', () => {
    const result = gate('hello world this is a normal sentence', [], [], 0);
    expect(result.verdict).toBe('accept');
    expect(result.compressionRatio).toBeDefined();
    expect(result.compressionRatio!).toBeLessThan(threshold);
  });

  it('rejects highly repetitive text (high compression ratio)', () => {
    const repeated = 'the the the the the the the the the the ' +
      'the the the the the the the the the the the the the the the the the the the';
    const result = gate(repeated, [], [], 0);
    expect(result.verdict).toBe('reject');
    expect(result.compressionRatio!).toBeGreaterThan(threshold);
    expect(result.reason).toContain('compression_ratio');
  });

  it('returns compressionRatio in result', () => {
    const result = gate('hello', [], [], 0);
    expect(result.compressionRatio).toBeDefined();
    expect(typeof result.compressionRatio).toBe('number');
  });
});

// ---------------------------------------------------------------------------
// Log Probability Gate
// ---------------------------------------------------------------------------

describe('logProbGate', () => {
  const threshold = -1.0;
  const gate = logProbGate(threshold);

  it('accepts high-confidence decode', () => {
    // token 1 chosen at position 0; logits highly peaked at token 1
    const logits = [peakedLogits(100, 1, 10.0)];
    const result = gate('', [1], logits, 100);
    expect(result.verdict).toBe('accept');
    expect(result.avgLogProb).toBeGreaterThan(threshold);
  });

  it('rejects low-confidence decode', () => {
    // Uniform logits → low log prob
    const logits = [uniformLogits(100)];
    const result = gate('', [0], logits, 100);
    expect(result.verdict).toBe('reject');
    expect(result.avgLogProb!).toBeLessThan(threshold);
  });

  it('returns avgLogProb in result', () => {
    const logits = [peakedLogits(100, 42, 10.0)];
    const result = gate('', [42], logits, 100);
    expect(result.avgLogProb).toBeDefined();
    expect(typeof result.avgLogProb).toBe('number');
  });

  it('computes average over multiple tokens', () => {
    const logits = [
      peakedLogits(100, 5, 10.0),
      peakedLogits(100, 10, 10.0),
      peakedLogits(100, 15, 10.0),
    ];
    const tokens = [5, 10, 15];
    const result = gate('', tokens, logits, 100);
    expect(result.avgLogProb).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// No-Speech Gate
// ---------------------------------------------------------------------------

describe('noSpeechGate', () => {
  const noSpeechTokenId = 50362;
  const threshold = 0.6;
  const logProbThreshold = -1.0;
  const gate = noSpeechGate(threshold, logProbThreshold);

  it('accepts when no-speech probability is low', () => {
    // No-speech token probability is tiny
    const logits = new Float32Array(52000);
    logits[42] = 10.0;  // high prob for token 42
    // token 50362 has 0 → exp(0)=1 → very small probability among 52k
    const result = gate('', [42], [logits], 52000);
    expect(result.verdict).toBe('accept');
  });

  it('flags no_speech when no-speech prob > threshold and avgLogProb < logProbThreshold', () => {
    // The dual condition in faster-whisper:
    //   noSpeechProb = softmax(first_logits)[50362] > 0.6  AND
    //   avgLogProb (over generated tokens) < -1.0
    // This needs 2+ tokens: first position has high no-speech probability,
    // while generated tokens have low confidence.
    const vocabSize = 52000;

    // First token logits: no-speech boosted but not dominant → moderate prob
    const firstLogits = new Float32Array(vocabSize);
    firstLogits.fill(0.0);
    firstLogits[noSpeechTokenId] = 12.0; // ~75% prob for no-speech, well above 0.6

    // Generated token logits: very flat → low logprob
    const genLogits = uniformLogits(vocabSize);

    const logits = [firstLogits, genLogits];
    const tokens = [noSpeechTokenId, 42]; // first=tried no-speech, second=random token
    const result = gate('', tokens, logits, vocabSize);
    expect(result.verdict).toBe('no_speech');
    expect(result.noSpeechProb).toBeDefined();
    expect(result.noSpeechProb!).toBeGreaterThan(threshold);
  });

  it('computes no-speech probability from logits', () => {
    const logits = peakedLogits(52000, noSpeechTokenId, 5.0);
    const result = gate('', [noSpeechTokenId], [logits], 52000);
    expect(result.noSpeechProb).toBeDefined();
    expect(typeof result.noSpeechProb).toBe('number');
    expect(result.noSpeechProb!).toBeGreaterThan(0);
    expect(result.noSpeechProb!).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------------------
// Entropy Gate
// ---------------------------------------------------------------------------

describe('entropyGate', () => {
  const threshold = 2.4;
  const gate = entropyGate(threshold);

  it('accepts focused logits (low entropy)', () => {
    const logits = [peakedLogits(100, 1, 10.0)];
    const result = gate('', [], logits, 100);
    expect(result.verdict).toBe('accept');
    expect(result.entropy!).toBeLessThan(threshold);
  });

  it('rejects uniform logits (high entropy)', () => {
    const logits = [uniformLogits(100)];
    const result = gate('', [], logits, 100);
    expect(result.verdict).toBe('reject');
    expect(result.entropy!).toBeGreaterThan(threshold);
  });

  it('rejects very high-entropy distribution', () => {
    const vocabSize = 100;
    const logits = uniformLogits(vocabSize);
    // Shannon entropy for uniform over N: ln(N) nats
    const expectedEntropy = Math.log(vocabSize); // ~4.605 for N=100
    const result = gate('', [], [logits], vocabSize);
    expect(result.verdict).toBe('reject');
    expect(result.entropy!).toBeCloseTo(expectedEntropy, 0);
  });

  it('returns entropy in result', () => {
    const logits = [peakedLogits(100, 5, 10.0)];
    const result = gate('', [], logits, 100);
    expect(result.entropy).toBeDefined();
    expect(typeof result.entropy).toBe('number');
  });
});

// ---------------------------------------------------------------------------
// Composite Gate Runner
// ---------------------------------------------------------------------------

describe('evaluateGates', () => {
  it('returns accept when all gates pass', () => {
    const gate1: QualityGate = () => ({ verdict: 'accept' });
    const gate2: QualityGate = () => ({ verdict: 'accept' });
    const result = evaluateGates('hello', [1, 2, 3], [], 100, [gate1, gate2]);
    expect(result.verdict).toBe('accept');
  });

  it('stops at first reject (short-circuit)', () => {
    let gate2Called = false;
    const gate1: QualityGate = () => ({ verdict: 'reject', reason: 'low_prob' });
    const gate2: QualityGate = () => { gate2Called = true; return { verdict: 'accept' }; };
    const result = evaluateGates('hello', [1, 2, 3], [], 100, [gate1, gate2]);
    expect(result.verdict).toBe('reject');
    expect(result.reason).toBe('low_prob');
    expect(gate2Called).toBe(false);
  });

  it('handles no_speech verdict', () => {
    const gate: QualityGate = () => ({
      verdict: 'no_speech',
      noSpeechProb: 0.8,
      reason: 'no_speech_detected',
    });
    const result = evaluateGates('', [], [], 100, [gate]);
    expect(result.verdict).toBe('no_speech');
    expect(result.noSpeechProb).toBe(0.8);
  });

  it('accepts when gates array is empty', () => {
    const result = evaluateGates('hello', [1, 2, 3], [], 100, []);
    expect(result.verdict).toBe('accept');
  });
});
