/**
 * Tests for Whisper temperature fallback retry loop.
 * Phase 3: generic retry with escalating temperatures + quality gates.
 */

import { describe, it, expect, vi } from 'vitest';
import { withTemperatureFallback, DEFAULT_TEMPERATURES } from '../src/models/whisper-seq2seq/temperature-fallback.js';
import type { QualityGate, QualityGateResult } from '../src/models/whisper-seq2seq/enhanced-types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface MockDecodeResult {
  text: string;
  tokens: number[];
  temperature: number;
}

/** Always-accepting gate */
const acceptGate: QualityGate = () => ({ verdict: 'accept' });

/** Always-rejecting gate */
const rejectGate: QualityGate = () => ({ verdict: 'reject', reason: 'test_reject' });

/** Gate that accepts text containing a specific word. */
function wordGate(requiredWord: string): QualityGate {
  return (text: string): QualityGateResult => {
    if (text.includes(requiredWord)) {
      return { verdict: 'accept' };
    }
    return { verdict: 'reject', reason: `missing_${requiredWord}` };
  };
}

/** No-speech gate when text is empty. */
const emptyTextNoSpeech: QualityGate = (text: string): QualityGateResult => {
  if (text.length === 0) {
    return { verdict: 'no_speech', reason: 'silence' };
  }
  return { verdict: 'accept' };
};

/** Make a mock transcribe function */
function mockTranscribe(results: MockDecodeResult[]) {
  let callCount = 0;
  return vi.fn(async (temperature: number) => {
    const result = results[callCount] ?? results[results.length - 1]!;
    callCount++;
    return {
      result,
      text: result.text,
      tokens: result.tokens,
      logits: [] as Float32Array[],
      vocabSize: 100,
    };
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('DEFAULT_TEMPERATURES', () => {
  it('matches faster-whisper schedule', () => {
    expect(DEFAULT_TEMPERATURES).toEqual([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
  });
});

describe('withTemperatureFallback', () => {
  it('returns first result when gate accepts immediately', async () => {
    const fn = mockTranscribe([{ text: 'hello', tokens: [1, 2], temperature: 0.0 }]);
    const result = await withTemperatureFallback(fn, [acceptGate]);
    expect(result.result.text).toBe('hello');
    expect(result.temperature).toBe(0.0);
    expect(result.attempts).toBe(1);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('retries at next temperature when first is rejected', async () => {
    const results = [
      { text: 'bad output', tokens: [1], temperature: 0.0 },
      { text: 'good output', tokens: [1, 2], temperature: 0.2 },
    ];
    const fn = mockTranscribe(results);
    const gates = [wordGate('good')]; // only accept text containing 'good'
    const result = await withTemperatureFallback(fn, gates, [0.0, 0.2]);
    expect(result.result.text).toBe('good output');
    expect(result.temperature).toBe(0.2);
    expect(result.attempts).toBe(2);
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('returns last result when all temperatures are rejected', async () => {
    const results = [
      { text: 'a', tokens: [1], temperature: 0.0 },
      { text: 'b', tokens: [1], temperature: 0.2 },
      { text: 'c', tokens: [1], temperature: 0.4 },
    ];
    const fn = mockTranscribe(results);
    const temps = [0.0, 0.2, 0.4];
    const result = await withTemperatureFallback(fn, [rejectGate], temps);
    expect(result.result.text).toBe('c');
    expect(result.temperature).toBe(0.4);
    expect(result.attempts).toBe(3);
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it('returns immediately on no_speech verdict', async () => {
    const results = [
      { text: '', tokens: [], temperature: 0.0 },
      { text: 'should not reach', tokens: [1], temperature: 0.2 },
    ];
    const fn = mockTranscribe(results);
    const result = await withTemperatureFallback(fn, [emptyTextNoSpeech]);
    expect(result.result.text).toBe('');
    expect(result.temperature).toBe(0.0);
    expect(result.attempts).toBe(1);
    expect(result.gateResults[0]?.verdict).toBe('no_speech');
    expect(fn).toHaveBeenCalledTimes(1); // did not retry
  });

  it('uses default temperatures when none provided', async () => {
    const results = Array.from({ length: 7 }, (_, i) => ({
      text: `temp_${i}`,
      tokens: [i],
      temperature: i * 0.2,
    }));
    const fn = mockTranscribe(results);
    // Accept immediately, no retry needed
    const result = await withTemperatureFallback(fn, [acceptGate]);
    expect(result.attempts).toBe(1);
  });

  it('records gateResults for each attempt', async () => {
    const results = [
      { text: 'bad output', tokens: [1], temperature: 0.0 },
      { text: 'good output', tokens: [1], temperature: 0.2 },
    ];
    const fn = mockTranscribe(results);
    const gates = [wordGate('good')];
    const result = await withTemperatureFallback(fn, gates, [0.0, 0.2]);
    expect(result.gateResults).toHaveLength(2);
    expect(result.gateResults[0]?.verdict).toBe('reject');
    expect(result.gateResults[1]?.verdict).toBe('accept');
  });

  it('stops when temperatures exhausted', async () => {
    const results = [{ text: 'only', tokens: [1], temperature: 0.0 }];
    const fn = mockTranscribe(results);
    const temps = [0.0];
    const result = await withTemperatureFallback(fn, [rejectGate], temps);
    expect(result.attempts).toBe(1);
    expect(result.gateResults[0]?.verdict).toBe('reject');
  });

  it('returns correct temperature after partial retries', async () => {
    const results = [
      { text: 'nope', tokens: [1], temperature: 0.0 },
      { text: 'nope', tokens: [1], temperature: 0.2 },
      { text: 'yes', tokens: [1, 2], temperature: 0.4 },
    ];
    const fn = mockTranscribe(results);
    const gates = [wordGate('yes')];
    const result = await withTemperatureFallback(fn, gates, [0.0, 0.2, 0.4]);
    expect(result.temperature).toBe(0.4);
    expect(result.attempts).toBe(3);
  });
});
