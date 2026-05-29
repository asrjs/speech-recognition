import { describe, expect, it } from 'vitest';
import { WhisperTimestampLogitProcessor } from '../src/models/whisper-seq2seq/processors.js';

const VOCAB_SIZE = 51865;
const EOS = 50257;
const NO_TIMESTAMPS = 50363;
const TIMESTAMP_BEGIN = 50364;
const TIMESTAMP_END = 51864;
const SUPPRESS_TOKENS = [1, 2, 7, 50257];
const BEGIN_SUPPRESS_TOKENS = [220, 50257];

function makeLogits(tokenId: number, value = 10.0): Float32Array {
  const arr = new Float32Array(VOCAB_SIZE);
  arr[tokenId] = value;
  for (let i = 0; i < Math.min(10, VOCAB_SIZE); i++) {
    if (arr[i] === 0) arr[i] = -1.0;
  }
  return arr;
}

describe('WhisperTimestampLogitProcessor', () => {
  it('suppresses suppress_tokens in every step', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: SUPPRESS_TOKENS,
      beginSuppressTokens: BEGIN_SUPPRESS_TOKENS,
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;

    const logits = makeLogits(7, 10.0);
    processor.process(logits, [...prompt], beginIndex);
    expect(logits[7]).toBe(-Infinity);
    expect(logits[1]).toBe(-Infinity);
    expect(logits[2]).toBe(-Infinity);
  });

  it('suppresses begin_suppress_tokens only on first generated token', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: BEGIN_SUPPRESS_TOKENS,
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;

    const firstLogits = makeLogits(220, 10.0);
    processor.process(firstLogits, [...prompt], beginIndex);
    expect(firstLogits[220]).toBe(-Infinity);
    expect(firstLogits[50257]).toBe(-Infinity);

    const secondLogits = makeLogits(220, 10.0);
    processor.process(secondLogits, [...prompt, 100], beginIndex);
    expect(secondLogits[220]).toBe(10.0);
  });

  it('suppresses timestamp tokens in no_timestamps mode', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359, NO_TIMESTAMPS];
    const beginIndex = prompt.length;

    const logits = makeLogits(TIMESTAMP_BEGIN + 5, 10.0);
    processor.process(logits, [...prompt], beginIndex);
    for (let ts = TIMESTAMP_BEGIN; ts <= TIMESTAMP_END; ts++) {
      expect(logits[ts]).toBe(-Infinity);
    }
  });

  it('allows timestamp tokens when no_timestamps token is absent', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;

    const logits = makeLogits(TIMESTAMP_BEGIN + 3, 10.0);
    processor.process(logits, [...prompt], beginIndex);
    expect(logits[TIMESTAMP_BEGIN + 3]).toBe(10.0);
  });

  it('enforces monotonic timestamps — suppresses earlier values', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;
    const ts5 = TIMESTAMP_BEGIN + 5;
    const ts3 = TIMESTAMP_BEGIN + 3;

    const logits = makeLogits(ts3, 10.0);
    processor.process(logits, [...prompt, ts5], beginIndex);
    expect(logits[ts3]).toBe(-Infinity);
  });

  it('suppresses timestamps when two timestamps are consecutive', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;
    const ts2 = TIMESTAMP_BEGIN + 2;
    const ts3 = TIMESTAMP_BEGIN + 3;

    const logits = makeLogits(ts3, 10.0);
    processor.process(logits, [...prompt, ts2, ts3], beginIndex);
    expect(logits[ts3]).toBe(-Infinity);
  });

  it('suppresses text tokens after an unpaired timestamp (forces EOS)', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;
    const ts2 = TIMESTAMP_BEGIN + 2;

    const logits = makeLogits(100, 10.0);
    processor.process(logits, [...prompt, 42, ts2], beginIndex);
    for (let t = 0; t < EOS; t++) {
      expect(logits[t]).toBe(-Infinity);
    }
    expect(Number.isFinite(logits[EOS])).toBe(true);
  });

  it('on first generated token, suppresses all text tokens (only timestamps/EOS allowed)', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;

    const logits = makeLogits(100, 10.0);
    processor.process(logits, [...prompt], beginIndex);
    for (let t = 0; t < TIMESTAMP_BEGIN; t++) {
      expect(logits[t]).toBe(-Infinity);
    }
  });

  it('does NOT suppress anything when no special processing applies', () => {
    const processor = new WhisperTimestampLogitProcessor({
      eosTokenId: EOS,
      noTimestampsTokenId: NO_TIMESTAMPS,
      timestampBegin: TIMESTAMP_BEGIN,
      suppressTokens: [],
      beginSuppressTokens: [],
    });
    const prompt = [50258, 50259, 50359];
    const beginIndex = prompt.length;

    const logits = makeLogits(42, 10.0);
    processor.process(logits, [...prompt, 10, 20], beginIndex);
    expect(logits[42]).toBe(10.0);
  });
});
