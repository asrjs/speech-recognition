/**
 * Tests for ChunkContextBuilder — condition-on-previous-text.
 * Phase 4: prompt building with previous segment context.
 */

import { describe, it, expect } from 'vitest';
import {
  ChunkContextBuilder,
  buildPromptWithContext,
} from '../src/models/whisper-seq2seq/chunk-context.js';

// Standard English prompt tokens: [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
const EN_PROMPT = Object.freeze([50258, 50259, 50359, 50363]);

describe('ChunkContextBuilder', () => {
  it('starts with empty context', () => {
    const builder = new ChunkContextBuilder(200);
    expect(builder.getPreviousTokens()).toEqual([]);
    expect(builder.getTotalTokenCount()).toBe(0);
  });

  it('adds segment tokens and retrieves them', () => {
    const builder = new ChunkContextBuilder(200);
    builder.addSegmentTokens([400, 370, 452]);
    expect(builder.getPreviousTokens()).toEqual([400, 370, 452]);
    expect(builder.getTotalTokenCount()).toBe(3);
  });

  it('crops to maxContextTokens from the end', () => {
    const builder = new ChunkContextBuilder(3);
    builder.addSegmentTokens([1, 2, 3, 4, 5, 6]);
    // Should keep last 3: [4, 5, 6]
    expect(builder.getPreviousTokens()).toEqual([4, 5, 6]);
    expect(builder.getTotalTokenCount()).toBe(3);
  });

  it('accumulates across multiple segments', () => {
    const builder = new ChunkContextBuilder(100);
    builder.addSegmentTokens([1, 2, 3]);
    builder.addSegmentTokens([4, 5]);
    builder.addSegmentTokens([6, 7, 8]);
    expect(builder.getPreviousTokens()).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(builder.getTotalTokenCount()).toBe(8);
  });

  it('crops accumulated tokens to maxContextTokens', () => {
    const builder = new ChunkContextBuilder(5);
    builder.addSegmentTokens([1, 2, 3]);
    builder.addSegmentTokens([4, 5, 6, 7, 8]);
    expect(builder.getPreviousTokens()).toEqual([4, 5, 6, 7, 8]);
    expect(builder.getTotalTokenCount()).toBe(5);
  });

  it('resets all state', () => {
    const builder = new ChunkContextBuilder(200);
    builder.addSegmentTokens([1, 2, 3]);
    expect(builder.getTotalTokenCount()).toBe(3);
    builder.reset();
    expect(builder.getPreviousTokens()).toEqual([]);
    expect(builder.getTotalTokenCount()).toBe(0);
  });

  it('handles empty segments without error', () => {
    const builder = new ChunkContextBuilder(200);
    builder.addSegmentTokens([]);
    expect(builder.getTotalTokenCount()).toBe(0);
  });
});

describe('buildPromptWithContext', () => {
  it('returns base prompt when no previous tokens', () => {
    const result = buildPromptWithContext(EN_PROMPT, [], 200);
    expect(result).toEqual([...EN_PROMPT]);
  });

  it('appends previous tokens after prompt with timestamp prefix', () => {
    const previous = [400, 370, 452];
    const result = buildPromptWithContext(EN_PROMPT, previous, 200);
    // Structure: [...base_prompt, <|0.00|>, ...previous_tokens]
    // <|0.00|> = 50363 (notimestamps) actually... no, <|0.00|> is a different token
    // Actually: prompt is [SOT, lang, transcribe, notimestamps]
    // Context: [..., SOT, lang, transcribe, timestamp0, ...prev_tokens]
    // Timestamp 0 = 50364
    expect(result.length).toBeGreaterThan(EN_PROMPT.length);
    const promptEnd = EN_PROMPT.slice();
    expect(result.slice(0, EN_PROMPT.length)).toEqual(promptEnd);
    // After prompt: timestamp 0 token, then previous tokens
    expect(result[EN_PROMPT.length]).toBe(50364); // <|0.00|>
    expect(result.slice(EN_PROMPT.length + 1)).toEqual(previous);
  });

  it('crops previous tokens to maxContextTokens from the end', () => {
    const previous = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const maxContext = 3;
    const result = buildPromptWithContext(EN_PROMPT, previous, maxContext);
    // Should contain only last 3: [8, 9, 10]
    expect(result[result.length - 3]).toBe(8);
    expect(result[result.length - 2]).toBe(9);
    expect(result[result.length - 1]).toBe(10);
  });

  it('uses maxContextTokens=0 by dropping all previous tokens', () => {
    const result = buildPromptWithContext(EN_PROMPT, [1, 2, 3], 0);
    expect(result).toEqual([...EN_PROMPT]);
  });
});
