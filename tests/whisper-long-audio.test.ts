import { describe, expect, it } from 'vitest';
import { mergeWhisperChunkTranscripts } from '../src/models/whisper-seq2seq/chunking.js';
import type { WhisperNativeTranscript } from '../src/models/whisper-seq2seq/types.js';

function chunk(text: string, start: number, end: number, tokenBase: number): WhisperNativeTranscript {
  return {
    utteranceText: text,
    isFinal: true,
    language: 'en',
    segments: [{ index: 0, text, startTime: 0, endTime: end - start, confidence: 0.8 }],
    words: text.split(' ').map((word, index) => ({
      index,
      text: word,
      startTime: index * 0.5,
      endTime: index * 0.5 + 0.4,
      tokenIndices: [index],
      confidence: 0.7,
    })),
    tokens: text.split(' ').map((word, index) => ({
      index,
      id: tokenBase + index,
      text: index === 0 ? word : ` ${word}`,
      startTime: index * 0.5,
      endTime: index * 0.5 + 0.4,
      confidence: 0.7,
    })),
  };
}

describe('Whisper native chunk transcript merge', () => {
  it('offsets chunk-local timings and concatenates native transcript details', () => {
    const merged = mergeWhisperChunkTranscripts([
      { chunkStartTime: 0, transcript: chunk('hello world', 0, 1, 10) },
      { chunkStartTime: 20, transcript: chunk('again soon', 20, 21, 20) },
    ]);

    expect(merged.utteranceText).toBe('hello world again soon');
    expect(merged.segments).toEqual([
      { index: 0, text: 'hello world', startTime: 0, endTime: 1, confidence: 0.8 },
      { index: 1, text: 'again soon', startTime: 20, endTime: 21, confidence: 0.8 },
    ]);
    expect(merged.words?.map((word) => ({ index: word.index, text: word.text, startTime: word.startTime, endTime: word.endTime }))).toEqual([
      { index: 0, text: 'hello', startTime: 0, endTime: 0.4 },
      { index: 1, text: 'world', startTime: 0.5, endTime: 0.9 },
      { index: 2, text: 'again', startTime: 20, endTime: 20.4 },
      { index: 3, text: 'soon', startTime: 20.5, endTime: 20.9 },
    ]);
    expect(merged.tokens?.map((token) => ({ index: token.index, text: token.text, startTime: token.startTime }))).toEqual([
      { index: 0, text: 'hello', startTime: 0 },
      { index: 1, text: ' world', startTime: 0.5 },
      { index: 2, text: 'again', startTime: 20 },
      { index: 3, text: ' soon', startTime: 20.5 },
    ]);
  });

  it('deduplicates overlapping native words while keeping the higher confidence copy', () => {
    const lowerConfidence = chunk('world', 0, 1, 30);
    const higherConfidence: WhisperNativeTranscript = {
      ...chunk('world', 0, 1, 40),
      words: [{
        ...chunk('world', 0, 1, 40).words![0]!,
        confidence: 0.95,
      }],
    };
    const merged = mergeWhisperChunkTranscripts([
      { chunkStartTime: 0, transcript: lowerConfidence },
      { chunkStartTime: 0.2, transcript: higherConfidence },
    ]);

    expect(merged.words).toHaveLength(1);
    expect(merged.words![0]!.confidence).toBe(0.95);
    expect(merged.words![0]!.startTime).toBe(0.2);
    expect(merged.words![0]!.index).toBe(0);
  });
});
