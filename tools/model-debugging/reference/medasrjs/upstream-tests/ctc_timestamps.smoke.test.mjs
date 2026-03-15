import { describe, test, expect } from 'vitest';

import { MedAsrTokenizer } from '../src/tokenizer.js';
import {
  argmaxAndSelectedLogProbs,
  ctcCollapseWithSpans,
  addTimesToTokenSpans,
  buildUtteranceTiming,
  buildSentenceTimings,
} from '../src/core/ctc_timestamps.js';

function logitsFromFrameIds(frameIds, vocabSize, peak = 8, floor = -8) {
  const logits = new Float32Array(frameIds.length * vocabSize);
  for (let t = 0; t < frameIds.length; t++) {
    const row = t * vocabSize;
    for (let v = 0; v < vocabSize; v++) logits[row + v] = floor;
    logits[row + frameIds[t]] = peak;
  }
  return logits;
}

describe('ctc timestamp helpers smoke', () => {
  test('builds utterance and sentence timings from CTC frame ids', () => {
    const id2token = [
      '<epsilon>', // 0 blank
      '▁hello',    // 1
      '▁world',    // 2
      '.',         // 3
      '▁next',     // 4
      '▁line',     // 5
      '!',         // 6
    ];
    const tokenizer = new MedAsrTokenizer(id2token);

    // Decodes to: "hello world. next line!"
    const targetFrameIds = [0, 1, 1, 0, 2, 2, 3, 0, 4, 4, 5, 6];
    const logits = logitsFromFrameIds(targetFrameIds, id2token.length);
    const { ids, selectedLogProbs } = argmaxAndSelectedLogProbs(logits, targetFrameIds.length, id2token.length);

    expect(ids).toEqual(targetFrameIds);

    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(ids, selectedLogProbs, tokenizer.blankId);
    expect(collapsedIds).toEqual([1, 2, 3, 4, 5, 6]);

    const secondsPerFrame = 0.1;
    const timedSpans = addTimesToTokenSpans(tokenizer, tokenSpans, secondsPerFrame);
    const utterance = buildUtteranceTiming(ids, selectedLogProbs, tokenizer.blankId, secondsPerFrame);
    const text = tokenizer.decode(collapsedIds);
    const sentences = buildSentenceTimings(text, tokenizer, collapsedIds, timedSpans);

    expect(text).toBe('hello world. next line!');
    expect(utterance.has_speech).toBe(true);
    expect(utterance.start_frame).toBe(1);
    expect(utterance.end_frame).toBe(11);
    expect(utterance.start_time).toBeCloseTo(0.1, 6);
    expect(utterance.end_time).toBeCloseTo(1.2, 6);
    expect(utterance.confidence).toBeGreaterThan(0.99);
    expect(utterance.confidence).toBeLessThanOrEqual(1);

    expect(sentences).toHaveLength(2);
    expect(sentences[0].text).toBe('hello world.');
    expect(sentences[0].start_time).toBeCloseTo(0.1, 6);
    expect(sentences[0].end_time).toBeCloseTo(0.7, 6);
    expect(sentences[1].text).toBe('next line!');
    expect(sentences[1].start_time).toBeCloseTo(0.8, 6);
    expect(sentences[1].end_time).toBeCloseTo(1.2, 6);
  });
});
