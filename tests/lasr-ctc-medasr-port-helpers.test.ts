import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, test } from 'vitest';

import {
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  buildSentenceTimings,
  buildUtteranceTiming,
  ctcCollapseWithSpans,
} from '../src/models/lasr-ctc/ctc.js';
import { MedAsrTextTokenizer } from '../src/models/lasr-ctc/tokenizer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MEDASR_REFERENCE_TESTS_DIR = path.resolve(
  __dirname,
  '../tools/model-debugging/reference/medasrjs/upstream-tests',
);
const MEDASR_FIXTURE_WAV = path.join(MEDASR_REFERENCE_TESTS_DIR, 'fixtures', 'sanity_sample.wav');
const MEDASR_FIXTURE_LABEL = path.join(
  MEDASR_REFERENCE_TESTS_DIR,
  'fixtures',
  'sanity_sample.label.json',
);

function logitsFromFrameIds(
  frameIds: readonly number[],
  vocabularySize: number,
  peak = 8,
  floor = -8,
): Float32Array {
  const logits = new Float32Array(frameIds.length * vocabularySize);

  for (let frameIndex = 0; frameIndex < frameIds.length; frameIndex += 1) {
    const rowOffset = frameIndex * vocabularySize;
    for (let vocabIndex = 0; vocabIndex < vocabularySize; vocabIndex += 1) {
      logits[rowOffset + vocabIndex] = floor;
    }
    const targetId = frameIds[frameIndex] ?? 0;
    logits[rowOffset + targetId] = peak;
  }

  return logits;
}

describe('MedASR port helper parity', () => {
  test('keeps copied MedASR sanity fixture assets available for troubleshooting', () => {
    expect(fs.existsSync(MEDASR_FIXTURE_WAV)).toBe(true);
    expect(fs.statSync(MEDASR_FIXTURE_WAV).size).toBeGreaterThan(0);

    expect(fs.existsSync(MEDASR_FIXTURE_LABEL)).toBe(true);
    const label = JSON.parse(fs.readFileSync(MEDASR_FIXTURE_LABEL, 'utf-8')) as {
      readonly transcription?: string;
    };
    expect(typeof label.transcription).toBe('string');
    expect(label.transcription?.trim().length).toBeGreaterThan(0);
  });

  test('builds utterance and sentence timings from CTC frame ids', () => {
    const tokenizer = new MedAsrTextTokenizer([
      '<epsilon>',
      '▁hello',
      '▁world',
      '.',
      '▁next',
      '▁line',
      '!',
    ]);

    const expectedFrameIds = [0, 1, 1, 0, 2, 2, 3, 0, 4, 4, 5, 6];
    const logits = logitsFromFrameIds(expectedFrameIds, 7);

    const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(
      logits,
      expectedFrameIds.length,
      7,
    );
    expect(frameIds).toEqual(expectedFrameIds);

    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(
      frameIds,
      selectedLogProbs,
      tokenizer.blankId,
    );
    expect(collapsedIds).toEqual([1, 2, 3, 4, 5, 6]);

    const secondsPerFrame = 0.1;
    const timedSpans = addTimesToTokenSpans(tokenizer, tokenSpans, secondsPerFrame);
    const utterance = buildUtteranceTiming(
      frameIds,
      selectedLogProbs,
      tokenizer.blankId,
      secondsPerFrame,
    );
    const text = tokenizer.decode(collapsedIds);
    const sentences = buildSentenceTimings(text, tokenizer, collapsedIds, timedSpans);

    expect(text).toBe('hello world. next line!');
    expect(utterance.hasSpeech).toBe(true);
    expect(utterance.startFrame).toBe(1);
    expect(utterance.endFrame).toBe(11);
    expect(utterance.startTime).toBeCloseTo(0.1, 6);
    expect(utterance.endTime).toBeCloseTo(1.2, 6);
    expect(utterance.confidence).toBeGreaterThan(0.99);
    expect(utterance.confidence).toBeLessThanOrEqual(1);

    expect(sentences).toHaveLength(2);
    expect(sentences[0]?.text).toBe('hello world.');
    expect(sentences[0]?.startTime).toBeCloseTo(0.1, 6);
    expect(sentences[0]?.endTime).toBeCloseTo(0.7, 6);
    expect(sentences[1]?.text).toBe('next line!');
    expect(sentences[1]?.startTime).toBeCloseTo(0.8, 6);
    expect(sentences[1]?.endTime).toBeCloseTo(1.2, 6);
  });

  test('preserves spacing before bracketed section headers during decode', () => {
    const tokenizer = new MedAsrTextTokenizer([
      '<epsilon>',
      '▁Brain',
      '▁MRI',
      '.',
      '▁[INDICATION]',
      '▁Monitoring',
    ]);

    expect(tokenizer.decode([1, 2, 3, 4, 5])).toBe('Brain MRI. [INDICATION] Monitoring');
  });
});
