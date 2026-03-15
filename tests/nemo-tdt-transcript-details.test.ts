import { describe, expect, it } from 'vitest';

import { buildWordAndTokenDetails } from '../src/models/nemo-tdt/transcript-details.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';

describe('nemo-tdt transcript detail reconstruction', () => {
  it('reconstructs word and token details from sentencepiece-style token pieces', () => {
    const tokenizer = new ParakeetTokenizer(['<blk>', '▁hello', 'world', '▁again', '.']);

    const details = buildWordAndTokenDetails(
      tokenizer,
      [1, 2, 3, 4],
      [
        [0.0, 0.08],
        [0.08, 0.16],
        [0.16, 0.24],
        [0.24, 0.32],
      ],
      [0.9, 0.7, 0.8, 0.95],
      [0, 1, 2, 3],
      [-0.1, -0.2, -0.3, -0.05],
      [0, 0, 1, 0],
    );

    expect(details.words).toEqual([
      {
        index: 0,
        text: 'helloworld',
        startTime: 0.0,
        endTime: 0.16,
        confidence: 0.8,
      },
      {
        index: 1,
        text: 'again.',
        startTime: 0.16,
        endTime: 0.32,
        confidence: 0.875,
      },
    ]);

    expect(details.tokens).toEqual([
      expect.objectContaining({
        index: 0,
        id: 1,
        text: 'hello',
        rawText: '▁hello',
        isWordStart: true,
        startTime: 0.0,
        endTime: 0.08,
        frameIndex: 0,
        logProb: -0.1,
        tdtStep: 0,
      }),
      expect.objectContaining({
        index: 1,
        id: 2,
        text: 'world',
        rawText: 'world',
        isWordStart: false,
        startTime: 0.08,
        endTime: 0.16,
      }),
      expect.objectContaining({
        index: 2,
        id: 3,
        text: 'again',
        rawText: '▁again',
        isWordStart: true,
      }),
      expect.objectContaining({
        index: 3,
        id: 4,
        text: '.',
        rawText: '.',
        isWordStart: false,
      }),
    ]);
  });

  it('starts a new word when the tokenizer emits no leading word marker but no active word exists', () => {
    const tokenizer = new ParakeetTokenizer(['<blk>', 'hello', 'world']);

    const details = buildWordAndTokenDetails(
      tokenizer,
      [1, 2],
      [
        [0.0, 0.1],
        [0.1, 0.2],
      ],
      [0.5, Number.NaN],
      [0, 1],
      [-1.2, -0.6],
      [0, 0],
    );

    expect(details.words).toEqual([
      {
        index: 0,
        text: 'helloworld',
        startTime: 0.0,
        endTime: 0.2,
        confidence: 0.5,
      },
    ]);
    expect(details.tokens[0]?.isWordStart).toBe(true);
    expect(details.tokens[1]?.isWordStart).toBe(false);
  });
});
