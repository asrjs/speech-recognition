import { describe, test, expect } from 'vitest';

import { MedAsrDecoder } from '../src/pipeline/decoder.js';

function toLogits(rows) {
  const frames = rows.length;
  const vocab = rows[0].length;
  const out = new Float32Array(frames * vocab);
  for (let t = 0; t < frames; t++) {
    for (let v = 0; v < vocab; v++) out[t * vocab + v] = rows[t][v];
  }
  return { logits: out, frames, vocab };
}

describe('medasrjs decoder smoke', () => {
  test('ctc collapse removes repeats and blanks', () => {
    const blankId = 0;
    const decoder = new MedAsrDecoder({ blankId });

    // frame argmax IDs should resolve to: [0, 5, 5, 0, 7, 7, 7, 0, 2]
    const { logits, frames, vocab } = toLogits([
      [10, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 10, 0, 0],
      [0, 0, 0, 0, 0, 10, 0, 0],
      [10, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 10],
      [0, 0, 0, 0, 0, 0, 0, 10],
      [0, 0, 0, 0, 0, 0, 0, 10],
      [10, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 10, 0, 0, 0, 0, 0],
    ]);

    const decoded = decoder.decodeLogits(logits, frames, vocab);
    expect(decoded.frameIds).toEqual([0, 5, 5, 0, 7, 7, 7, 0, 2]);
    expect(decoded.tokenIds).toEqual([5, 7, 2]);
  });
});
