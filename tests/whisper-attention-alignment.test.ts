import { describe, expect, it } from 'vitest';
import {
  computeWhisperDtwTokenTimestamps,
  medianFilterWhisperAttention,
  spreadWhisperDtwTimestamps,
} from '../src/models/whisper-seq2seq/index.js';

describe('Whisper attention DTW alignment helpers', () => {
  it('median filters attention over the audio-frame axis with reflected edges', () => {
    const filtered = medianFilterWhisperAttention(
      Float32Array.from([
        0, 10, 0, 8, 0,
        1, 1, 9, 1, 1,
      ]),
      { tokenCount: 2, frameCount: 5, width: 3 },
    );

    expect(Array.from(filtered)).toEqual([
      10, 0, 8, 0, 8,
      1, 1, 1, 1, 1,
    ]);
  });

  it('computes monotonic token timestamps from selected cross-attention heads', () => {
    const timestamps = computeWhisperDtwTokenTimestamps({
      // one selected alignment head, 3 decoder text tokens, 6 encoder frames
      attentionHeads: [
        {
          values: Float32Array.from([
            9, 8, 2, 1, 0, 0,
            0, 2, 9, 8, 2, 0,
            0, 0, 1, 2, 8, 9,
          ]),
          tokenCount: 3,
          frameCount: 6,
        },
      ],
      tokenCount: 3,
      frameCount: 6,
      medianFilterWidth: 1,
      timePrecisionSeconds: 0.02,
    });

    expect(timestamps).toHaveLength(4);
    expect(timestamps[0]).toBe(0);
    expect(timestamps[1]).toBeGreaterThanOrEqual(0.02);
    expect(timestamps[2]).toBeGreaterThanOrEqual(timestamps[1]!);
    expect(timestamps[3]).toBeCloseTo(0.1, 5);
  });

  it('spreads identical DTW jump times so tokens are not zero-duration', () => {
    expect(spreadWhisperDtwTimestamps([0, 0.5, 0.5, 1.5])).toEqual([0, 0.5, 1, 1.5]);
  });
  it('softmaxes each raw-logit head after cropping padded frames', () => {
    const head0 = Float32Array.from([
      -6.33, -9.52, -2.49, -8.28, -10.4, -2.36, 10.03, 7.21,
      6.36, -6.67, 0.88, -5.36, -7.86, -9.45, -6.85, 10.26,
      7.89, 7.36, 7.21, -7.36, -4.56, 3.05, 5.57, 8.51,
    ]);
    const head1 = Float32Array.from([
      9.12, -9.92, 2.54, 4.12, 0.14, -7.73, -0.63, -9.86,
      10.43, 8.77, 1.14, -4.79, 9.81, 1.74, 9.18, 8.35,
      0.2, -2.07, 2.37, -1.65, -8.13, -4.68, 7.5, -10.96,
    ]);
    const head2 = Float32Array.from([
      -10.89, 3.03, -5.27, 0.83, -0.69, -3.77, 11.93, -7.31,
      -2.09, -7.14, 3.18, -5.37, -3.46, 5.93, -4.3, 1.4,
      9.7, -9.58, -10.52, -6.51, 6.36, 2.77, -6.3, -4.05,
    ]);

    const perHead = computeWhisperDtwTokenTimestamps({
      attentionHeads: [
        { values: head0, tokenCount: 3, frameCount: 8, valuesAreLogits: true },
        { values: head1, tokenCount: 3, frameCount: 8, valuesAreLogits: true },
        { values: head2, tokenCount: 3, frameCount: 8, valuesAreLogits: true },
      ],
      tokenCount: 3,
      frameCount: 4,
      medianFilterWidth: 1,
      timePrecisionSeconds: 0.02,
    });

    // This is the legacy graph contract: average full-window probabilities,
    // then crop and renormalize the already-averaged rows.
    const softmax = (values: Float32Array, row: number) => {
      const offset = row * 8;
      const max = Math.max(...Array.from(values.slice(offset, offset + 8)));
      const weights = Array.from(values.slice(offset, offset + 8), (value) => Math.exp(value - max));
      const sum = weights.reduce((total, value) => total + value, 0);
      return weights.map((value) => value / sum);
    };
    const legacy = new Float32Array(3 * 8);
    for (let token = 0; token < 3; token++) {
      const p0 = softmax(head0, token);
      const p1 = softmax(head1, token);
      const p2 = softmax(head2, token);
      for (let frame = 0; frame < 8; frame++) {
        legacy[token * 8 + frame] = (p0[frame]! + p1[frame]! + p2[frame]!) / 3;
      }
    }
    const legacyTimestamps = computeWhisperDtwTokenTimestamps({
      attentionHeads: [{ values: legacy, tokenCount: 3, frameCount: 8 }],
      tokenCount: 3,
      frameCount: 4,
      medianFilterWidth: 1,
      timePrecisionSeconds: 0.02,
    });

    expect(perHead).not.toEqual(legacyTimestamps);
    expect(perHead).toHaveLength(4);
    for (let index = 1; index < perHead.length; index++) {
      expect(perHead[index]!).toBeGreaterThanOrEqual(perHead[index - 1]!);
    }
  });

  it('normalizes all teacher-forced rows before selecting alignment rows', () => {
    const fullRows = Float32Array.from([
      0, 0, 0, 0,
      8, 0, 0, 0,
      0, 0, 8, 0,
      0, 0, 0, 0,
    ]);
    const selectedRows = fullRows.slice(4, 12);

    const fullMatrix = computeWhisperDtwTokenTimestamps({
      attentionHeads: [{ values: fullRows, tokenCount: 4, frameCount: 4, valuesAreLogits: true }],
      tokenCount: 2,
      normalizationTokenCount: 4,
      tokenRowIndices: [1, 2],
      frameCount: 4,
      medianFilterWidth: 1,
      timePrecisionSeconds: 0.02,
    });
    const compactMatrix = computeWhisperDtwTokenTimestamps({
      attentionHeads: [{ values: selectedRows, tokenCount: 2, frameCount: 4, valuesAreLogits: true }],
      tokenCount: 2,
      frameCount: 4,
      medianFilterWidth: 1,
      timePrecisionSeconds: 0.02,
    });

    expect(fullMatrix).not.toEqual(compactMatrix);
    expect(fullMatrix).toHaveLength(3);
  });
});
