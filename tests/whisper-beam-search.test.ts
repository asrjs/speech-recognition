import { describe, expect, it } from 'vitest';
import { createInitialWhisperBeam, rankWhisperBeamCandidates, selectBestWhisperBeam } from '../src/models/whisper-seq2seq/beam-search.js';

describe('Whisper beam search helpers', () => {
  it('keeps the highest scoring expanded hypotheses across beams', () => {
    const beams = [
      createInitialWhisperBeam([10], 0),
      createInitialWhisperBeam([20], -0.2),
    ];

    const next = rankWhisperBeamCandidates({
      beams,
      logitsByBeam: [
        new Float32Array([0, 2, 1]),
        new Float32Array([0, 1, 5]),
      ],
      beamWidth: 2,
      eosTokenId: 2,
    });

    expect(next).toHaveLength(2);
    expect(next[0]?.tokens).toEqual([20, 2]);
    expect(next[0]?.completed).toBe(true);
    expect(next[1]?.tokens).toEqual([10, 1]);
    expect(next[1]?.completed).toBe(false);
  });

  it('applies length penalty when selecting the final beam', () => {
    const short = { ...createInitialWhisperBeam([1, 2], -1), completed: true };
    const long = { ...createInitialWhisperBeam([1, 3, 4, 2], -1.4), completed: true };

    expect(selectBestWhisperBeam([short, long], 0)?.tokens).toEqual([1, 2]);
    expect(selectBestWhisperBeam([short, long], 1)?.tokens).toEqual([1, 3, 4, 2]);
  });
});
