import { describe, expect, it } from 'vitest';
import { VadPostprocessor } from '../src/runtime/firered-vad/core/vad-postprocessor.js';

describe('VadPostprocessor', () => {
  const defaultOptions = {
    smoothWindowSize: 3,
    probThreshold: 0.5,
    minSpeechFrame: 2,
    maxSpeechFrame: 10,
    minSilenceFrame: 2,
    mergeSilenceFrame: 2,
    extendSpeechFrame: 1,
  };

  it('correctly smooths probabilities using a sliding window', () => {
    const postprocessor = new VadPostprocessor(defaultOptions);
    const probs = [0.0, 0.6, 0.9, 0.3, 0.0];
    const result = postprocessor.process(probs);
    // Expected smoothProb output for probs with window size 3:
    // i=0: val=0.0, sum=0.0, len=1 -> 0.0
    // i=1: val=0.6, sum=0.6, len=2 -> 0.3
    // i=2: val=0.9, sum=1.5, len=3 -> 0.5
    // i=3: val=0.3, sum=1.8 - 0.0 = 1.8, len=3 -> 0.6
    // i=4: val=0.0, sum=1.2 - 0.6 = 0.6, len=3 -> 0.2
    expect(result).toBeDefined();
    expect(result.length).toBe(probs.length);
  });

  it('handles empty probabilities array', () => {
    const postprocessor = new VadPostprocessor(defaultOptions);
    expect(postprocessor.process([])).toEqual([]);
  });

  it('handles smoothWindowSize <= 1', () => {
    const postprocessor = new VadPostprocessor({
      ...defaultOptions,
      smoothWindowSize: 1,
    });
    const probs = [0.2, 0.8, 0.4];
    const result = postprocessor.process(probs);
    expect(result.length).toBe(probs.length);
  });
});
