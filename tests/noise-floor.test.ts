import { describe, expect, it } from 'vitest';
import { NoiseFloorTracker } from '@asrjs/speech-recognition/realtime';

describe('NoiseFloorTracker', () => {
  it('adapts downward during sustained confirmed silence', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: 0.02,
      fastAdaptationRate: 0.15,
      slowAdaptationRate: 0.05,
      minBackgroundDurationSec: 1,
    });

    let lastState = tracker.getState();
    for (let index = 0; index < 16; index += 1) {
      lastState = tracker.observeWindow('confirmed-silence-window', 0.003, 0.08);
    }

    expect(lastState.noiseFloor).toBeLessThan(0.02);
    expect(lastState.noiseFloor).toBeGreaterThan(0.001);
    expect(lastState.backgroundAverage).toBeGreaterThan(0.001);
    expect(lastState.confirmedBackgroundObservationCount).toBeGreaterThan(0);
    expect(lastState.rejectedBackgroundObservationCount).toBe(0);
  });

  it('uses rejected candidate windows as bounded background evidence', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: 0.004,
      fastAdaptationRate: 0.15,
      slowAdaptationRate: 0.05,
      minBackgroundDurationSec: 1,
    });

    for (let index = 0; index < 8; index += 1) {
      tracker.observeWindow('confirmed-silence-window', 0.003, 0.08);
    }

    const state = tracker.observeWindow('rejected-candidate-window', 0.02, 0.032);

    expect(state.noiseFloor).toBeGreaterThan(0.003);
    expect(state.noiseFloor).toBeLessThan(0.02);
    expect(state.backgroundAverage).toBeLessThan(0.02);
    expect(state.confirmedSilenceAverage).toBeLessThan(state.rejectedCandidateAverage);
  });

  it('prefers confirmed silence over rejected candidates when both are available', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: 0.004,
      fastAdaptationRate: 0.15,
      slowAdaptationRate: 0.05,
      minBackgroundDurationSec: 1,
    });

    for (let index = 0; index < 10; index += 1) {
      tracker.observeWindow('confirmed-silence-window', 0.0025, 0.08);
    }
    for (let index = 0; index < 3; index += 1) {
      tracker.observeWindow('rejected-candidate-window', 0.015, 0.032);
    }

    const state = tracker.getState();
    expect(state.backgroundAverage).toBeGreaterThan(state.confirmedSilenceAverage);
    expect(state.backgroundAverage).toBeLessThan(state.rejectedCandidateAverage);
  });
});
