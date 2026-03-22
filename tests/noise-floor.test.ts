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

  it('sanitizes invalid adaptation config values', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: Number.NaN,
      fastAdaptationRate: 5,
      slowAdaptationRate: -2,
      minBackgroundDurationSec: Number.NaN,
    });

    const state = tracker.observeWindow('confirmed-silence-window', 0.003, 0.08);

    expect(Number.isFinite(state.noiseFloor)).toBe(true);
    expect(state.noiseFloor).toBeGreaterThan(0);
    expect(Number.isFinite(state.backgroundAverage)).toBe(true);
  });

  it('resets accumulated state when initialNoiseFloor is explicitly updated', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: 0.004,
      fastAdaptationRate: 0.15,
      slowAdaptationRate: 0.05,
      minBackgroundDurationSec: 1,
    });

    for (let index = 0; index < 6; index += 1) {
      tracker.observeWindow('confirmed-silence-window', 0.002, 0.08);
    }

    tracker.updateConfig({
      initialNoiseFloor: 0.01,
    });

    const state = tracker.getState();
    expect(state.noiseFloor).toBe(0.01);
    expect(state.confirmedSilenceDurationSec).toBe(0);
    expect(state.confirmedBackgroundObservationCount).toBe(0);
    expect(state.rejectedBackgroundObservationCount).toBe(0);
  });

  it('does not reset learned state when initialNoiseFloor is unchanged after sanitization', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: 0.004,
      fastAdaptationRate: 0.15,
      slowAdaptationRate: 0.05,
      minBackgroundDurationSec: 1,
    });

    for (let index = 0; index < 6; index += 1) {
      tracker.observeWindow('confirmed-silence-window', 0.002, 0.08);
    }

    const before = tracker.getState();
    tracker.updateConfig({
      initialNoiseFloor: 0.004,
    });
    const after = tracker.getState();

    expect(after.noiseFloor).toBe(before.noiseFloor);
    expect(after.confirmedSilenceDurationSec).toBe(before.confirmedSilenceDurationSec);
    expect(after.confirmedBackgroundObservationCount).toBe(
      before.confirmedBackgroundObservationCount,
    );
  });

  it('honors zero adaptation rates for rejected candidate windows', () => {
    const tracker = new NoiseFloorTracker({
      initialNoiseFloor: 0.004,
      fastAdaptationRate: 0,
      slowAdaptationRate: 0,
      minBackgroundDurationSec: 1,
    });

    const before = tracker.getState();
    const after = tracker.observeWindow('rejected-candidate-window', 0.02, 0.032);

    expect(after.noiseFloor).toBe(before.noiseFloor);
  });
});
