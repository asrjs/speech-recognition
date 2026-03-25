import { describe, expect, it } from 'vitest';
import { scoreSegmentForeground } from '../src/runtime/segment-foreground-filter.js';

function createConstantPcm(durationMs: number, sampleRate: number, amplitude: number): Float32Array {
  const frameCount = Math.max(1, Math.round((durationMs / 1000) * sampleRate));
  const pcm = new Float32Array(frameCount);
  pcm.fill(amplitude);
  return pcm;
}

describe('scoreSegmentForeground', () => {
  const sampleRate = 16000;
  const baseConfig = {
    foregroundFilterEnabled: true,
    minSpeechDurationMs: 240,
    minEnergyPerSecond: 5,
    minEnergyIntegral: 22,
    useAdaptiveEnergyThresholds: true,
    adaptiveEnergyIntegralFactor: 25,
    adaptiveEnergyPerSecondFactor: 10,
    minAdaptiveEnergyIntegral: 3,
    minAdaptiveEnergyPerSecond: 1,
  };

  it('rejects segments that are shorter than the final duration gate', () => {
    const result = scoreSegmentForeground(
      createConstantPcm(160, sampleRate, 0.2),
      sampleRate,
      0.005,
      baseConfig,
      { noiseWindowFrames: 1280 },
    );

    expect(result.accepted).toBe(false);
    expect(result.reason).toBe('too-short');
  });

  it('rejects weak segments with too little total normalized energy', () => {
    const result = scoreSegmentForeground(
      createConstantPcm(1000, sampleRate, 0.025),
      sampleRate,
      0.005,
      {
        ...baseConfig,
        minSpeechDurationMs: 240,
        useAdaptiveEnergyThresholds: false,
      },
    );

    expect(result.accepted).toBe(false);
    expect(result.reason).toBe('low-energy-integral');
    expect(result.normalizedPowerAt16k).toBeGreaterThan(5);
    expect(result.normalizedEnergyIntegralAt16k).toBeLessThan(22);
  });

  it('rejects very long weak speech when average normalized power stays below threshold', () => {
    const result = scoreSegmentForeground(
      createConstantPcm(10_000, sampleRate, 0.015),
      sampleRate,
      0.005,
      {
        ...baseConfig,
        minSpeechDurationMs: 240,
        useAdaptiveEnergyThresholds: false,
      },
    );

    expect(result.accepted).toBe(false);
    expect(result.reason).toBe('low-energy-per-second');
    expect(result.normalizedEnergyIntegralAt16k).toBeGreaterThan(22);
    expect(result.normalizedPowerAt16k).toBeLessThan(5);
  });

  it('raises adaptive thresholds from the live noise floor', () => {
    const result = scoreSegmentForeground(
      createConstantPcm(1000, sampleRate, 0.04),
      sampleRate,
      0.2,
      baseConfig,
      { noiseWindowFrames: 1280 },
    );

    expect(result.usedAdaptiveThresholds).toBe(true);
    expect(result.minEnergyIntegralThreshold).toBeGreaterThan(baseConfig.minEnergyIntegral);
    expect(result.minEnergyPerSecondThreshold).toBeGreaterThan(baseConfig.minEnergyPerSecond);
    expect(result.accepted).toBe(false);
    expect(result.reason).toBe('low-energy-integral');
  });

  it('accepts strong near-mic speech when duration and normalized energy both pass', () => {
    const result = scoreSegmentForeground(
      createConstantPcm(1000, sampleRate, 0.05),
      sampleRate,
      0.005,
      {
        ...baseConfig,
        useAdaptiveEnergyThresholds: false,
      },
    );

    expect(result.accepted).toBe(true);
    expect(result.reason).toBe('accepted');
    expect(result.normalizedPowerAt16k).toBeGreaterThan(22);
    expect(result.normalizedEnergyIntegralAt16k).toBeGreaterThan(22);
  });
});
