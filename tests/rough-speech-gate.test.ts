import { describe, expect, it } from 'vitest';
import { RoughSpeechGate } from '@asrjs/speech-recognition/realtime';

function createChunk(length: number, amplitude: number): Float32Array {
  const chunk = new Float32Array(length);
  chunk.fill(amplitude);
  return chunk;
}

describe('RoughSpeechGate', () => {
  it('detects onset with lookback before the confirmed speech frame', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 20,
      minSpeechDurationMs: 40,
      minSilenceDurationMs: 120,
      minSpeechLevelDbfs: -40,
      useSnrGate: false,
    });

    gate.process(createChunk(320, 0.001));
    gate.process(createChunk(320, 0.004));
    gate.process(createChunk(320, 0.05));
    const result = gate.process(createChunk(320, 0.05));

    expect(result.speechStart).toBe(true);
    expect(result.onsetFrame).toBeLessThan(result.chunkStartFrame);
  });

  it('adapts the noise floor during sustained silence', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 20,
      minSpeechDurationMs: 40,
      minSilenceDurationMs: 120,
      initialNoiseFloor: 0.02,
    });

    let lastResult = null;
    for (let index = 0; index < 12; index += 1) {
      lastResult = gate.process(createChunk(320, 0.003));
    }

    expect(lastResult?.noiseFloor).toBeLessThan(0.02);
    expect(lastResult?.noiseFloor).toBeGreaterThan(0.001);
  });

  it('reports calibrated dBFS levels for the current and rolling windows', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 100,
      levelWindowMs: 1000,
      minSpeechLevelDbfs: -40,
    });

    let lastResult = null;
    for (let index = 0; index < 10; index += 1) {
      lastResult = gate.process(createChunk(1600, 0.01));
    }

    expect(lastResult?.levelDbfs).toBeCloseTo(-40, 0);
    expect(lastResult?.levelWindowDbfs).toBeCloseTo(-40, 0);
    expect(lastResult?.thresholdDbfs).toBe(-40);
  });
});
