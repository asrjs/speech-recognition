import { describe, expect, it } from 'vitest';
import { RoughSpeechGate } from '@asrjs/speech-recognition/realtime';

function createChunk(length: number, amplitude: number): Float32Array {
  const chunk = new Float32Array(length);
  chunk.fill(amplitude);
  return chunk;
}

describe('RoughSpeechGate', () => {
  it('starts speech immediately once a full analysis window clears the energy threshold', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      energySmoothingWindows: 1,
      energyThreshold: 0.08,
      minSilenceDurationMs: 400,
      useSnrGate: false,
    });

    for (let index = 0; index < 4; index += 1) {
      const result = gate.process(createChunk(256, 0.12));
      expect(result.speechStart).toBe(false);
      expect(result.energyPass).toBe(false);
    }

    const result = gate.process(createChunk(256, 0.12));
    expect(result.speechStart).toBe(true);
    expect(result.isSpeech).toBe(true);
    expect(result.energyPass).toBe(true);
    expect(result.candidateReason).toBe('energy-threshold');
  });

  it('backtracks onset using rising energy and low-SNR boundaries', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      energySmoothingWindows: 1,
      energyThreshold: 0.08,
      minSilenceDurationMs: 400,
      useSnrGate: false,
    });

    gate.process(createChunk(1280, 0.001));
    gate.process(createChunk(1280, 0.02));
    const result = gate.process(createChunk(1280, 0.12));

    expect(result.speechStart).toBe(true);
    expect(result.onsetFrame).toBe(0);
    expect(result.onsetFrame).toBeLessThan(result.chunkStartFrame);
  });

  it('adapts the noise floor during sustained silence', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      energySmoothingWindows: 1,
      energyThreshold: 0.08,
      initialNoiseFloor: 0.02,
    });

    let lastResult = null;
    for (let index = 0; index < 12; index += 1) {
      lastResult = gate.process(createChunk(1280, 0.003));
    }

    expect(lastResult?.noiseFloor).toBeLessThan(0.02);
    expect(lastResult?.noiseFloor).toBeGreaterThan(0.001);
  });

  it('feeds rejected candidate windows back into the background model', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      energySmoothingWindows: 1,
      energyThreshold: 0.08,
      initialNoiseFloor: 0.002,
      useSnrGate: false,
    });

    gate.process(createChunk(1280, 0.002));
    gate.process(createChunk(1280, 0.02));
    const result = gate.process(createChunk(1280, 0.002));

    expect(result.isSpeech).toBe(false);
    expect(result.noiseFloor).toBeGreaterThan(0.002);
    expect(result.backgroundAverage).toBeGreaterThan(0.002);
    expect(result.backgroundAverage).toBeLessThan(0.02);
    expect(result.rejectedCandidateAverageDbfs).toBeGreaterThan(-100);
  });

  it('uses the legacy silence release window count even with 80 ms analysis windows', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      energySmoothingWindows: 1,
      energyThreshold: 0.08,
      minSilenceDurationMs: 400,
      useSnrGate: false,
    });

    gate.process(createChunk(1280, 0.12));

    for (let index = 0; index < 3; index += 1) {
      const result = gate.process(createChunk(1280, 0.001));
      expect(result.speechEnd).toBe(false);
      expect(result.isSpeech).toBe(true);
    }

    const release = gate.process(createChunk(1280, 0.001));
    expect(release.speechEnd).toBe(true);
    expect(release.isSpeech).toBe(false);
  });

  it('resets frame-based state when the sample rate changes', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      energySmoothingWindows: 1,
      energyThreshold: 0.08,
    });

    gate.process(createChunk(1280, 0.12));
    gate.updateConfig({ sampleRate: 8000 });

    const result = gate.process(createChunk(640, 0.01));
    expect(result.chunkStartFrame).toBe(0);
    expect(result.chunkEndFrame).toBe(640);
  });
});
