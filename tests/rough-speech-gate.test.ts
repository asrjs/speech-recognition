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
      analysisWindowMs: 16,
      minSpeechDurationMs: 32,
      minSilenceDurationMs: 96,
      minSpeechLevelDbfs: -40,
      useSnrGate: false,
    });

    gate.process(createChunk(256, 0.001));
    gate.process(createChunk(256, 0.004));
    gate.process(createChunk(256, 0.05));
    const result = gate.process(createChunk(256, 0.05));

    expect(result.speechStart).toBe(true);
    expect(result.onsetFrame).toBeLessThan(result.chunkStartFrame);
  });

  it('adapts the noise floor during sustained silence', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 16,
      minSpeechDurationMs: 32,
      minSilenceDurationMs: 96,
      initialNoiseFloor: 0.02,
    });

    let lastResult = null;
    for (let index = 0; index < 12; index += 1) {
      lastResult = gate.process(createChunk(256, 0.003));
    }

    expect(lastResult?.noiseFloor).toBeLessThan(0.02);
    expect(lastResult?.noiseFloor).toBeGreaterThan(0.001);
  });

  it('reports calibrated dBFS levels for the current and rolling windows', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 96,
      levelWindowMs: 1008,
      minSpeechLevelDbfs: -40,
    });

    let lastResult = null;
    for (let index = 0; index < 10; index += 1) {
      lastResult = gate.process(createChunk(1536, 0.01));
    }

    expect(lastResult?.levelDbfs).toBeCloseTo(-40, 0);
    expect(lastResult?.levelWindowDbfs).toBeCloseTo(-40, 0);
    expect(lastResult?.thresholdDbfs).toBe(-40);
  });

  it('preserves candidate state between aligned analysis windows', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      minSpeechDurationMs: 160,
      minSilenceDurationMs: 240,
      minSpeechLevelDbfs: -40,
      useSnrGate: false,
    });

    const speechChunk = createChunk(256, 0.05);

    for (let index = 0; index < 4; index += 1) {
      const result = gate.process(speechChunk);
      expect(result.isSpeech).toBe(false);
      expect(result.energyPass).toBe(false);
      expect(result.candidateReason).toBe('none');
    }

    const firstWindow = gate.process(speechChunk);
    expect(firstWindow.isSpeech).toBe(false);
    expect(firstWindow.energyPass).toBe(true);
    expect(firstWindow.candidateReason).toBe('energy-threshold');

    for (let index = 0; index < 4; index += 1) {
      const result = gate.process(speechChunk);
      expect(result.isSpeech).toBe(false);
      expect(result.energyPass).toBe(true);
      expect(result.candidateReason).toBe('energy-threshold');
    }

    const activationWindow = gate.process(speechChunk);
    expect(activationWindow.isSpeech).toBe(true);
    expect(activationWindow.speechStart).toBe(true);
    expect(activationWindow.energyPass).toBe(true);
    expect(activationWindow.candidateReason).toBe('energy-threshold');
  });

  it('resets frame-based state when the sample rate changes', () => {
    const gate = new RoughSpeechGate({
      sampleRate: 16000,
      analysisWindowMs: 80,
      minSpeechDurationMs: 160,
      minSilenceDurationMs: 240,
    });

    gate.process(createChunk(1280, 0.05));
    gate.updateConfig({ sampleRate: 8000 });

    const result = gate.process(createChunk(640, 0.01));
    expect(result.chunkStartFrame).toBe(0);
    expect(result.chunkEndFrame).toBe(640);
  });
});
