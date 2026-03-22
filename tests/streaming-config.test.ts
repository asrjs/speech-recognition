import { isStreamingConfigEqual, mergeStreamingConfig } from '@asrjs/speech-recognition/realtime';
import { describe, expect, it } from 'vitest';

describe('streaming config helpers', () => {
  it('drops runtime-only energyThreshold after normalizing overrides', () => {
    const merged = mergeStreamingConfig('generic-streaming', {
      energyThreshold: 0.01,
    });

    expect('energyThreshold' in merged).toBe(false);
    expect(merged.gateMode).toBe('rough-and-ten-vad');
    expect(
      isStreamingConfigEqual(
        merged,
        mergeStreamingConfig('generic-streaming', {
          minSpeechLevelDbfs: merged.minSpeechLevelDbfs,
        }),
      ),
    ).toBe(true);
  });

  it('ignores deprecated waveformPointCount overrides', () => {
    const merged = mergeStreamingConfig('generic-streaming', {
      waveformPointCount: 42,
    });

    expect('waveformPointCount' in merged).toBe(false);
  });

  it('derives analysis-window counts from analysis duration when chunk timing changes', () => {
    const merged = mergeStreamingConfig('generic-streaming', {
      analysisWindowMs: 160,
    });

    expect(merged.energySmoothingWindows).toBe(3);
    expect(merged.maxOnsetLookbackChunks).toBe(2);
    expect(merged.defaultOnsetLookbackChunks).toBe(2);
  });

  it('aligns TEN-VAD smoothing durations to the configured chunk size', () => {
    const merged = mergeStreamingConfig('generic-streaming', {
      chunkDurationMs: 16,
      tenVadMinSpeechDurationMs: 241,
      tenVadMinSilenceDurationMs: 81,
      tenVadSpeechPaddingMs: 40,
    });

    expect(merged.chunkDurationMs).toBe(16);
    expect(merged.tenVadMinSpeechDurationMs).toBe(256);
    expect(merged.tenVadMinSilenceDurationMs).toBe(96);
    expect(merged.tenVadSpeechPaddingMs).toBe(48);
  });

  it('re-derives chunk-based defaults when chunk duration changes', () => {
    const merged = mergeStreamingConfig('generic-streaming', {
      chunkDurationMs: 32,
    });

    expect(merged.chunkDurationMs).toBe(32);
    expect(merged.energySmoothingWindows).toBe(5);
    expect(merged.maxOnsetLookbackChunks).toBe(3);
    expect(merged.defaultOnsetLookbackChunks).toBe(3);
    expect(merged.tenVadSpeechPaddingMs).toBe(96);
  });
});
