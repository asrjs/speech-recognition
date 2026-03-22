import {
  formatStreamingControlHint,
  formatStreamingControlValue,
  getStreamingControlDefinition,
  normalizeStreamingControlValue,
  resolveStreamingControlConstraints,
} from '@asrjs/speech-recognition/realtime';
import { describe, expect, it } from 'vitest';

describe('streaming control helpers', () => {
  it('resolves dynamic level-window constraints from the runtime config', () => {
    const definition = getStreamingControlDefinition('levelWindowMs');

    expect(definition).toBeTruthy();

    const constraints = resolveStreamingControlConstraints(definition!, {
      chunkDurationMs: 32,
      ringBufferDurationMs: 12000,
    });

    expect(constraints.max).toBe(12000);
    expect(constraints.step).toBe(128);
  });

  it('normalizes chunk-aligned millisecond controls to runtime step size', () => {
    const definition = getStreamingControlDefinition('minSilenceDurationMs');

    expect(definition).toBeTruthy();
    expect(
      normalizeStreamingControlValue(definition!, 23, {
        chunkDurationMs: 16,
      }),
    ).toBe(16);
    expect(
      normalizeStreamingControlValue(definition!, 0, {
        chunkDurationMs: 16,
      }),
    ).toBe(0);
  });

  it('formats values and hints with canonical units', () => {
    const dbfsDefinition = getStreamingControlDefinition('minSpeechLevelDbfs');
    const levelWindowDefinition = getStreamingControlDefinition('levelWindowMs');

    expect(formatStreamingControlValue(dbfsDefinition!, -47)).toBe('-47.0 dBFS');
    expect(
      formatStreamingControlHint(levelWindowDefinition!, {
        chunkDurationMs: 16,
        ringBufferDurationMs: 12000,
      }),
    ).toContain('200..12000 ms');
  });
});

