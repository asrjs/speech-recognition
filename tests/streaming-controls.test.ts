import {
  formatStreamingControlHint,
  formatStreamingControlValue,
  getStreamingControlDefinition,
  normalizeStreamingControlValue,
  resolveStreamingControlConstraints,
} from '@asrjs/speech-recognition/realtime';
import { describe, expect, it } from 'vitest';

describe('streaming control helpers', () => {
  it('resolves control constraints from the canonical definition metadata', () => {
    const definition = getStreamingControlDefinition('maxSegmentDurationMs');

    expect(definition).toBeTruthy();

    const constraints = resolveStreamingControlConstraints(definition!, {
      chunkDurationMs: 32,
      ringBufferDurationMs: 12000,
    });

    expect(constraints.max).toBe(12000);
    expect(constraints.step).toBe(80);
  });

  it('normalizes chunk-aligned millisecond controls to runtime step size', () => {
    const definition = getStreamingControlDefinition('minSilenceDurationMs');
    const prerollDefinition = getStreamingControlDefinition('prerollMs');

    expect(definition).toBeTruthy();
    expect(prerollDefinition).toBeTruthy();
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
    expect(
      resolveStreamingControlConstraints(prerollDefinition!, {
        chunkDurationMs: 32,
      }).min,
    ).toBe(96);
    expect(
      normalizeStreamingControlValue(prerollDefinition!, 80, {
        chunkDurationMs: 32,
      }),
    ).toBe(96);
  });

  it('formats values and hints with canonical units', () => {
    const thresholdDefinition = getStreamingControlDefinition('energyThreshold');
    const prerollDefinition = getStreamingControlDefinition('prerollMs');
    const backgroundDefinition = getStreamingControlDefinition('minBackgroundDurationSec');

    expect(formatStreamingControlValue(thresholdDefinition!, 0.08)).toBe('0.08');
    expect(formatStreamingControlValue(backgroundDefinition!, 1.2)).toBe('1.2 s');
    expect(
      formatStreamingControlHint(prerollDefinition!, {
        chunkDurationMs: 16,
        ringBufferDurationMs: 12000,
      }),
    ).toContain('80..800 ms');
  });
});
