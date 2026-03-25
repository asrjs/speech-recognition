import {
  estimateStreamingReleaseMs,
  getStreamingSegmentDurationSeconds,
  listStreamingControlGroups,
  resolveStreamingControlGroups,
  resolveStreamingForegroundThresholdDbfs,
  resolveStreamingSnapshotNoiseFloorDbfs,
} from '@asrjs/speech-recognition/realtime';
import { describe, expect, it } from 'vitest';

describe('streaming consumer helpers', () => {
  it('exposes stable detector tuning groups for consumer apps', () => {
    const groups = listStreamingControlGroups();
    const resolvedGroups = resolveStreamingControlGroups();

    expect(groups.map((group) => group.id)).toEqual(['segmenter', 'foreground']);
    expect(resolvedGroups).toHaveLength(2);
    expect(resolvedGroups[0]?.controls.length).toBeGreaterThan(0);
    expect(resolvedGroups[1]?.controls.length).toBeGreaterThan(0);
    expect(resolvedGroups[0]?.controls.every((control) => groups[0]?.fields.includes(control.field))).toBe(true);
    expect(resolvedGroups[1]?.controls.every((control) => groups[1]?.fields.includes(control.field))).toBe(true);
  });

  it('estimates release timing from the slowest TEN-VAD tail budget', () => {
    expect(
      estimateStreamingReleaseMs({
        tenVadConfirmationWindowMs: 120,
        tenVadMinSilenceDurationMs: 260,
        tenVadHangoverMs: 180,
      }),
    ).toBe(260);

    expect(estimateStreamingReleaseMs(null)).toBeNull();
    expect(
      estimateStreamingReleaseMs({
        tenVadConfirmationWindowMs: 120,
        tenVadMinSilenceDurationMs: Number.NaN,
        tenVadHangoverMs: 180,
      }),
    ).toBeNull();
  });

  it('prefers the foreground noise floor and falls back to rough energy when needed', () => {
    expect(
      resolveStreamingSnapshotNoiseFloorDbfs({
        foreground: { noiseFloorDbfs: -52 },
        rough: { noiseFloorDbfs: -46 },
      } as never),
    ).toBe(-52);

    expect(
      resolveStreamingSnapshotNoiseFloorDbfs({
        foreground: { noiseFloorDbfs: Number.NaN },
        rough: { noiseFloorDbfs: -46 },
      } as never),
    ).toBe(-46);

    expect(resolveStreamingSnapshotNoiseFloorDbfs(null)).toBeNull();
  });

  it('derives the live foreground threshold only when the filter is enabled', () => {
    expect(
      resolveStreamingForegroundThresholdDbfs(
        {
          foregroundFilterEnabled: true,
          foregroundMinDb: 8,
        },
        -50,
      ),
    ).toBe(-42);

    expect(
      resolveStreamingForegroundThresholdDbfs(
        {
          foregroundFilterEnabled: false,
          foregroundMinDb: 8,
        },
        -50,
      ),
    ).toBeNull();
  });

  it('computes segment duration from frame bounds and sample rate', () => {
    expect(
      getStreamingSegmentDurationSeconds(
        {
          startFrame: 1600,
          endFrame: 4000,
        } as never,
        16000,
      ),
    ).toBeCloseTo(0.15, 5);

    expect(getStreamingSegmentDurationSeconds(null, 16000)).toBeNull();
    expect(
      getStreamingSegmentDurationSeconds(
        {
          startFrame: 0,
          endFrame: 100,
        } as never,
        0,
      ),
    ).toBeNull();
  });
});
