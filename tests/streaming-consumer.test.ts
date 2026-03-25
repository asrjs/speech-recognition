import {
  estimateStreamingReleaseMs,
  getStreamingSegmentDurationSeconds,
  listStreamingControlGroups,
  resolveStreamingControlGroups,
  resolveStreamingForegroundThresholdDbfs,
  resolveStreamingOnsetThresholdDbfs,
  resolveStreamingSnapshotNoiseFloorDbfs,
} from '@asrjs/speech-recognition/realtime';
import { describe, expect, it } from 'vitest';

describe('streaming consumer helpers', () => {
  it('exposes stable Parakeet-style detector tuning groups for consumer apps', () => {
    const groups = listStreamingControlGroups();
    const resolvedGroups = resolveStreamingControlGroups();

    expect(groups.map((group) => group.id)).toEqual([
      'segmenter',
      'timing',
      'acceptance',
      'adaptation',
      'acceptance-advanced',
    ]);
    expect(resolvedGroups).toHaveLength(5);
    expect(resolvedGroups.every((group) => group.controls.length > 0)).toBe(true);
    expect(
      resolvedGroups.every((group, index) =>
        group.controls.every((control) => groups[index]?.fields.includes(control.field)),
      ),
    ).toBe(true);
  });

  it('estimates release timing from silence hold versus extraction hangover', () => {
    expect(
      estimateStreamingReleaseMs({
        minSilenceDurationMs: 400,
        speechHangoverMs: 160,
      }),
    ).toBe(400);

    expect(estimateStreamingReleaseMs(null)).toBeNull();
    expect(
      estimateStreamingReleaseMs({
        minSilenceDurationMs: Number.NaN,
        speechHangoverMs: 160,
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

  it('derives the displayed live speech threshold from minSpeechLevelDbfs', () => {
    expect(
      resolveStreamingForegroundThresholdDbfs(
        {
          minSpeechLevelDbfs: -22,
        },
        -50,
      ),
    ).toBe(-22);

    expect(
      resolveStreamingOnsetThresholdDbfs(
        {
          minSpeechLevelDbfs: -22,
        },
        -50,
      ),
    ).toBe(-22);

    expect(resolveStreamingForegroundThresholdDbfs(null, -50)).toBeNull();
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
