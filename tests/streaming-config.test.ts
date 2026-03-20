import {
  isStreamingConfigEqual,
  mergeStreamingConfig,
} from '@asrjs/speech-recognition/realtime';
import { describe, expect, it } from 'vitest';

describe('streaming config helpers', () => {
  it('drops runtime-only energyThreshold after normalizing overrides', () => {
    const merged = mergeStreamingConfig('generic-streaming', {
      energyThreshold: 0.01,
    });

    expect('energyThreshold' in merged).toBe(false);
    expect(
      isStreamingConfigEqual(merged, mergeStreamingConfig('generic-streaming', {
        minSpeechLevelDbfs: merged.minSpeechLevelDbfs,
      })),
    ).toBe(true);
  });
});
