import { createBrowserRealtimeStarter } from '@asrjs/speech-recognition/browser';
import { DEFAULT_STREAMING_DETECTOR_CONFIG } from '../src/runtime/streaming-config.js';
import { describe, expect, it } from 'vitest';

describe('browser realtime starter', () => {
  it('requires a transcribe callback when controllerOptions are provided', () => {
    expect(() =>
      createBrowserRealtimeStarter({
        controllerOptions: {
          finalizeSilenceSeconds: 0.5,
        },
      }),
    ).toThrow('requires transcribe when controllerOptions are provided');
  });

  it('uses the library default ring-buffer duration for the VAD buffer', () => {
    const starter = createBrowserRealtimeStarter();

    expect(starter.vadBuffer.maxEntries).toBe(
      Math.ceil(
        ((DEFAULT_STREAMING_DETECTOR_CONFIG.ringBufferDurationMs / 1000) *
          DEFAULT_STREAMING_DETECTOR_CONFIG.sampleRate) /
          starter.vadBuffer.hopFrames,
      ),
    );
  });

  it('exposes a shared aligned plot raster in each snapshot', () => {
    const starter = createBrowserRealtimeStarter();
    const snapshot = starter.getSnapshot();

    expect(snapshot.plot.pointCount).toBe(
      Math.round(
        DEFAULT_STREAMING_DETECTOR_CONFIG.ringBufferDurationMs /
          snapshot.plot.chunkDurationMs,
      ),
    );
    expect(snapshot.plot.columns).toHaveLength(snapshot.plot.pointCount);
    expect(snapshot.plot.columns[0]).toMatchObject({
      index: 0,
      waveformMin: 0,
      waveformMax: 0,
      roughEnergy: 0,
      vadProbability: 0,
    });
  });
});
