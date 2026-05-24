import {
  createBrowserRealtimeStarter,
  FireRedVadAdapter,
  TenVadAdapter,
} from '@asrjs/speech-recognition/browser';
import { DEFAULT_STREAMING_DETECTOR_CONFIG } from '../src/runtime/streaming-config.js';
import { describe, expect, it } from 'vitest';

class FakeFireRedWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  postMessage(message: { type: string; payload?: any; id?: number }) {
    if (message.type === 'INIT') {
      this.onmessage?.({
        data: { type: 'INIT', id: message.id, payload: { success: true, version: 'test' } },
      } as MessageEvent);
      return;
    }

    if (message.type === 'PROCESS') {
      this.onmessage?.({
        data: {
          type: 'RESULT',
          payload: {
            probabilities: new Float32Array([0.2, 0.8, 0.3, 0.84]),
            flags: new Uint8Array([0, 1, 0, 1]),
            globalSampleOffset: message.payload.globalSampleOffset,
            hopCount: 4,
          },
        },
      } as MessageEvent);
      return;
    }

    this.onmessage?.({
      data: { type: message.type, id: message.id, payload: { success: true } },
    } as MessageEvent);
  }

  terminate() {}
}

class FakeDelayedFireRedWorker extends FakeFireRedWorker {
  private processCount = 0;

  override postMessage(message: { type: string; payload?: any; id?: number }) {
    if (message.type !== 'PROCESS') {
      super.postMessage(message);
      return;
    }

    this.processCount += 1;
    if (this.processCount === 1) {
      return;
    }

    super.postMessage(message);
  }
}

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

  it('aggregates FireRed 10ms VAD hops into the configured 20ms visual buckets', async () => {
    const starter = createBrowserRealtimeStarter({
      config: {
        tenVadEnabled: true,
        vadBackend: 'firered-vad',
      },
      fireRedVadOptions: {
        workerFactory: () => new FakeFireRedWorker(),
      },
    });

    await starter.start({ sampleRate: 16000 });
    starter.processChunk(new Float32Array(256), { startFrame: 0, endFrame: 256 });

    const snapshot = starter.getSnapshot();
    expect(starter.vadBuffer.hopFrames).toBe(320);
    expect(snapshot.vadBuffer.totalHops).toBe(2);
    expect(snapshot.vadBuffer.maxProbability).toBeCloseTo(0.84, 3);

    await starter.dispose();
  });

  it('aligns FireRed visual VAD buckets to absolute audio frame offsets', async () => {
    const starter = createBrowserRealtimeStarter({
      config: {
        tenVadEnabled: true,
        vadBackend: 'firered-vad',
      },
      fireRedVadOptions: {
        workerFactory: () => new FakeDelayedFireRedWorker(),
      },
    });

    await starter.start({ sampleRate: 16000 });
    starter.processChunk(new Float32Array(640), { startFrame: 0, endFrame: 640 });
    starter.processChunk(new Float32Array(256), { startFrame: 640, endFrame: 896 });

    expect(starter.vadBuffer.getTimeline(0, 1280, 4)).toEqual([
      expect.objectContaining({ startFrame: 0, endFrame: 320, probability: 0 }),
      expect.objectContaining({ startFrame: 320, endFrame: 640, probability: 0 }),
      expect.objectContaining({ startFrame: 640, endFrame: 960, probability: expect.closeTo(0.8) }),
      expect.objectContaining({ startFrame: 960, endFrame: 1280, probability: expect.closeTo(0.84) }),
    ]);

    await starter.dispose();
  });

  it('rebuilds the VAD adapter and visual buffer when the backend changes', async () => {
    const starter = createBrowserRealtimeStarter({
      config: {
        tenVadEnabled: true,
        vadBackend: 'firered-vad',
      },
      fireRedVadOptions: {
        workerFactory: () => new FakeFireRedWorker(),
      },
    });

    expect(starter.tenVad).toBeInstanceOf(FireRedVadAdapter);
    expect(starter.vadBuffer.hopFrames).toBe(320);

    starter.updateConfig({ vadBackend: 'ten-vad' });

    let snapshot = starter.getSnapshot();
    expect(snapshot.config.vadBackend).toBe('ten-vad');
    expect(snapshot.config.vadHopDurationMs).toBe(snapshot.config.chunkDurationMs);
    expect(snapshot.config.vadVisualBucketDurationMs).toBe(snapshot.config.chunkDurationMs);
    expect(starter.tenVad).toBeInstanceOf(TenVadAdapter);
    expect(starter.vadBuffer.hopFrames).toBe(256);

    starter.updateConfig({ vadBackend: 'firered-vad' });

    snapshot = starter.getSnapshot();
    expect(snapshot.config.vadBackend).toBe('firered-vad');
    expect(snapshot.config.vadHopDurationMs).toBe(10);
    expect(snapshot.config.vadVisualBucketDurationMs).toBe(20);
    expect(starter.tenVad).toBeInstanceOf(FireRedVadAdapter);
    expect(starter.vadBuffer.hopFrames).toBe(320);

    await starter.dispose();
  });
});
