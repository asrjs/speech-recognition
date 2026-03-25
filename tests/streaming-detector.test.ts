import { describe, expect, it } from 'vitest';
import { StreamingSpeechDetector } from '@asrjs/speech-recognition/realtime';
import {
  STREAMING_PROCESSING_SAMPLE_RATE,
  STREAMING_TIMELINE_CHUNK_FRAMES,
} from '@asrjs/speech-recognition/browser';

function createChunk(length: number, amplitude: number): Float32Array {
  const chunk = new Float32Array(length);
  chunk.fill(amplitude);
  return chunk;
}

class FakeTenVad {
  private readonly behavior: Record<string, unknown>;
  private readonly listeners = new Set<(event: { type: 'result'; payload: unknown }) => void>();
  public updateCalls: Array<Record<string, unknown>> = [];
  private status = {
    state: 'ready',
    error: null as string | null,
    probability: 0,
    speaking: false,
    threshold: 0.5,
  };

  constructor(behavior: Record<string, unknown> = {}) {
    this.behavior = behavior;
    if (behavior.failInit) {
      this.status.state = 'idle';
    }
  }

  subscribe(listener: (event: { type: 'result'; payload: unknown }) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async init(): Promise<void> {
    if (this.behavior.failInit) {
      this.status = {
        ...this.status,
        state: 'degraded',
        error: 'boom',
      };
      throw new Error('boom');
    }
  }

  async reset(): Promise<void> {}
  async dispose(): Promise<void> {}
  updateConfig(config: Record<string, unknown> = {}): void {
    this.updateCalls.push(config);
  }
  process(): boolean { return false; }
  getStatus() { return this.status; }
  findFirstSpeechFrame(): number | null { return (this.behavior.startFrame as number | null) ?? null; }

  getWindowSummary() {
    const hasRecentSpeech = Boolean(this.behavior.hasRecentSpeech);
    return {
      totalHops: hasRecentSpeech ? 4 : 0,
      speechHopCount: hasRecentSpeech ? 4 : 0,
      nonSpeechHopCount: hasRecentSpeech ? 0 : 4,
      maxConsecutiveSpeech: hasRecentSpeech ? 4 : 0,
      maxProbability: hasRecentSpeech ? 0.8 : 0,
      recent: [],
    };
  }

  hasRecentSpeech(): boolean {
    return Boolean(this.behavior.hasRecentSpeech);
  }

  hasRecentSilence(): boolean {
    return this.behavior.hasRecentSilence !== false;
  }
}

describe('StreamingSpeechDetector', () => {
  it('splits long speech at the configured max duration', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 192,
        maxSegmentDurationMs: 60,
        tenVadEnabled: false,
      },
    });

    const events: Array<{ type: string; payload: any }> = [];
    detector.subscribe((event) => events.push(event as any));
    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.05), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.05), { startFrame: 256 });
    detector.processChunk(createChunk(256, 0.05), { startFrame: 512 });
    detector.processChunk(createChunk(256, 0.05), { startFrame: 768 });

    const segmentEvent = events.find((event) => event.type === 'segment-ready');
    expect(segmentEvent).toBeTruthy();
    expect(segmentEvent?.payload.reason).toBe('max-duration');
  });

  it('falls back to rough-gate segmentation when TEN-VAD init fails', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 32,
      },
      tenVadFactory: () => new FakeTenVad({ failInit: true }) as any,
    });

    const events: Array<{ type: string; payload: any }> = [];
    detector.subscribe((event) => events.push(event as any));
    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.05), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 256 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 512 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 768 });

    const snapshot = detector.getSnapshot();
    expect(snapshot.tenVad.state).toBe('degraded');
    expect(events.some((event) => event.type === 'segment-ready')).toBe(true);
  });

  it('holds rough onset as a candidate until TEN-VAD confirms speech', async () => {
    let confirmed = false;
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 32,
      },
      tenVadFactory: () =>
        new FakeTenVad({
          get startFrame() {
            return confirmed ? 0 : null;
          },
          get hasRecentSpeech() {
            return confirmed;
          },
        }) as any,
    });

    await detector.start({ sampleRate: 16000 });
    detector.processChunk(createChunk(256, 0.05), { startFrame: 0 });

    let snapshot = detector.getSnapshot();
    expect(snapshot.activeSegment).toBeNull();
    expect(snapshot.pendingSegmentStartFrame).toBeNull();

    confirmed = true;
    detector.processChunk(createChunk(256, 0.05), { startFrame: 256 });
    snapshot = detector.getSnapshot();

    expect(snapshot.activeSegment).not.toBeNull();
    expect(snapshot.pendingSegmentStartFrame).toBeNull();
  });

  it('can segment with TEN-VAD only when that gate mode is selected', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        gateMode: 'ten-vad-only',
        analysisWindowMs: 16,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 32,
      },
      tenVadFactory: () =>
        new FakeTenVad({
          startFrame: 0,
          hasRecentSpeech: true,
        }) as any,
    });

    await detector.start({ sampleRate: 16000 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 0 });

    const snapshot = detector.getSnapshot();
    expect(snapshot.gate.effectiveMode).toBe('ten-vad-only');
    expect(snapshot.activeSegment).not.toBeNull();
  });

  it('keeps the rough onset start when TEN-VAD confirms later in rough-and-ten-vad mode', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        gateMode: 'rough-and-ten-vad',
        analysisWindowMs: 16,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 32,
        prerollMs: 0,
      },
      tenVadFactory: () =>
        new FakeTenVad({
          startFrame: 256,
          hasRecentSpeech: true,
        }) as any,
    });

    await detector.start({ sampleRate: 16000 });
    detector.processChunk(createChunk(256, 0.05), { startFrame: 0 });

    const snapshot = detector.getSnapshot();
    expect(snapshot.gate.effectiveMode).toBe('ten-vad-only');
    expect(snapshot.activeSegment).not.toBeNull();
    expect(snapshot.activeSegment?.startFrame).toBe(0);
  });

  it('lets TEN-VAD keep the tail alive even after rough speech has ended', async () => {
    const tenVadState = {
      hasRecentSpeech: true,
      hasRecentSilence: false,
    };
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        gateMode: 'rough-and-ten-vad',
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 16,
        minSpeechLevelDbfs: -45,
        prerollMs: 0,
      },
      tenVadFactory: () =>
        new FakeTenVad({
          startFrame: 0,
          get hasRecentSpeech() {
            return tenVadState.hasRecentSpeech;
          },
          get hasRecentSilence() {
            return tenVadState.hasRecentSilence;
          },
        }) as any,
    });

    const events: Array<{ type: string; payload: any }> = [];
    detector.subscribe((event) => events.push(event as any));
    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.05), { startFrame: 0 });
    expect(detector.getSnapshot().activeSegment).not.toBeNull();

    detector.processChunk(createChunk(256, 0.001), { startFrame: 256 });

    const segmentEvent = events.find((event) => event.type === 'segment-ready');
    expect(segmentEvent).toBeFalsy();
    expect(detector.getSnapshot().activeSegment).not.toBeNull();
  });

  it('builds waveform snapshots from stable timeline-aligned chunk buckets', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        tenVadEnabled: false,
        ringBufferDurationMs: 8000,
      },
    });

    await detector.start({ sampleRate: STREAMING_PROCESSING_SAMPLE_RATE });

    const chunkCount = 8000 / 16;
    for (let index = 0; index < chunkCount; index += 1) {
      detector.processChunk(createChunk(STREAMING_TIMELINE_CHUNK_FRAMES, 0.25), {
        startFrame: index * STREAMING_TIMELINE_CHUNK_FRAMES,
      });
    }

    const snapshot = detector.getSnapshot();

    expect(snapshot.waveform.minMax.length / 2).toBe(chunkCount);
    expect(snapshot.waveform.endFrame - snapshot.waveform.startFrame).toBe(
      STREAMING_TIMELINE_CHUNK_FRAMES * chunkCount,
    );
  });

  it('does not push unrelated rough-gate config changes into TEN-VAD worker settings', async () => {
    let fakeTenVad: FakeTenVad | null = null;
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 32,
      },
      tenVadFactory: () => {
        fakeTenVad = new FakeTenVad() as any;
        return fakeTenVad as any;
      },
    });

    await detector.start({ sampleRate: 16000 });
    fakeTenVad?.updateCalls.splice(0);

    detector.updateConfig({
      minSpeechLevelDbfs: -55,
    });

    expect(fakeTenVad?.updateCalls).toHaveLength(1);
    expect(fakeTenVad?.updateCalls[0]).toMatchObject({
      threshold: 0.55,
      sampleRate: 16000,
      hopSize: STREAMING_TIMELINE_CHUNK_FRAMES,
    });
  });

  it('rejects quiet completed segments with the foreground filter', async () => {
    const tenVadState = {
      hasRecentSpeech: true,
      hasRecentSilence: false,
    };
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        gateMode: 'ten-vad-only',
        prerollMs: 0,
        foregroundMinDb: 18,
        foregroundOnsetMinDb: 18,
        foregroundLongMinDb: 18,
      },
      tenVadFactory: () =>
        new FakeTenVad({
          startFrame: 0,
          get hasRecentSpeech() {
            return tenVadState.hasRecentSpeech;
          },
          get hasRecentSilence() {
            return tenVadState.hasRecentSilence;
          },
        }) as any,
    });

    const events: Array<{ type: string; payload: any }> = [];
    detector.subscribe((event) => events.push(event as any));
    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.01), { startFrame: 0 });
    tenVadState.hasRecentSpeech = false;
    tenVadState.hasRecentSilence = true;
    detector.processChunk(createChunk(256, 0.01), { startFrame: 256 });
    detector.processChunk(createChunk(256, 0.01), { startFrame: 512 });

    expect(events.some((event) => event.type === 'segment-ready')).toBe(false);
    expect(detector.getSnapshot().acceptance).toMatchObject({
      accepted: false,
    });
  });
});
