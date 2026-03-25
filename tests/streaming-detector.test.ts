import { describe, expect, it } from 'vitest';
import { StreamingSpeechDetector } from '@asrjs/speech-recognition/realtime';

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

  process(): boolean {
    return false;
  }

  getStatus() {
    return this.status;
  }

  findFirstSpeechFrame(): number | null {
    return null;
  }

  hasRecentSpeech(): boolean {
    return false;
  }

  hasRecentSilence(): boolean {
    return true;
  }

  getWindowSummary() {
    return {
      totalHops: 0,
      speechHopCount: 0,
      nonSpeechHopCount: 0,
      maxConsecutiveSpeech: 0,
      maxProbability: 0,
      recent: [],
    };
  }
}

describe('StreamingSpeechDetector', () => {
  it('splits long speech at max duration and immediately continues with a new active segment', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 16,
        maxSegmentDurationMs: 48,
        minEnergyPerSecond: 0,
        minEnergyIntegral: 0,
        useAdaptiveEnergyThresholds: false,
        tenVadEnabled: false,
      },
    });

    const events: Array<{ type: string; payload: any }> = [];
    detector.subscribe((event) => events.push(event as any));
    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.12), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 256 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 512 });

    const segmentEvent = events.find((event) => event.type === 'segment-ready');
    expect(segmentEvent).toBeTruthy();
    expect(segmentEvent?.payload.reason).toBe('max-duration');
    expect(detector.getSnapshot().activeSegment).not.toBeNull();
  });

  it('reports degraded TEN-VAD status but still segments on the rough gate when init fails', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 16,
        minEnergyPerSecond: 0,
        minEnergyIntegral: 0,
        useAdaptiveEnergyThresholds: false,
        tenVadEnabled: true,
      },
      tenVadFactory: () => new FakeTenVad({ failInit: true }) as any,
    });

    const events: Array<{ type: string; payload: any }> = [];
    detector.subscribe((event) => events.push(event as any));
    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.12), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 256 });

    const snapshot = detector.getSnapshot();
    expect(snapshot.tenVad.state).toBe('degraded');
    expect(snapshot.gate.effectiveMode).toBe('rough-only');
    expect(events.some((event) => event.type === 'segment-ready')).toBe(true);
  });

  it('keeps requested TEN-VAD-only mode for UI state but runs detection in rough-only mode', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        gateMode: 'ten-vad-only',
        tenVadEnabled: true,
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 16,
        minEnergyPerSecond: 0,
        minEnergyIntegral: 0,
        useAdaptiveEnergyThresholds: false,
      },
      tenVadFactory: () => new FakeTenVad() as any,
    });

    await detector.start({ sampleRate: 16000 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 0 });

    const snapshot = detector.getSnapshot();
    expect(snapshot.gate.requestedMode).toBe('ten-vad-only');
    expect(snapshot.gate.effectiveMode).toBe('rough-only');
    expect(snapshot.activeSegment).not.toBeNull();
    expect(snapshot.warnings.some((warning) => warning.includes('ignored'))).toBe(true);
  });

  it('still updates TEN-VAD diagnostics config when the detector config changes', async () => {
    let fakeTenVad: FakeTenVad | null = null;
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        tenVadEnabled: true,
      },
      tenVadFactory: () => {
        fakeTenVad = new FakeTenVad() as any;
        return fakeTenVad as any;
      },
    });

    await detector.start({ sampleRate: 16000 });
    fakeTenVad?.updateCalls.splice(0);

    detector.updateConfig({
      energyThreshold: 0.12,
    });

    expect(fakeTenVad?.updateCalls).toHaveLength(1);
    expect(fakeTenVad?.updateCalls[0]).toMatchObject({
      threshold: 0.5,
      sampleRate: 16000,
      hopSize: 256,
    });
  });

  it('updates the adaptive noise floor during silence and freezes it during speech', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        initialNoiseFloor: 0.001,
        fastAdaptationRate: 0.5,
        slowAdaptationRate: 0.5,
        minBackgroundDurationSec: 0,
        tenVadEnabled: false,
      },
    });

    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.004), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.004), { startFrame: 256 });
    const silenceFloorDbfs = detector.getSnapshot().foreground.noiseFloorDbfs;

    detector.processChunk(createChunk(256, 0.12), { startFrame: 512 });
    const speechFloorDbfs = detector.getSnapshot().foreground.noiseFloorDbfs;

    expect(silenceFloorDbfs).toBeGreaterThan(-60);
    expect(speechFloorDbfs).toBeCloseTo(silenceFloorDbfs, 1);
  });

  it('rejects segments that fail the final minimum-duration gate', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        minSpeechDurationMs: 64,
        minSilenceDurationMs: 16,
        minEnergyPerSecond: 0,
        minEnergyIntegral: 0,
        useAdaptiveEnergyThresholds: false,
        tenVadEnabled: false,
      },
    });

    await detector.start({ sampleRate: 16000 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 256 });

    expect(detector.getSnapshot().acceptance).toMatchObject({
      accepted: false,
      reason: 'too-short',
    });
  });

  it('rejects strong but very short segments when total normalized energy is still too low', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 16,
        minEnergyPerSecond: 0,
        minEnergyIntegral: 22,
        useAdaptiveEnergyThresholds: false,
        tenVadEnabled: false,
      },
    });

    await detector.start({ sampleRate: 16000 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 256 });

    expect(detector.getSnapshot().acceptance).toMatchObject({
      accepted: false,
      reason: 'low-energy-integral',
    });
  });

  it('stores extracted bounds separately from logical bounds after a forced split', async () => {
    const detector = new StreamingSpeechDetector({
      profileId: 'generic-streaming',
      config: {
        analysisWindowMs: 16,
        energySmoothingWindows: 1,
        energyThreshold: 0.08,
        prerollMs: 16,
        overlapDurationMs: 16,
        minSpeechDurationMs: 16,
        minSilenceDurationMs: 16,
        maxSegmentDurationMs: 64,
        minEnergyPerSecond: 0,
        minEnergyIntegral: 0,
        useAdaptiveEnergyThresholds: false,
        tenVadEnabled: false,
      },
    });

    await detector.start({ sampleRate: 16000 });

    detector.processChunk(createChunk(256, 0.12), { startFrame: 0 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 256 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 512 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 768 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 1024 });
    detector.processChunk(createChunk(256, 0.12), { startFrame: 1280 });
    detector.processChunk(createChunk(256, 0.001), { startFrame: 1536 });

    const segments = detector.getSnapshot().recentSegments;
    expect(segments).toHaveLength(2);
    const secondSegment = segments[1]!;

    expect(secondSegment.startFrame).toBeLessThan(secondSegment.metadata.logicalStartFrame);
    expect(secondSegment.endFrame).toBeGreaterThanOrEqual(secondSegment.metadata.logicalEndFrame);
  });
});
