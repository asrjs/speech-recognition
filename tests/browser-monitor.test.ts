import {
  createBrowserRealtimeMonitor,
  type BrowserRealtimeStarterSnapshot,
} from '@asrjs/speech-recognition/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

function createSnapshot(revision: number): BrowserRealtimeStarterSnapshot {
  return {
    state: 'idle',
    sampleRate: 16000,
    profileId: 'generic-streaming',
    config: {
      sampleRate: 16000,
      chunkDurationMs: 16,
      gateMode: 'rough-only',
      ringBufferDurationMs: 12000,
      analysisWindowMs: 80,
      energySmoothingDurationMs: 480,
      energySmoothingWindows: 6,
      prerollMs: 120,
      overlapDurationMs: 80,
      speechHangoverMs: 160,
      minSpeechDurationMs: 240,
      minSilenceDurationMs: 400,
      maxSegmentDurationMs: 4800,
      energyThreshold: 0.08,
      minSpeechLevelDbfs: -22,
      useSnrGate: false,
      snrThreshold: 3,
      minSnrThreshold: 1,
      energyRiseThreshold: 0.08,
      maxOnsetLookbackChunks: 6,
      defaultOnsetLookbackChunks: 4,
      maxHistoryChunks: 20,
      initialNoiseFloor: 0.005,
      fastAdaptationRate: 0.15,
      slowAdaptationRate: 0.05,
      minBackgroundDurationSec: 1,
      levelWindowMs: 480,
      minEnergyIntegral: 22,
      minEnergyPerSecond: 5,
      useAdaptiveEnergyThresholds: true,
      adaptiveEnergyIntegralFactor: 25,
      adaptiveEnergyPerSecondFactor: 10,
      minAdaptiveEnergyIntegral: 3,
      minAdaptiveEnergyPerSecond: 1,
      tenVadEnabled: true,
      tenVadThreshold: 0.5,
      tenVadConfirmationWindowMs: 192,
      tenVadHangoverMs: 320,
      tenVadMinSpeechDurationMs: 240,
      tenVadMinSilenceDurationMs: 80,
      tenVadSpeechPaddingMs: 48,
      foregroundFilterEnabled: true,
      foregroundMinDb: 8,
      foregroundOnsetMinDb: 10,
      foregroundOnsetWindowMs: 192,
      foregroundShortSpeechMs: 240,
      foregroundLongSpeechMs: 1200,
      foregroundLongExcessDbMs: 1800,
    },
    waveform: {
      startFrame: 0,
      endFrame: 256,
      minMax: new Float32Array([0, 0]),
    },
    activeSegment: null,
    pendingSegmentStartFrame: null,
    recentSegments: [],
    recentDecisions: [],
    acceptance: { revision },
    gate: {
      requestedMode: 'rough-only',
      effectiveMode: 'rough-only',
      tenVadReady: true,
    },
    foreground: {
      enabled: true,
      noiseFloor: 0.005,
      noiseFloorDbfs: -46,
      energyThreshold: 0.08,
      energyThresholdDbfs: -22,
      minSpeechDurationMs: 240,
      minEnergyPerSecond: 5,
      minEnergyIntegral: 22,
      adaptiveEnergyThresholdsEnabled: true,
      activeResult: null,
      lastResult: null,
    },
    rough: {
      energy: 0,
      snr: 0,
      noiseFloor: 0.005,
      noiseFloorDbfs: -46,
      threshold: 0.08,
      thresholdDbfs: -22,
      levelDbfs: -100,
      levelWindowRms: 0,
      levelWindowDbfs: -100,
      levelWindowMs: 480,
      energyPass: false,
      candidateReason: 'none',
      snrThreshold: 3,
      minSnrThreshold: 1,
      useSnrGate: false,
      snrPass: false,
      isSpeech: false,
      recent: [],
      timeline: [],
    },
    tenVad: {
      state: 'ready',
      error: null,
      probability: 0,
      speaking: false,
      threshold: 0.5,
    },
    warnings: [],
    error: null,
    vadBuffer: {
      totalHops: 0,
      speechHopCount: 0,
      nonSpeechHopCount: 0,
      maxConsecutiveSpeech: 0,
      maxProbability: 0,
      recent: [],
      timeline: [],
    },
    plot: {
      startFrame: -256,
      endFrame: 0,
      chunkFrames: 256,
      chunkDurationMs: 16,
      pointCount: 1,
      filledPointCount: 0,
      padPoints: 1,
      livePointIndex: -1,
      gateMode: 'rough-only',
      columns: [
        {
          index: 0,
          hasData: false,
          waveformMin: 0,
          waveformMax: 0,
          roughEnergy: 0,
          roughSpeechRatio: 0,
          roughIsSpeech: false,
          roughPass: false,
          vadProbability: 0,
          vadSpeechRatio: 0,
          vadSpeaking: false,
          tenVadPass: false,
          detectorPass: false,
          activeSegment: false,
          recentSegment: false,
        },
      ],
    },
  };
}

describe('browser realtime monitor', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('coalesces frequent metrics events into fewer snapshot publications', () => {
    vi.useFakeTimers();

    let revision = 0;
    const listeners = new Set<(event: { type: 'metrics' | 'speech-start' }) => void>();
    const getSnapshot = vi.fn(() => createSnapshot(revision));
    const source = {
      getSnapshot,
      subscribe(listener: (event: { type: 'metrics' | 'speech-start' }) => void) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };

    const monitor = createBrowserRealtimeMonitor(source, { frameIntervalMs: 50 });
    const received: number[] = [];
    const unsubscribe = monitor.subscribe((snapshot) => {
      received.push(Number(snapshot.acceptance?.revision ?? -1));
    });

    revision = 1;
    listeners.forEach((listener) => listener({ type: 'metrics' }));
    revision = 2;
    listeners.forEach((listener) => listener({ type: 'metrics' }));
    revision = 3;
    listeners.forEach((listener) => listener({ type: 'metrics' }));

    expect(received.at(-1)).toBe(0);
    expect(getSnapshot).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(51);

    expect(received.at(-1)).toBe(3);
    expect(getSnapshot).toHaveBeenCalledTimes(2);

    unsubscribe();
    monitor.dispose();
  });

  it('flushes immediately for non-metrics events', () => {
    vi.useFakeTimers();

    let revision = 0;
    const listeners = new Set<(event: { type: 'metrics' | 'speech-start' }) => void>();
    const source = {
      getSnapshot: () => createSnapshot(revision),
      subscribe(listener: (event: { type: 'metrics' | 'speech-start' }) => void) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };

    const monitor = createBrowserRealtimeMonitor(source, { frameIntervalMs: 50 });
    const received: number[] = [];
    monitor.subscribe((snapshot) => {
      received.push(Number(snapshot.acceptance?.revision ?? -1));
    });

    revision = 4;
    listeners.forEach((listener) => listener({ type: 'speech-start' }));

    expect(received.at(-1)).toBe(4);

    monitor.dispose();
  });
});
