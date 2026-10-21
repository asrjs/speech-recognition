import { describe, expect, it, vi } from 'vitest';
import {
  StreamingSpeechDetector,
  STREAMING_GATE_MODES,
  STREAMING_VAD_BACKENDS,
} from '../src/runtime/streaming-detector.js';
import {
  FireRedVadAdapter,
  DEFAULT_FIRERED_MODEL_URL,
} from '../src/runtime/firered-vad-browser.js';
import { createBrowserRealtimeMicrophoneController } from '../src/runtime/browser-controller.js';

function createChunk(length: number, amplitude: number): Float32Array {
  const chunk = new Float32Array(length);
  chunk.fill(amplitude);
  return chunk;
}

class FakeNeuralVad {
  public status = {
    state: 'ready' as 'idle' | 'initializing' | 'ready' | 'degraded' | 'disabled',
    error: null as string | null,
    probability: 0.9,
    speaking: true,
    threshold: 0.5,
  };
  public speechHops = 10;
  public silenceHops = 10;
  public updateCalls: Array<Record<string, unknown>> = [];

  constructor(statusOverrides: Partial<typeof FakeNeuralVad.prototype.status> = {}) {
    Object.assign(this.status, statusOverrides);
  }

  subscribe(): () => void {
    return () => undefined;
  }

  async init(): Promise<void> {
    if (this.status.state === 'degraded') {
      throw new Error('VAD model initialization failed');
    }
  }

  async reset(): Promise<void> {}
  async dispose(): Promise<void> {}

  updateConfig(config?: Record<string, unknown>): void {
    if (config) {
      this.updateCalls.push(config);
    }
  }

  process(): boolean {
    return this.status.speaking;
  }

  getStatus() {
    return this.status;
  }

  findFirstSpeechFrame(startFrame: number): number | null {
    return startFrame;
  }

  hasRecentSpeech(): boolean {
    return this.status.speaking;
  }

  hasRecentSilence(): boolean {
    return !this.status.speaking;
  }

  getWindowSummary() {
    return {
      totalHops: 10,
      speechHopCount: this.status.speaking ? 10 : 0,
      nonSpeechHopCount: this.status.speaking ? 0 : 10,
      maxConsecutiveSpeech: this.status.speaking ? 10 : 0,
      maxProbability: this.status.probability,
      recent: [],
    };
  }
}

describe('StreamingSpeechDetector Neural VAD & Honest Controller', () => {
  describe('1. Effective Gate Mode Resolution', () => {
    it.each([
      ['rough-only', 'rough-only'], ['hybrid', 'hybrid'], ['vad-only', 'vad-only'],
      ['ten-vad-only', 'vad-only'], ['rough-and-ten-vad', 'hybrid'],
    ] as const)('resolves public mode %s to %s with a ready VAD', async (gateMode, expected) => {
      const detector = new StreamingSpeechDetector({
        config: { gateMode, tenVadEnabled: true },
        tenVadFactory: () => new FakeNeuralVad() as any,
      });
      await detector.start({ sampleRate: 16000 });
      expect(detector.getSnapshot().gate.requestedMode).toBe(gateMode);
      expect(detector.getSnapshot().gate.effectiveMode).toBe(expected);
      await detector.dispose();
    });
    it('returns rough-only when VAD is disabled via config', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          gateMode: 'hybrid',
          tenVadEnabled: false,
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(snapshot.gate.requestedMode).toBe('hybrid');
      expect(snapshot.gate.effectiveMode).toBe(STREAMING_GATE_MODES.ROUGH_ONLY);
    });

    it('returns rough-only when vadMode is explicitly disabled', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          gateMode: 'hybrid',
          tenVadEnabled: true,
          // @ts-expect-error test vadMode override
          vadMode: 'disabled',
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(snapshot.gate.requestedMode).toBe('hybrid');
      expect(snapshot.gate.effectiveMode).toBe(STREAMING_GATE_MODES.ROUGH_ONLY);
    });

    it('returns rough-only when VAD is degraded or fails init', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'degraded', error: 'Weights corrupted' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          gateMode: 'hybrid',
          tenVadEnabled: true,
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(snapshot.gate.requestedMode).toBe('hybrid');
      expect(snapshot.gate.effectiveMode).toBe(STREAMING_GATE_MODES.ROUGH_ONLY);
      expect(snapshot.tenVad.state).toBe('degraded');
    });

    it('returns hybrid when VAD is ready and gateMode is hybrid', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          gateMode: 'hybrid',
          tenVadEnabled: true,
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(snapshot.gate.requestedMode).toBe('hybrid');
      expect(snapshot.gate.effectiveMode).toBe('hybrid');
      expect(snapshot.gate.tenVadReady).toBe(true);
    });

    it('returns vad-only when VAD is ready and gateMode is vad-only', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          gateMode: 'vad-only',
          tenVadEnabled: true,
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(snapshot.gate.requestedMode).toBe('vad-only');
      expect(snapshot.gate.effectiveMode).toBe('vad-only');
    });

    it('falls back to rough-only when requested gateMode is rough-only even if VAD is ready', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          gateMode: 'rough-only',
          tenVadEnabled: true,
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(snapshot.gate.requestedMode).toBe('rough-only');
      expect(snapshot.gate.effectiveMode).toBe('rough-only');
    });
  });

  describe('2. Continuous Utterance Soft-Split & Trailing Short Segment Flush', () => {
    it('flushes 5120ms max-duration segment and preserves trailing ~210ms word without dropping as too-short', async () => {
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          sampleRate: 16000,
          chunkDurationMs: 16,
          analysisWindowMs: 16,
          energySmoothingWindows: 1,
          energyThreshold: 0.05,
          minSpeechDurationMs: 240, // 240ms standard threshold
          minSilenceDurationMs: 80,
          maxSegmentDurationMs: 5120, // 5120ms max duration
          overlapDurationMs: 80,
          prerollMs: 80,
          foregroundFilterEnabled: true,
          minEnergyPerSecond: 0,
          minEnergyIntegral: 0,
          useAdaptiveEnergyThresholds: false,
          tenVadEnabled: false,
        },
      });

      const readySegments: any[] = [];
      detector.subscribe((event) => {
        if (event.type === 'segment-ready') {
          readySegments.push(event.payload);
        }
      });

      await detector.start({ sampleRate: 16000 });

      // 5.33 seconds total = 5330ms
      // 5120ms speech = 320 chunks of 256 samples (16ms each)
      const chunkSize = 256;
      const initialSpeechChunks = 320; // 320 * 16ms = 5120ms
      const speechAmplitude = 0.15;

      for (let i = 0; i < initialSpeechChunks; i++) {
        detector.processChunk(createChunk(chunkSize, speechAmplitude), {
          startFrame: i * chunkSize,
        });
      }

      // After 320 chunks, exactly 5120ms of speech processed -> first segment must have flushed as 'max-duration'
      expect(readySegments).toHaveLength(1);
      const firstSegment = readySegments[0];
      expect(firstSegment.reason).toBe('max-duration');
      expect(firstSegment.metadata.logicalEndFrame - firstSegment.metadata.logicalStartFrame).toBe(
        5120 * 16,
      );

      // Trailing word ("country."): ~208ms = 13 chunks of 256 samples (13 * 16ms = 208ms)
      // 208ms is < minSpeechDurationMs (240ms), so previously this would have been rejected as 'too-short'!
      const trailingChunks = 13;
      for (let i = 0; i < trailingChunks; i++) {
        const frame = (initialSpeechChunks + i) * chunkSize;
        detector.processChunk(createChunk(chunkSize, speechAmplitude), {
          startFrame: frame,
        });
      }

      // Speech ends, followed by silence (e.g., 10 chunks of 256 samples = 160ms of silence > minSilenceDurationMs 80ms)
      const silenceChunks = 10;
      for (let i = 0; i < silenceChunks; i++) {
        const frame = (initialSpeechChunks + trailingChunks + i) * chunkSize;
        detector.processChunk(createChunk(chunkSize, 0.0001), {
          startFrame: frame,
        });
      }

      // Second segment MUST be accepted and flushed with reason 'silence'
      expect(readySegments).toHaveLength(2);
      const secondSegment = readySegments[1];
      expect(secondSegment.reason).toBe('silence');
      expect(secondSegment.metadata.filter.accepted).toBe(true);
      expect(secondSegment.metadata.filter.reason).toBe('accepted');
      expect(secondSegment.metadata.filter.durationMs).toBeGreaterThanOrEqual(200);
      expect(secondSegment.metadata.filter.durationMs).toBeLessThan(350);
    });

    it('flushes trailing continuation segment on manual flush without dropping as too-short', async () => {
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          sampleRate: 16000,
          chunkDurationMs: 16,
          analysisWindowMs: 16,
          energySmoothingWindows: 1,
          energyThreshold: 0.05,
          minSpeechDurationMs: 240,
          maxSegmentDurationMs: 5120,
          overlapDurationMs: 80,
          foregroundFilterEnabled: true,
          minEnergyPerSecond: 0,
          minEnergyIntegral: 0,
          useAdaptiveEnergyThresholds: false,
          tenVadEnabled: false,
        },
      });

      const readySegments: any[] = [];
      detector.subscribe((event) => {
        if (event.type === 'segment-ready') {
          readySegments.push(event.payload);
        }
      });

      await detector.start({ sampleRate: 16000 });
      const chunkSize = 256;
      for (let i = 0; i < 320; i++) {
        detector.processChunk(createChunk(chunkSize, 0.15), { startFrame: i * chunkSize });
      }
      expect(readySegments).toHaveLength(1);
      expect(readySegments[0].reason).toBe('max-duration');

      // Trailing 208ms speech (13 chunks)
      for (let i = 0; i < 13; i++) {
        detector.processChunk(createChunk(chunkSize, 0.15), {
          startFrame: (320 + i) * chunkSize,
        });
      }

      // Manual flush instead of silence
      const flushed = detector.flush('manual');
      expect(flushed).not.toBeNull();
      expect(flushed?.reason).toBe('manual');
      expect(flushed?.metadata.filter.accepted).toBe(true);
      expect(readySegments).toHaveLength(2);
    });
  });

  describe('3. Honest Status Messages', () => {
    function createMockStarter(snapshotData: Record<string, any>) {
      return {
        subscribe: () => () => undefined,
        start: vi.fn(async () => undefined),
        processChunk: vi.fn(),
        flush: vi.fn(),
        stop: vi.fn(async () => null),
        updateConfig: vi.fn(),
        getSnapshot: () => snapshotData,
        dispose: vi.fn(async () => undefined),
      };
    }

    it('reports RoughSpeechGate when VAD is disabled', async () => {
      const starter = createMockStarter({
        tenVad: { state: 'disabled' },
        gate: { effectiveMode: 'rough-only', vadBackend: 'firered-vad' },
        config: { tenVadEnabled: false },
      });

      const statusMessages: string[] = [];
      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        createStarter: () => starter as any,
        createMonitor: () => ({ subscribe: () => () => undefined, getSnapshot: () => null, flush: vi.fn(), dispose: vi.fn() }) as any,
        startCapture: async (_options) => ({
          sampleRate: 16000,
          deviceSampleRate: null,
          contextSampleRate: 16000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {} as MediaStream,
          stop: vi.fn(async () => undefined),
        }),
        onStatus: (msg) => {
          statusMessages.push(msg);
        },
      });

      await controller.start();
      expect(controller.getState().micStatus).toContain('with RoughSpeechGate…');
      expect(controller.getState().micStatus).not.toContain('FireRed VAD segmenter');
      await controller.stop();
    });

    it('reports TEN-VAD when VAD backend is ten-vad and gateMode is hybrid', async () => {
      const starter = createMockStarter({
        tenVad: { state: 'ready' },
        gate: { effectiveMode: 'hybrid', vadBackend: 'ten-vad' },
        config: { tenVadEnabled: true },
      });

      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        createStarter: () => starter as any,
        createMonitor: () => ({ subscribe: () => () => undefined, getSnapshot: () => null, flush: vi.fn(), dispose: vi.fn() }) as any,
        startCapture: async () => ({
          sampleRate: 16000,
          deviceSampleRate: null,
          contextSampleRate: 16000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {} as MediaStream,
          stop: vi.fn(async () => undefined),
        }),
      });

      await controller.start();
      expect(controller.getState().micStatus).toContain('with TEN-VAD…');
      await controller.stop();
    });

    it('reports FireRed VAD when VAD backend is firered-vad and gateMode is hybrid', async () => {
      const starter = createMockStarter({
        tenVad: { state: 'ready' },
        gate: { effectiveMode: 'hybrid', vadBackend: 'firered-vad' },
        config: { tenVadEnabled: true },
      });

      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        createStarter: () => starter as any,
        createMonitor: () => ({ subscribe: () => () => undefined, getSnapshot: () => null, flush: vi.fn(), dispose: vi.fn() }) as any,
        startCapture: async () => ({
          sampleRate: 16000,
          deviceSampleRate: null,
          contextSampleRate: 16000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {} as MediaStream,
          stop: vi.fn(async () => undefined),
        }),
      });

      await controller.start();
      expect(controller.getState().micStatus).toContain('with FireRed VAD…');
      await controller.stop();
    });

    it('reports rough fallback when VAD is degraded', async () => {
      const starter = createMockStarter({
        tenVad: { state: 'degraded' },
        gate: { effectiveMode: 'rough-only', vadBackend: 'firered-vad' },
        config: { tenVadEnabled: true },
      });

      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        createStarter: () => starter as any,
        createMonitor: () => ({ subscribe: () => () => undefined, getSnapshot: () => null, flush: vi.fn(), dispose: vi.fn() }) as any,
        startCapture: async () => ({
          sampleRate: 16000,
          deviceSampleRate: null,
          contextSampleRate: 16000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {} as MediaStream,
          stop: vi.fn(async () => undefined),
        }),
      });

      await controller.start();
      expect(controller.getState().micStatus).toContain('with rough fallback…');
      await controller.stop();
    });

    it('StreamingSpeechDetector records honest backendName decision logs for TEN-VAD', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready' });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          vadBackend: STREAMING_VAD_BACKENDS.TEN_VAD,
          tenVadEnabled: true,
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const snapshot = detector.getSnapshot();
      expect(
        snapshot.recentDecisions.some((decision) =>
          decision.message.includes('TEN-VAD ready for diagnostics'),
        ),
      ).toBe(true);
      expect(
        snapshot.recentDecisions.some((decision) =>
          decision.message.includes('FireRed VAD ready for diagnostics'),
        ),
      ).toBe(false);
    });
  });

  describe('4. Configurable local model URL in FireRedVadAdapter', () => {
    class FakeFireRedWorker {
      public onmessage: ((event: any) => void) | null = null;
      public onerror: ((event: any) => void) | null = null;
      public messages: any[] = [];

      postMessage(message: any) {
        this.messages.push(message);
        if (message.type === 'INIT') {
          this.onmessage?.({
            data: { type: 'INIT', id: message.id, payload: { success: true, version: 'test' } },
          });
          return;
        }
        if (message.type === 'PROCESS') {
          return;
        }
        this.onmessage?.({
          data: {
            type: message.type,
            id: message.id,
            payload: { success: true },
          },
        });
      }

      terminate() {}
    }

    it('exports DEFAULT_FIRERED_MODEL_URL cleanly', () => {
      expect(DEFAULT_FIRERED_MODEL_URL).toBeTypeOf('string');
      expect(DEFAULT_FIRERED_MODEL_URL).toContain('fireredvad_stream_vad_with_cache.onnx');
    });

    it('respects caller-provided modelUrl option in options', async () => {
      const worker = new FakeFireRedWorker();
      const customModelUrl = 'http://localhost:8080/models/custom-firered.onnx';
      const adapter = new FireRedVadAdapter(
        {},
        {
          workerFactory: () => worker,
          modelUrl: customModelUrl,
        },
      );

      await adapter.init();
      const initMessage = worker.messages.find((m) => m.type === 'INIT');
      expect(initMessage).toBeDefined();
      expect(initMessage.payload.modelUrl).toBe(customModelUrl);
      await adapter.dispose();
    });

    it('respects caller-provided modelUrl in config', async () => {
      const worker = new FakeFireRedWorker();
      const customModelUrl = '/local-assets/fireredvad.onnx';
      const adapter = new FireRedVadAdapter(
        {
          modelUrl: customModelUrl,
        },
        {
          workerFactory: () => worker,
        },
      );

      await adapter.init();
      const initMessage = worker.messages.find((m) => m.type === 'INIT');
      expect(initMessage).toBeDefined();
      expect(initMessage.payload.modelUrl).toBe(customModelUrl);
      await adapter.dispose();
    });
  });

  describe('5. Hybrid Mode Speech Continuation with VAD', () => {
    it('keeps active utterance alive in hybrid mode when neural VAD hasRecentSpeech even if energy drops', async () => {
      const fakeVad = new FakeNeuralVad({ state: 'ready', speaking: true });
      const detector = new StreamingSpeechDetector({
        profileId: 'generic-streaming',
        config: {
          sampleRate: 16000,
          chunkDurationMs: 16,
          analysisWindowMs: 16,
          energySmoothingWindows: 1,
          energyThreshold: 0.08,
          minSpeechDurationMs: 32,
          minSilenceDurationMs: 48,
          tenVadEnabled: true,
          gateMode: 'hybrid',
        },
        tenVadFactory: () => fakeVad as any,
      });

      await detector.start({ sampleRate: 16000 });
      const chunkSize = 256;

      // Start speech with strong energy
      detector.processChunk(createChunk(chunkSize, 0.15), { startFrame: 0 });
      detector.processChunk(createChunk(chunkSize, 0.15), { startFrame: 256 });
      expect(detector.getSnapshot().state).toBe('speaking');

      // Now energy drops to near-silence (quiet trailing syllables), but neural VAD still reports speaking = true
      fakeVad.status.speaking = true;
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 512 });
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 768 });
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 1024 });

      // Detector should remain in speaking state because VAD keeps it alive
      expect(detector.getSnapshot().state).toBe('speaking');
      expect(detector.getSnapshot().activeSegment).not.toBeNull();

      // Now neural VAD transitions to silence
      fakeVad.status.speaking = false;
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 1280 });
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 1536 });
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 1792 });
      detector.processChunk(createChunk(chunkSize, 0.001), { startFrame: 2048 });

      // After silence is confirmed, segment finalizes
      const segments = detector.getSnapshot().recentSegments;
      expect(segments).toHaveLength(1);
      expect(segments[0].reason).toBe('silence');
    });
  });
});
