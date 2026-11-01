import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SincResampler, StatefulLinearResampler } from '../src/audio/resample.js';
import {
  createStreamingTelemetryCollector,
  getGlobalStreamingTelemetry,
  resetGlobalStreamingTelemetry,
} from '../src/runtime/streaming-telemetry.js';
import { TenVadAdapter } from '../src/runtime/ten-vad-browser.js';
import { StreamingSpeechDetector } from '../src/runtime/streaming-detector.js';
import { createBrowserRealtimeStarter } from '../src/runtime/browser-realtime.js';
import type { TenVadRequestMessage } from '../src/runtime/ten-vad-protocol.js';

class MockReliabilityWorker {
  onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  messages: TenVadRequestMessage[] = [];
  terminated = false;
  private readonly mode: 'normal' | 'fail-init' | 'crash-on-process' | 'hold-init';

  constructor(mode: 'normal' | 'fail-init' | 'crash-on-process' | 'hold-init' = 'normal') {
    this.mode = mode;
  }

  postMessage(message: TenVadRequestMessage): void {
    this.messages.push(message);

    if (this.mode === 'hold-init') {
      return;
    }

    if (message.type === 'INIT') {
      if (this.mode === 'fail-init') {
        queueMicrotask(() => {
          this.onmessage?.({
            data: {
              type: 'ERROR',
              id: message.id,
              payload: 'Failed to fetch TEN-VAD script: 404 Not Found',
            },
          } as MessageEvent<unknown>);
        });
      } else {
        queueMicrotask(() => {
          this.onmessage?.({
            data: {
              type: 'INIT',
              id: message.id,
              payload: { success: true, version: '1.0.0-mock' },
            },
          } as MessageEvent<unknown>);
        });
      }
      return;
    }

    if (message.type === 'PROCESS') {
      if (this.mode === 'crash-on-process') {
        queueMicrotask(() => {
          this.onerror?.({ message: 'Mock Worker Panic: memory corruption' } as ErrorEvent);
        });
        return;
      }
      const samples = message.payload.samples;
      const hopCount = Math.max(1, Math.floor(samples.length / 256));
      const probabilities = new Float32Array(hopCount).fill(0.88);
      const flags = new Uint8Array(hopCount).fill(1);
      queueMicrotask(() => {
        this.onmessage?.({
          data: {
            type: 'RESULT',
            payload: {
              probabilities,
              flags,
              globalSampleOffset: message.payload.globalSampleOffset,
              hopCount,
            },
          },
        } as MessageEvent<unknown>);
      });
      return;
    }

    if (message.type === 'RESET' || message.type === 'DISPOSE') {
      queueMicrotask(() => {
        this.onmessage?.({
          data: {
            type: message.type,
            id: message.id,
            payload: { success: true },
          },
        } as MessageEvent<unknown>);
      });
    }
  }

  terminate(): void {
    this.terminated = true;
    this.onmessage = null;
    this.onerror = null;
  }
}

function generateSyntheticBlock(
  type: 'speech' | 'silence' | 'empty' | 'burst',
  sampleRate: number,
): Float32Array {
  if (type === 'empty') {
    return new Float32Array(0);
  }
  if (type === 'burst') {
    const samples = new Float32Array(8192);
    for (let i = 0; i < samples.length; i += 1) {
      samples[i] = 0.35 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }
    return samples;
  }
  if (type === 'silence') {
    const length = Math.round(sampleRate * 0.5);
    const samples = new Float32Array(length);
    for (let i = 0; i < samples.length; i += 1) {
      samples[i] = Math.sin(i * 0.05) * 0.0001;
    }
    return samples;
  }
  // speech block (1.5s)
  const length = Math.round(sampleRate * 1.5);
  const samples = new Float32Array(length);
  for (let i = 0; i < samples.length; i += 1) {
    const t = i / sampleRate;
    const envelope = 0.5 * (1 + Math.sin(2 * Math.PI * 4 * t));
    const tone =
      0.5 * Math.sin(2 * Math.PI * 300 * t) +
      0.3 * Math.sin(2 * Math.PI * 800 * t) +
      0.2 * Math.sin(2 * Math.PI * 2400 * t);
    samples[i] = tone * envelope;
  }
  return samples;
}

describe('G2 Streaming Reliability, Telemetry & Soak Qualification', () => {
  beforeEach(() => {
    resetGlobalStreamingTelemetry();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Synthetic PCM Soak Tests (44.1 kHz & 48 kHz)', () => {
    const jitterSchedule = [1, 7, 64, 128, 256, 511, 1024, 4096];

    it.each([
      {
        rate: 48000,
        name: 'StatefulLinearResampler',
        create: () => new StatefulLinearResampler(48000, 16000),
      },
      { rate: 48000, name: 'SincResampler', create: () => new SincResampler(48000, 16000) },
      {
        rate: 44100,
        name: 'StatefulLinearResampler',
        create: () => new StatefulLinearResampler(44100, 16000),
      },
      { rate: 44100, name: 'SincResampler', create: () => new SincResampler(44100, 16000) },
    ])(
      'executes accelerated soak schedule at $rate Hz using $name with zero unaccounted dropped samples',
      ({ rate, create }) => {
        const telemetry = createStreamingTelemetryCollector();
        const resampler = create();
        const blocks: ('speech' | 'silence' | 'empty' | 'burst')[] = [
          'speech',
          'silence',
          'empty',
          'burst',
        ];

        let jitterIndex = 0;
        let totalInputSamples = 0;
        let totalResampledSamples = 0;

        // Run 4 full alternating cycles across all conditions and jitter chunks
        for (let cycle = 0; cycle < 4; cycle += 1) {
          for (const blockType of blocks) {
            const blockPcm = generateSyntheticBlock(blockType, rate);
            let offset = 0;

            if (blockPcm.length === 0) {
              // Empty chunk test
              telemetry.recordInputSamples(0);
              const out = resampler.process(new Float32Array(0));
              expect(out.length).toBe(0);
              continue;
            }

            while (offset < blockPcm.length) {
              const chunkSize = jitterSchedule[jitterIndex % jitterSchedule.length]!;
              jitterIndex += 1;
              const end = Math.min(offset + chunkSize, blockPcm.length);
              const chunk = blockPcm.subarray(offset, end);
              offset = end;

              telemetry.recordInputSamples(chunk.length);
              totalInputSamples += chunk.length;

              const resampled = resampler.process(chunk);
              telemetry.recordAcceptedSamples(resampled.length);
              telemetry.recordEmittedSamples(resampled.length);
              totalResampledSamples += resampled.length;

              // Ensure no NaNs or Infinities
              for (let i = 0; i < resampled.length; i += 1) {
                expect(Number.isFinite(resampled[i])).toBe(true);
              }
            }
          }
        }

        const flushed = resampler.flush();
        if (flushed.length > 0) {
          telemetry.recordAcceptedSamples(flushed.length);
          telemetry.recordEmittedSamples(flushed.length);
          totalResampledSamples += flushed.length;
        }

        const snapshot = telemetry.getSnapshot();
        expect(snapshot.inputSampleCount).toBe(totalInputSamples);
        expect(snapshot.acceptedSampleCount).toBe(totalResampledSamples);
        expect(snapshot.emittedSampleCount).toBe(totalResampledSamples);
        expect(snapshot.droppedSamples).toBe(0);
        expect(snapshot.duplicatedSamples).toBe(0);

        const verification = telemetry.verifyConservation();
        expect(verification.ok).toBe(true);
        expect(verification.unaccountedDroppedSamples).toBe(0);

        // Check expected ratio
        const expectedRatio = 16000 / rate;
        const actualRatio = totalResampledSamples / totalInputSamples;
        expect(Math.abs(actualRatio - expectedRatio)).toBeLessThan(0.002);
      },
      30_000,
    );
  });

  describe('Deterministic Fault Injection Scenarios', () => {
    it('Scenario A: Network failure during TEN-VAD asset fetch terminates worker and degrades cleanly', async () => {
      let createdWorker: MockReliabilityWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://cdn.invalid/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new MockReliabilityWorker('fail-init');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await expect(adapter.init()).rejects.toThrow('404 Not Found');
      expect(createdWorker).not.toBeNull();
      expect(createdWorker!.terminated).toBe(true);
      expect(adapter.getStatus().state).toBe('degraded');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
    });

    it('Scenario B: Worker runtime crash (onerror) triggers immediate termination and status transition', async () => {
      let createdWorker: MockReliabilityWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://cdn.invalid/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new MockReliabilityWorker('crash-on-process');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await adapter.init();
      expect(adapter.getStatus().state).toBe('ready');
      expect(createdWorker).not.toBeNull();
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(1);

      // Processing triggers worker crash
      adapter.process(new Float32Array(512), 0);
      await new Promise((resolve) => setTimeout(resolve, 30));

      expect(createdWorker!.terminated).toBe(true);
      expect(adapter.getStatus().state).toBe('degraded');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);

      await adapter.dispose();
      expect(adapter.getStatus().state).toBe('idle');
    });

    it('Scenario C: Abort during in-flight initialization cleans up and terminates worker', async () => {
      let createdWorker: MockReliabilityWorker | null = null;
      const controller = new AbortController();
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://cdn.invalid/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new MockReliabilityWorker('hold-init');
            return createdWorker as unknown as Worker;
          },
        },
      );

      const initPromise = adapter.init(controller.signal);
      await new Promise((resolve) => setTimeout(resolve, 10));
      controller.abort();

      await expect(initPromise).rejects.toMatchObject({
        name: 'AssetLoadAbortedError',
      });

      expect(createdWorker).not.toBeNull();
      expect(createdWorker!.terminated).toBe(true);
      expect(adapter.getStatus().state).toBe('idle');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
    });

    it('Scenario D: Dispose during active streaming terminates all workers and completes in < 500 ms', async () => {
      let createdWorker: MockReliabilityWorker | null = null;
      const starter = createBrowserRealtimeStarter({
        profileId: 'realtime-rnnt',
        config: {
          gateMode: 'hybrid',
          tenVadEnabled: true,
          vadBackend: 'ten-vad',
        },
        tenVadConfig: {
          assetBaseUrl: 'https://cdn.invalid/ten-vad/',
        },
        tenVadOptions: {
          workerFactory: () => {
            createdWorker = new MockReliabilityWorker('normal');
            return createdWorker as unknown as Worker;
          },
        },
      });

      await starter.start();
      expect(createdWorker).not.toBeNull();
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(1);

      // Push active speech frames
      for (let i = 0; i < 5; i += 1) {
        starter.processChunk(new Float32Array(512).fill(0.2));
      }

      const startDispose = performance.now();
      await starter.dispose();
      const disposeDuration = performance.now() - startDispose;

      expect(disposeDuration).toBeLessThan(500);
      expect(createdWorker!.terminated).toBe(true);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      expect(getGlobalStreamingTelemetry().getSnapshot().disposalDurationMs).not.toBeNull();
      expect(getGlobalStreamingTelemetry().getSnapshot().disposalDurationMs!).toBeLessThan(500);
    });
  });

  describe('Detector Integration & Lifecycle Telemetry', () => {
    it('StreamingSpeechDetector records vadStart and vadEnd events in telemetry', async () => {
      const detector = new StreamingSpeechDetector({
        profileId: 'parakeet-tdt-v3-realtime-hybrid',
        config: {
          gateMode: 'rough-only',
          tenVadEnabled: false,
        },
      });

      await detector.start();
      const snapshotInitial = getGlobalStreamingTelemetry().getSnapshot();
      expect(snapshotInitial.vadStartCount).toBe(0);

      // Simulate loud speech chunk to trigger rough speech start
      const speechChunk = new Float32Array(512);
      for (let i = 0; i < speechChunk.length; i += 1) {
        speechChunk[i] = 0.5 * Math.sin(i * 0.1);
      }

      for (let i = 0; i < 5; i += 1) {
        detector.processChunk(speechChunk);
      }

      // Finalize segment
      detector.finalizeSegment('silence');

      const snapshotAfter = getGlobalStreamingTelemetry().getSnapshot();
      expect(snapshotAfter.vadStartCount).toBeGreaterThan(0);
      expect(snapshotAfter.vadEndCount).toBeGreaterThan(0);
      expect(snapshotAfter.acceptedSampleCount).toBe(512 * 5);
      expect(snapshotAfter.emittedSampleCount).toBeGreaterThan(0);

      await detector.dispose();
    });
  });
});
