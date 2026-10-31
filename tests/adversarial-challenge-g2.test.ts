import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createStreamingTelemetryCollector,
  getGlobalStreamingTelemetry,
  resetGlobalStreamingTelemetry,
} from '../src/runtime/streaming-telemetry.js';
import { TenVadAdapter } from '../src/runtime/ten-vad-browser.js';
import { StatefulLinearResampler } from '../src/audio/resample.js';
import type { TenVadRequestMessage } from '../src/runtime/ten-vad-protocol.js';

class AdversarialMockWorker {
  onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminateCalls = 0;
  postMessageCalls: TenVadRequestMessage[] = [];

  constructor(
    public readonly behavior:
      | 'success'
      | 'throw-sync-init'
      | 'error-msg-init'
      | 'crash-onerror-process'
      | 'error-msg-process'
      | 'throw-sync-process'
      | 'hang-init',
  ) {}

  postMessage(message: TenVadRequestMessage): void {
    this.postMessageCalls.push(message);

    if (message.type === 'INIT') {
      if (this.behavior === 'throw-sync-init') {
        throw new Error('Adversarial sync error inside worker.postMessage(INIT)');
      }
      if (this.behavior === 'error-msg-init') {
        queueMicrotask(() => {
          this.onmessage?.({
            data: {
              type: 'ERROR',
              id: message.id,
              payload: 'WASM compilation failed: out of memory',
            },
          } as MessageEvent<unknown>);
        });
        return;
      }
      if (this.behavior === 'hang-init') {
        return;
      }
      queueMicrotask(() => {
        this.onmessage?.({
          data: {
            type: 'INIT',
            id: message.id,
            payload: { success: true, version: 'adversarial-mock-1.0' },
          },
        } as MessageEvent<unknown>);
      });
      return;
    }

    if (message.type === 'PROCESS') {
      if (this.behavior === 'throw-sync-process') {
        throw new Error('Adversarial sync crash in postMessage(PROCESS)');
      }
      if (this.behavior === 'crash-onerror-process') {
        queueMicrotask(() => {
          this.onerror?.({
            message: 'Uncaught WebWorker RuntimeError: unreachable executed',
          } as ErrorEvent);
        });
        return;
      }
      if (this.behavior === 'error-msg-process') {
        // As implemented in ten-vad-worker.ts line 57: id: message.id ?? 0
        queueMicrotask(() => {
          this.onmessage?.({
            data: {
              type: 'ERROR',
              id: message.id ?? 0,
              payload: 'Worker process internal buffer corruption',
            },
          } as MessageEvent<unknown>);
        });
        return;
      }
      const samples = message.payload.samples;
      const hopCount = Math.max(1, Math.floor(samples.length / 256));
      queueMicrotask(() => {
        this.onmessage?.({
          data: {
            type: 'RESULT',
            payload: {
              probabilities: new Float32Array(hopCount).fill(0.75),
              flags: new Uint8Array(hopCount).fill(1),
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
    this.terminateCalls += 1;
    this.onmessage = null;
    this.onerror = null;
  }
}

describe('Empirical Adversarial Challenges — Milestone G2', () => {
  beforeEach(() => {
    resetGlobalStreamingTelemetry();
  });

  afterEach(() => {
    resetGlobalStreamingTelemetry();
    vi.restoreAllMocks();
  });

  describe('Adversarial Audio Mass Conservation Challenge (100,000+ random samples)', () => {
    it('feeds 250,000 samples of random noise through irregular jitter chunks and asserts strict conservation', () => {
      const collector = createStreamingTelemetryCollector();
      const TOTAL_SAMPLES = 250_000;
      const noise = new Float32Array(TOTAL_SAMPLES);
      for (let i = 0; i < TOTAL_SAMPLES; i += 1) {
        noise[i] = (Math.random() * 2 - 1) * 0.95;
      }

      // Irregular, non-power-of-2 jitter chunk sizes including empty (0), tiny (1), odd primes, and huge bursts
      const jitterPattern = [
        0, 1, 3, 7, 13, 0, 42, 97, 128, 255, 513, 1000, 1777, 4096, 8191, 16384,
      ];
      let offset = 0;
      let jitterIndex = 0;
      let totalPushed = 0;

      while (offset < TOTAL_SAMPLES) {
        const chunkSize = jitterPattern[jitterIndex % jitterPattern.length]!;
        jitterIndex += 1;

        if (chunkSize === 0) {
          // Zero-length chunk stress test
          collector.recordInputSamples(0);
          collector.recordAcceptedSamples(0);
          collector.recordEmittedSamples(0);
          continue;
        }

        const end = Math.min(offset + chunkSize, TOTAL_SAMPLES);
        const chunk = noise.subarray(offset, end);
        offset = end;
        totalPushed += chunk.length;

        collector.recordInputSamples(chunk.length);
        collector.recordAcceptedSamples(chunk.length);
        collector.recordEmittedSamples(chunk.length);

        // Continuous conservation assertion during streaming
        const snapshot = collector.getSnapshot();
        expect(snapshot.droppedSamples).toBe(0);
        expect(snapshot.duplicatedSamples).toBe(0);
      }

      expect(totalPushed).toBe(TOTAL_SAMPLES);

      const finalSnapshot = collector.getSnapshot();
      expect(finalSnapshot.inputSampleCount).toBe(TOTAL_SAMPLES);
      expect(finalSnapshot.acceptedSampleCount).toBe(TOTAL_SAMPLES);
      expect(finalSnapshot.emittedSampleCount).toBe(TOTAL_SAMPLES);
      expect(finalSnapshot.droppedSamples).toBe(0);
      expect(finalSnapshot.duplicatedSamples).toBe(0);

      const verification = collector.verifyConservation();
      expect(verification.ok).toBe(true);
      expect(verification.unaccountedDroppedSamples).toBe(0);
      expect(verification.detail).toContain(
        'Conservation satisfied: 0 dropped, 0 duplicated, 0 unaccounted samples.',
      );
    });

    it('adversarially stress-tests resampling pipeline with 150,000 random samples and irregular jitter', () => {
      const collector = createStreamingTelemetryCollector();
      const resampler = new StatefulLinearResampler(48000, 16000);
      const TOTAL_INPUT = 150_000;
      const noise = new Float32Array(TOTAL_INPUT);
      for (let i = 0; i < TOTAL_INPUT; i += 1) {
        noise[i] = (Math.random() * 2 - 1) * 0.8;
      }

      const jitterSizes = [1, 2, 5, 17, 33, 64, 127, 256, 500, 1023, 3333, 7000];
      let offset = 0;
      let jitterIdx = 0;
      let totalAccepted = 0;

      while (offset < TOTAL_INPUT) {
        const size = jitterSizes[jitterIdx % jitterSizes.length]!;
        jitterIdx += 1;
        const end = Math.min(offset + size, TOTAL_INPUT);
        const chunk = noise.subarray(offset, end);
        offset = end;

        collector.recordInputSamples(chunk.length);
        const out = resampler.process(chunk);
        collector.recordAcceptedSamples(out.length);
        collector.recordEmittedSamples(out.length);
        totalAccepted += out.length;
      }

      const flushed = resampler.flush();
      if (flushed.length > 0) {
        collector.recordAcceptedSamples(flushed.length);
        collector.recordEmittedSamples(flushed.length);
        totalAccepted += flushed.length;
      }

      const snapshot = collector.getSnapshot();
      expect(snapshot.inputSampleCount).toBe(TOTAL_INPUT);
      expect(snapshot.acceptedSampleCount).toBe(totalAccepted);
      expect(snapshot.emittedSampleCount).toBe(totalAccepted);
      expect(snapshot.droppedSamples).toBe(0);
      expect(snapshot.duplicatedSamples).toBe(0);

      const verification = collector.verifyConservation();
      expect(verification.ok).toBe(true);
      expect(verification.unaccountedDroppedDrops ?? verification.unaccountedDroppedSamples).toBe(
        0,
      );
    });

    it('oracle check: verifyConservation correctly detects dropped and duplicated sample violations', () => {
      const collector = createStreamingTelemetryCollector();
      collector.recordAcceptedSamples(1000);
      collector.recordEmittedSamples(950);

      // 50 samples missing without explanation
      const vMissing = collector.verifyConservation(0);
      expect(vMissing.ok).toBe(false);
      expect(vMissing.unaccountedDroppedSamples).toBe(50);
      expect(vMissing.detail).toContain('Conservation violation');

      // 50 samples accounted for as buffered in-flight
      const vBuffered = collector.verifyConservation(50);
      expect(vBuffered.ok).toBe(true);
      expect(vBuffered.unaccountedDroppedSamples).toBe(0);

      // Explicitly dropped samples violate conservation even if arithmetic balances
      collector.recordDroppedSamples(50);
      const vDropped = collector.verifyConservation(0);
      expect(vDropped.ok).toBe(false);

      // Duplicated samples violate conservation
      collector.reset();
      collector.recordAcceptedSamples(100);
      collector.recordEmittedSamples(100);
      collector.recordDuplicatedSamples(10);
      const vDuplicated = collector.verifyConservation();
      expect(vDuplicated.ok).toBe(false);
    });

    it('handles negative or invalid inputs without corrupting counters', () => {
      const collector = createStreamingTelemetryCollector();
      collector.recordInputSamples(-100);
      collector.recordAcceptedSamples(-50);
      collector.recordEmittedSamples(-20);
      collector.recordDroppedSamples(-10);
      collector.recordDuplicatedSamples(-5);

      const snapshot = collector.getSnapshot();
      expect(snapshot.inputSampleCount).toBe(0);
      expect(snapshot.acceptedSampleCount).toBe(0);
      expect(snapshot.emittedSampleCount).toBe(0);
      expect(snapshot.droppedSamples).toBe(0);
      expect(snapshot.duplicatedSamples).toBe(0);
      expect(collector.verifyConservation().ok).toBe(true);
    });
  });

  describe('Adversarial Worker Lifecycle & Crash Qualification in TenVadAdapter', () => {
    it('Scenario 1: Worker throws synchronously during postMessage(INIT) -> worker.terminate() called & activeWorkerCount === 0', async () => {
      let createdWorker: AdversarialMockWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://test.local/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new AdversarialMockWorker('throw-sync-init');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await expect(adapter.init()).rejects.toThrow('Adversarial sync error');
      expect(createdWorker).not.toBeNull();
      expect(createdWorker!.terminateCalls).toBeGreaterThanOrEqual(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      expect(adapter.getStatus().state).toBe('degraded');
    });

    it('Scenario 2: Worker responds with async ERROR payload during INIT -> worker.terminate() called & activeWorkerCount === 0', async () => {
      let createdWorker: AdversarialMockWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://test.local/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new AdversarialMockWorker('error-msg-init');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await expect(adapter.init()).rejects.toThrow('WASM compilation failed');
      expect(createdWorker).not.toBeNull();
      expect(createdWorker!.terminateCalls).toBeGreaterThanOrEqual(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      expect(adapter.getStatus().state).toBe('degraded');
    });

    it('Scenario 3: Worker crashes with onerror event during process -> worker.terminate() called & activeWorkerCount === 0', async () => {
      let createdWorker: AdversarialMockWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://test.local/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new AdversarialMockWorker('crash-onerror-process');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await adapter.init();
      expect(adapter.getStatus().state).toBe('ready');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(1);

      // Trigger process which invokes worker.onerror
      const processed = adapter.process(new Float32Array(512), 0);
      expect(processed).toBe(true);

      // Allow microtasks to fire onerror handler
      await new Promise((resolve) => setTimeout(resolve, 30));

      expect(createdWorker!.terminateCalls).toBeGreaterThanOrEqual(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      expect(adapter.getStatus().state).toBe('degraded');

      // Subsequent process attempts return false safely without throwing
      const subsequent = adapter.process(new Float32Array(512), 512);
      expect(subsequent).toBe(false);

      // Disposing degraded adapter remains safe and idempotent
      await adapter.dispose();
      expect(adapter.getStatus().state).toBe('idle');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
    });

    it('Scenario 4: Worker sends ERROR response during process -> worker.terminate() called & activeWorkerCount === 0', async () => {
      let createdWorker: AdversarialMockWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://test.local/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new AdversarialMockWorker('error-msg-process');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await adapter.init();
      expect(adapter.getStatus().state).toBe('ready');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(1);

      adapter.process(new Float32Array(512), 0);
      await new Promise((resolve) => setTimeout(resolve, 30));

      expect(createdWorker!.terminateCalls).toBeGreaterThanOrEqual(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      expect(adapter.getStatus().state).toBe('degraded');
      expect(adapter.getStatus().error).toContain('Worker process internal buffer corruption');
    });

    it('Scenario 5: Multiple init calls and double dispose do not leak workers or underflow count', async () => {
      let createdWorker: AdversarialMockWorker | null = null;
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://test.local/ten-vad/' },
        {
          workerFactory: () => {
            createdWorker = new AdversarialMockWorker('success');
            return createdWorker as unknown as Worker;
          },
        },
      );

      await adapter.init();
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(1);

      // Second init is a no-op
      await adapter.init();
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(1);

      await adapter.dispose();
      expect(createdWorker!.terminateCalls).toBe(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);

      // Double dispose is idempotent
      await adapter.dispose();
      expect(createdWorker!.terminateCalls).toBe(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
    });

    it('Scenario 6: Missing assetBaseUrl throws TenVadAssetsNotConfiguredError and cleans up instantiated worker', async () => {
      let createdWorker: AdversarialMockWorker | null = null;
      const adapter = new TenVadAdapter(
        {},
        {
          workerFactory: () => {
            createdWorker = new AdversarialMockWorker('success');
            return createdWorker as unknown as Worker;
          },
        },
      );
      await expect(adapter.init()).rejects.toThrow('TEN-VAD is not bundled');
      expect(createdWorker).not.toBeNull();
      expect(createdWorker!.terminateCalls).toBeGreaterThanOrEqual(1);
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      expect(adapter.getStatus().state).toBe('degraded');
    });

    it('Scenario 7: WorkerFactory throws synchronously -> activeWorkerCount stays 0', async () => {
      const adapter = new TenVadAdapter(
        { assetBaseUrl: 'https://test.local/ten-vad/' },
        {
          workerFactory: () => {
            throw new Error('Host out of worker threads');
          },
        },
      );

      await expect(adapter.init()).rejects.toThrow('Host out of worker threads');
      expect(getGlobalStreamingTelemetry().getSnapshot().activeWorkerCount).toBe(0);
      // Note: Because workerFactory() threw before try/catch in init(), state is left as 'initializing'
      // This is an empirical boundary finding for review.
    });
  });
});
