import { describe, expect, it } from 'vitest';
import { StreamingSpeechDetector } from '../src/runtime/streaming-detector.js';
import { TenVadAdapter } from '../src/runtime/ten-vad-browser.js';
import type { TenVadRequestMessage } from '../src/runtime/ten-vad-protocol.js';

// Only inference is simulated: retain the real adapter's smoothing, absolute
// frame history and query implementation across successive utterances.
class ProbabilityWorker {
  onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;

  postMessage(message: TenVadRequestMessage): void {
    const response =
      message.type === 'PROCESS'
        ? {
            type: 'RESULT',
            payload: {
              probabilities: new Float32Array([message.payload.samples[0]! > 0 ? 0.9 : 0]),
              flags: new Uint8Array([message.payload.samples[0]! > 0 ? 1 : 0]),
              globalSampleOffset: message.payload.globalSampleOffset,
              hopCount: 1,
            },
          }
        : { type: message.type, id: message.id, payload: { success: true, version: 'test' } };
    this.onmessage?.({ data: response } as MessageEvent<unknown>);
  }

  terminate(): void {}
}

describe('neural VAD onset history across finalized segments', () => {
  it.each([
    ['vad-only', 'accepted'],
    ['hybrid', 'accepted'],
    ['vad-only', 'rejected'],
    ['hybrid', 'rejected'],
    ['vad-only', 'manual'],
    ['hybrid', 'manual'],
    ['vad-only', 'max-duration'],
    ['hybrid', 'max-duration'],
  ] as const)(
    '%s does not replay retained history after a %s segment',
    async (gateMode, firstOutcome) => {
      const detector = new StreamingSpeechDetector({
        config: {
          gateMode,
          vadBackend: 'ten-vad',
          tenVadEnabled: true,
          chunkDurationMs: 16,
          analysisWindowMs: 16,
          vadHopDurationMs: 16,
          energyThreshold: 1, // Keep the rough gate out of this neural-onset test.
          useSnrGate: false,
          minSpeechDurationMs: 16,
          minSilenceDurationMs: 32,
          maxSegmentDurationMs: firstOutcome === 'max-duration' ? 1600 : 4800,
          tenVadMinSpeechDurationMs: 32,
          tenVadMinSilenceDurationMs: 32,
          tenVadConfirmationWindowMs: 96,
          tenVadHangoverMs: 32,
          tenVadSpeechPaddingMs: 0,
          prerollMs: 0,
          overlapDurationMs: 0,
          speechHangoverMs: 0,
          foregroundFilterEnabled: true,
          minEnergyPerSecond: firstOutcome === 'rejected' ? 1e9 : 0,
          minEnergyIntegral: 0,
          useAdaptiveEnergyThresholds: false,
        },
        tenVadFactory: (config) =>
          new TenVadAdapter(config, {
            workerFactory: () => new ProbabilityWorker(),
            now: () => 0, // Both utterances must remain in the retained history.
          }),
      });
      const onsets: number[] = [];
      detector.subscribe((event) => {
        if (event.type === 'speech-start') onsets.push(event.payload.startFrame);
      });
      let frame = 0;
      const feed = (count: number, amplitude: number): void => {
        for (let index = 0; index < count; index += 1) {
          detector.processChunk(new Float32Array(256).fill(amplitude), { startFrame: frame });
          frame += 256;
        }
      };
      try {
        await detector.start();
        feed(firstOutcome === 'max-duration' ? 200 : 20, 0.2);
        if (firstOutcome === 'manual') detector.flush('manual');
        feed(20, 0);
        const first = detector.getSnapshot().recentSegments;
        if (firstOutcome === 'manual') {
          // Manual flush may be followed by the smoother's short speech tail.
          // It must not reopen the audio that was already flushed.
          expect(first.length).toBeGreaterThanOrEqual(1);
          expect(first.length).toBeLessThanOrEqual(2);
          for (let index = 1; index < first.length; index += 1) {
            expect(first[index]!.metadata.logicalStartFrame).toBeGreaterThanOrEqual(
              first[index - 1]!.metadata.logicalEndFrame,
            );
          }
        } else if (firstOutcome === 'max-duration') {
          expect(first.some((segment) => segment.reason === 'max-duration')).toBe(true);
          expect(first.length).toBeGreaterThanOrEqual(2);
          expect(first.length).toBeLessThanOrEqual(3);
          for (let index = 1; index < first.length; index += 1) {
            expect(first[index]!.metadata.logicalStartFrame).toBeGreaterThanOrEqual(
              Number(first[index - 1]!.metadata.logicalEndFrame) - 256,
            );
          }
        } else {
          expect(first).toHaveLength(firstOutcome === 'rejected' ? 0 : 1);
          expect(onsets).toHaveLength(1);
        }
        const firstOnsetCount = onsets.length;
        expect(detector.getSnapshot().activeSegment).toBeNull();
        const firstLogicalEnd = Number(detector.getSnapshot().acceptance?.logicalEndFrame);
        detector.updateConfig({ minEnergyPerSecond: 0 });
        const secondOnset = frame;
        feed(20, 0.2);
        feed(20, 0);
        const segments = detector.getSnapshot().recentSegments;
        expect(segments).toHaveLength(first.length + 1);
        expect(onsets).toHaveLength(firstOnsetCount + 1);
        expect(segments.at(-1)!.metadata.logicalStartFrame).toBeGreaterThanOrEqual(secondOnset);
        expect(segments.at(-1)!.metadata.logicalStartFrame).toBeGreaterThanOrEqual(firstLogicalEnd);
        expect(detector.getSnapshot().activeSegment).toBeNull();
        if (firstOutcome === 'accepted') {
          await detector.stop({ flush: false });
          await detector.start();
          frame = 0;
          feed(20, 0.2);
          feed(20, 0);
          const restarted = detector.getSnapshot().recentSegments;
          expect(restarted).toHaveLength(segments.length + 1);
          expect(restarted.at(-1)!.metadata.logicalStartFrame).toBe(0);
          expect(detector.getSnapshot().activeSegment).toBeNull();
        }
      } finally {
        await detector.dispose();
      }
    },
  );
});
