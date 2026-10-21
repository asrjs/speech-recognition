import { describe, expect, it, vi } from 'vitest';

const mockLoadSpeechModel = vi.fn();
vi.mock('../src/runtime/load.js', () => ({
  loadSpeechModel: (...args: unknown[]) => mockLoadSpeechModel(...args),
}));

import * as workerEntry from '../src/worker.js';
import {
  createBrowserTranscriptionWorkerClient,
} from '../src/runtime/browser-transcription-worker.js';
import {
  NemotronStreamingTranscriber,
} from '../src/models/nemotron-rnnt/streaming-transcriber.js';
import {
  createBrowserRealtimeMicrophoneController,
  type BrowserRealtimeStarter,
  type BrowserRealtimeMonitor,
} from '../src/runtime/browser-controller.js';
import type {
  PartialTranscript,
  StreamingTranscriber,
  TranscriptResult,
} from '../src/types/index.js';
import type { NemotronRnntSpeechSession } from '../src/models/nemotron-rnnt/model.js';

describe('worker-and-stateful-streaming suite', () => {
  describe('1. worker.ts exports and streaming worker RPC lifecycle', () => {
    it('re-exports worker client and thread entrypoints from src/worker.ts', () => {
      expect(workerEntry.createBrowserTranscriptionWorkerClient).toBeTypeOf('function');
      expect(typeof workerEntry).toBe('object');
    });

    it('manages full streaming worker RPC lifecycle (startStream, pushStreamAudio, flushStream, finalizeStream, resetStream)', async () => {
      type MockMessage = { id: number; type: string; payload: unknown };
      let workerOnMessage: ((event: MessageEvent<unknown>) => void) | null = null;
      const postedMessages: Array<{ message: unknown; transfer?: Transferable[] }> = [];

      const mockWorker = {
        get onmessage() {
          return workerOnMessage;
        },
        set onmessage(fn: ((event: MessageEvent<unknown>) => void) | null) {
          workerOnMessage = fn;
        },
        onerror: null,
        postMessage: vi.fn((message: unknown, transfer?: Transferable[]) => {
          postedMessages.push({ message, transfer });
          const raw = message as MockMessage;

          // Respond asynchronously as a real worker would
          queueMicrotask(() => {
            if (!workerOnMessage) return;
            switch (raw.type) {
              case 'LOAD_BUILT_IN_MODEL':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: { modelId: 'nemotron-model', source: 'built-in' },
                    meta: { state: 'ready' },
                  },
                } as MessageEvent<unknown>);
                break;
              case 'START_STREAM':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: { streamId: 'stream-123', supportsStreaming: true },
                    meta: { state: 'ready' },
                  },
                } as MessageEvent<unknown>);
                break;
              case 'PUSH_STREAM_AUDIO':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: {
                      kind: 'partial',
                      revision: 1,
                      text: 'hello',
                      committedText: '',
                      previewText: 'hello',
                      warnings: [],
                      meta: { isFinal: false, sampleRate: 16000 },
                    } satisfies PartialTranscript,
                    meta: { state: 'ready' },
                  },
                } as MessageEvent<unknown>);
                break;
              case 'FLUSH_STREAM':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: {
                      kind: 'partial',
                      revision: 2,
                      text: 'hello world',
                      committedText: 'hello',
                      previewText: 'world',
                      warnings: [],
                      meta: { isFinal: false, sampleRate: 16000 },
                    } satisfies PartialTranscript,
                    meta: { state: 'ready' },
                  },
                } as MessageEvent<unknown>);
                break;
              case 'FINALIZE_STREAM':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: {
                      text: 'hello world finalized',
                      warnings: [],
                      meta: { isFinal: true, sampleRate: 16000 },
                    } satisfies TranscriptResult,
                    meta: { state: 'ready' },
                  },
                } as MessageEvent<unknown>);
                break;
              case 'RESET_STREAM':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: null,
                    meta: { state: 'ready' },
                  },
                } as MessageEvent<unknown>);
                break;
              case 'DISPOSE_MODEL':
                workerOnMessage({
                  data: {
                    id: raw.id,
                    type: 'SUCCESS',
                    payload: null,
                    meta: { state: 'idle' },
                  },
                } as MessageEvent<unknown>);
                break;
            }
          });
        }),
        terminate: vi.fn(),
      };

      const client = createBrowserTranscriptionWorkerClient({
        workerFactory: () => mockWorker as never,
      });

      // 1. Load model
      const modelInfo = await client.loadBuiltInModel({ modelId: 'nemotron-model' });
      expect(modelInfo.modelId).toBe('nemotron-model');

      // 2. Start stream
      const streamInfo = await client.startStream({ emitPartials: true });
      expect(streamInfo.streamId).toBe('stream-123');
      expect(streamInfo.supportsStreaming).toBe(true);

      // 3. Push stream audio and check transfer of the client-owned copy.
      const inputPcm = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const partial = await client.pushStreamAudio(inputPcm, 16000);
      expect(partial).not.toBeNull();
      expect(partial?.text).toBe('hello');
      expect(partial?.kind).toBe('partial');

      // Verify the transfer list contains the owned message buffer.
      const pushCall = postedMessages.find((call) => (call.message as MockMessage).type === 'PUSH_STREAM_AUDIO');
      expect(pushCall).toBeDefined();
      expect(pushCall?.transfer).toHaveLength(1);
      expect(pushCall?.transfer?.[0]).toBeInstanceOf(ArrayBuffer);

      // 4. Flush stream
      const flushed = await client.flushStream();
      expect(flushed).not.toBeNull();
      expect(flushed?.text).toBe('hello world');

      // 5. Finalize stream
      const finalResult = await client.finalizeStream();
      expect(finalResult.text).toBe('hello world finalized');
      expect(finalResult.meta.isFinal).toBe(true);

      // 6. Reset stream
      await client.resetStream();
      const resetCall = postedMessages.find((call) => (call.message as MockMessage).type === 'RESET_STREAM');
      expect(resetCall).toBeDefined();

      // 7. Dispose model & client
      await client.disposeModel();
      await client.dispose();
      expect(mockWorker.terminate).toHaveBeenCalled();
    });

    it('worker thread dispatches streaming operations to loaded model createStreamingTranscriber', async () => {
      const mockTranscriber: StreamingTranscriber = {
        pushAudio: vi.fn(async (_input) => ({
          kind: 'partial',
          revision: 1,
          text: 'worker stream text',
          committedText: '',
          previewText: 'worker stream text',
          warnings: [],
          meta: { isFinal: false },
        })),
        flush: vi.fn(async () => ({
          kind: 'partial',
          revision: 2,
          text: 'worker stream text flushed',
          committedText: 'worker stream text flushed',
          previewText: '',
          warnings: [],
          meta: { isFinal: false },
        })),
        finalize: vi.fn(async () => ({
          kind: 'final',
          revision: 3,
          text: 'worker stream final',
          committedText: 'worker stream final',
          previewText: '',
          warnings: [],
          meta: { isFinal: true },
        })),
        reset: vi.fn(async () => undefined),
        dispose: vi.fn(async () => undefined),
        getState: vi.fn(() => ({
          revision: 1,
          bufferedDurationSeconds: 1,
          committedText: '',
          previewText: '',
          isFinalized: false,
        })),
      };

      const mockModel = {
        model: { id: 'test-stream-model', info: { family: 'nemotron-rnnt' } },
        transcribeMonoPcm: vi.fn(async () => ({ text: 'offline' })),
        createStreamingTranscriber: vi.fn(async () => mockTranscriber),
        dispose: vi.fn(async () => undefined),
      };

      mockLoadSpeechModel.mockResolvedValue(mockModel);

      // 1. Load model
      const loadRes = await workerEntry.handleRequest({
        id: 1,
        type: 'LOAD_BUILT_IN_MODEL',
        payload: { modelId: 'test-stream-model' },
      });
      expect(loadRes).toEqual(expect.objectContaining({ modelId: 'test-stream-model' }));

      // 2. Start stream
      const startRes = (await workerEntry.handleRequest({
        id: 2,
        type: 'START_STREAM',
        payload: { emitPartials: true },
      })) as { streamId: string; supportsStreaming: boolean };
      expect(startRes.supportsStreaming).toBe(true);
      expect(startRes.streamId).toMatch(/^stream-/);
      expect(mockModel.createStreamingTranscriber).toHaveBeenCalled();

      // 3. Push stream audio
      const pushRes = (await workerEntry.handleRequest({
        id: 3,
        type: 'PUSH_STREAM_AUDIO',
        payload: { pcm: new Float32Array([0.1, 0.2]), sampleRate: 16000 },
      })) as PartialTranscript;
      expect(pushRes.text).toBe('worker stream text');
      expect(mockTranscriber.pushAudio).toHaveBeenCalled();

      // 4. Flush stream
      const flushRes = (await workerEntry.handleRequest({
        id: 4,
        type: 'FLUSH_STREAM',
        payload: null,
      })) as PartialTranscript;
      expect(flushRes.text).toBe('worker stream text flushed');
      expect(mockTranscriber.flush).toHaveBeenCalled();

      // 5. Finalize stream
      const finalRes = (await workerEntry.handleRequest({
        id: 5,
        type: 'FINALIZE_STREAM',
        payload: null,
      })) as TranscriptResult;
      expect(finalRes.text).toBe('worker stream final');
      expect(mockTranscriber.finalize).toHaveBeenCalled();

      // 6. Reset stream
      await workerEntry.handleRequest({
        id: 6,
        type: 'RESET_STREAM',
        payload: null,
      });
      expect(mockTranscriber.reset).toHaveBeenCalled();

      // 7. Dispose model cleans up activeStream
      await workerEntry.handleRequest({
        id: 7,
        type: 'DISPOSE_MODEL',
        payload: null,
      });
      expect(mockTranscriber.dispose).toHaveBeenCalled();
      expect(mockModel.dispose).toHaveBeenCalled();

      // 8. Test model without createStreamingTranscriber returns supportsStreaming: false
      const mockNonStreamingModel = {
        model: { id: 'non-streaming-model' },
        transcribeMonoPcm: vi.fn(),
        dispose: vi.fn(async () => undefined),
      };
      mockLoadSpeechModel.mockResolvedValue(mockNonStreamingModel);

      await workerEntry.handleRequest({
        id: 8,
        type: 'LOAD_BUILT_IN_MODEL',
        payload: { modelId: 'non-streaming-model' },
      });
      const nonStreamRes = (await workerEntry.handleRequest({
        id: 9,
        type: 'START_STREAM',
        payload: undefined,
      })) as { streamId: string; supportsStreaming: boolean };
      expect(nonStreamRes.supportsStreaming).toBe(false);
      expect(nonStreamRes.streamId).toBe('');

      await workerEntry.handleRequest({
        id: 10,
        type: 'DISPOSE_MODEL',
        payload: null,
      });
    });
  });

  describe('2. NemotronStreamingTranscriber (pushAudio, onPartial, finalize, reset, cancel, dispose)', () => {
    function createMockSession() {
      let callCount = 0;
      const transcribe = vi.fn(async (input: any, _options: any) => {
        callCount += 1;
        const pcm = input instanceof Float32Array ? input : input.channels?.[0] ?? new Float32Array(0);
        const text = `chunk-${callCount}`;
        return {
          text,
          warnings: [],
          meta: {
            isFinal: false,
            sampleRate: 16000,
            durationSeconds: pcm.length / 16000,
          },
          segments: [
            {
              index: 0,
              text,
              startTime: 0,
              endTime: pcm.length / 16000,
            },
          ],
          words: [
            {
              index: 0,
              text,
              startTime: 0,
              endTime: pcm.length / 16000,
            },
          ],
          tokens: [
            {
              index: 0,
              text,
              startTime: 0,
              endTime: pcm.length / 16000,
            },
          ],
        } satisfies TranscriptResult;
      });

      const dispose = vi.fn(async () => undefined);
      const initialize = vi.fn(async () => undefined);

      return {
        transcribe,
        dispose,
        initialize,
      } as unknown as NemotronRnntSpeechSession;
    }

    it('buffers audio, emits partials, and commits chunks on step boundary with monotonic timestamps', async () => {
      const session = createMockSession();
      const partialsEmitted: PartialTranscript[] = [];

      const transcriber = new NemotronStreamingTranscriber(session, {
        stepSamples: 1600, // 100ms step for test speed
        emitPartials: true,
        sampleRate: 16000,
        onPartial: (p) => partialsEmitted.push(p),
      });

      // 1. Push small chunk (800 samples = 50ms < stepSamples)
      const chunk1 = new Float32Array(800);
      const p1 = await transcriber.pushAudio(chunk1);
      expect(p1.kind).toBe('partial');
      expect(p1.revision).toBe(1);
      expect(p1.meta.isFinal).toBe(false);

      // 2. Push second chunk (800 samples => total 1600 samples = step reached!)
      const chunk2 = new Float32Array(800);
      const p2 = await transcriber.pushAudio(chunk2);
      expect(p2.kind).toBe('partial');
      expect(p2.revision).toBe(2);
      expect(p2.committedText).toBe('chunk-2');
      expect(p2.segments).toBeDefined();
      expect(p2.segments?.[0]?.startTime).toBe(0);

      // 3. Push another 1600 samples (reaches second step)
      const chunk3 = new Float32Array(1600);
      const p3 = await transcriber.pushAudio(chunk3);
      expect(p3.committedText).toBe('chunk-2 chunk-3');
      expect(p3.segments).toHaveLength(2);
      // Monotonic timestamp check: segment 2 starts at or after 0.1s
      expect(p3.segments?.[1]?.startTime).toBeCloseTo(0.1, 2);

      // Check onPartial was called
      expect(partialsEmitted.length).toBeGreaterThanOrEqual(2);

      // 4. Test finalize()
      const finalTranscript = await transcriber.finalize();
      expect(finalTranscript.kind).toBe('final');
      expect(finalTranscript.meta.isFinal).toBe(true);
      expect(finalTranscript.committedText).toBe(finalTranscript.text);
      expect(transcriber.getState().isFinalized).toBe(true);

      // 5. Test reset()
      await transcriber.reset();
      const stateAfterReset = transcriber.getState();
      expect(stateAfterReset.committedText).toBe('');
      expect(stateAfterReset.previewText).toBe('');
      expect(stateAfterReset.isFinalized).toBe(false);
      expect(stateAfterReset.bufferedDurationSeconds).toBe(0);

      // 6. Test cancel()
      await transcriber.pushAudio(new Float32Array(500));
      transcriber.cancel();
      expect(transcriber.getState().previewText).toBe('');

      // 7. Test dispose()
      const onDispose = vi.fn(async () => undefined);
      const transcriber2 = new NemotronStreamingTranscriber(session, {}, onDispose);
      await transcriber2.dispose();
      expect(onDispose).toHaveBeenCalled();
      await expect(transcriber2.pushAudio(new Float32Array(100))).rejects.toThrow('disposed');
    });

    it('routes production-style sessions through cumulative stateful transcribe and finalizes the tail', async () => {
      const transcribe = vi.fn(async () => { throw new Error('offline chunk transcription must not run'); });
      const streamCalls: Array<{ samples: number; finalize: boolean }> = [];
      const transcribeStreaming = vi.fn(async (input: Float32Array, _options: unknown, finalize = false) => {
        streamCalls.push({ samples: input.length, finalize });
        return {
          text: `samples-${input.length}`,
          warnings: [],
          meta: { isFinal: finalize, sampleRate: 16000, durationSeconds: input.length / 16000 },
        } satisfies TranscriptResult;
      });
      const resetStreaming = vi.fn();
      const session = { transcribe, transcribeStreaming, resetStreaming } as unknown as NemotronRnntSpeechSession;
      const transcriber = new NemotronStreamingTranscriber(session, {
        stepSamples: 1600,
        emitPartials: false,
        sampleRate: 16000,
      });

      await transcriber.pushAudio(new Float32Array(800));
      await transcriber.pushAudio(new Float32Array(800));
      expect(transcriber.getState().committedText).toBe('samples-1600');
      await transcriber.pushAudio(new Float32Array(1600));
      expect(transcriber.getState().committedText).toBe('samples-3200');
      const final = await transcriber.finalize();
      expect(final.text).toBe('samples-3200');
      expect(streamCalls).toEqual([
        { samples: 1600, finalize: false },
        { samples: 3200, finalize: false },
        { samples: 3200, finalize: true },
      ]);
      expect(transcribe).not.toHaveBeenCalled();
      await transcriber.reset();
      expect(resetStreaming).toHaveBeenCalledOnce();
      await transcriber.dispose();
    });

    it('warns when a session falls back to independent offline chunk transcription', async () => {
      const session = createMockSession();
      const transcriber = new NemotronStreamingTranscriber(session, {
        stepSamples: 1000,
        emitPartials: false,
      });
      await transcriber.pushAudio(new Float32Array(1000));
      const partial = await transcriber.pushAudio(new Float32Array(1000));
      expect(partial.warnings).toEqual(expect.arrayContaining([
        expect.objectContaining({
          code: 'nemotron-rnnt.streaming-fallback-offline-chunks',
          recoverable: true,
        }),
      ]));
      await transcriber.dispose();
    });

    it('supports onPartial method subscription and unsubscription', async () => {
      const session = createMockSession();
      const transcriber = new NemotronStreamingTranscriber(session, {
        stepSamples: 1000,
        emitPartials: true,
      });

      const received: string[] = [];
      const unsubscribe = transcriber.onPartial((p) => {
        received.push(p.text);
      });

      await transcriber.pushAudio(new Float32Array(1000));
      expect(received).toHaveLength(1);

      unsubscribe();
      await transcriber.pushAudio(new Float32Array(1000));
      // No new events received after unsubscription
      expect(received).toHaveLength(1);
    });
  });

  describe('3. BrowserRealtimeMicrophoneController with streamingTranscriber and onPartialTranscript', () => {
    function createMockStarter() {
      const listeners = new Set<(event: any) => void>();
      return {
        listeners,
        starter: {
          detector: {} as never,
          tenVad: {} as never,
          vadBuffer: {} as never,
          controller: null,
          subscribe(listener: any) {
            listeners.add(listener);
            return () => listeners.delete(listener);
          },
          async start() {
            listeners.forEach((l) => l({ type: 'metrics', payload: { tenVad: { state: 'ready' } } }));
          },
          processChunk: vi.fn(),
          flush: vi.fn(),
          stop: vi.fn(async () => undefined),
          updateConfig: vi.fn(),
          getSnapshot() {
            return {
              state: 'listening',
              tenVad: { state: 'ready' },
              latency: null,
            } as never;
          },
          dispose: vi.fn(async () => undefined),
        } as BrowserRealtimeStarter,
      };
    }

    function createMockMonitor(): BrowserRealtimeMonitor {
      return {
        subscribe() {
          return () => undefined;
        },
        getSnapshot() {
          return null;
        },
        flush: vi.fn(),
        dispose: vi.fn(),
      };
    }

    it('feeds chunks to streamingTranscriber during speech and finalizes on silence (segment-ready)', async () => {
      let captureOnChunk!: (chunk: any) => void;
      const stopCapture = vi.fn(async () => undefined);

      const pushedChunks: Float32Array[] = [];
      const partialsEmitted: PartialTranscript[] = [];

      const mockTranscriber: StreamingTranscriber = {
        pushAudio: vi.fn(async (input: any) => {
          const pcm = input instanceof Float32Array ? input : input.channels?.[0];
          pushedChunks.push(pcm);
          const partial: PartialTranscript = {
            kind: 'partial',
            revision: pushedChunks.length,
            text: `partial-${pushedChunks.length}`,
            committedText: '',
            previewText: `partial-${pushedChunks.length}`,
            warnings: [],
            meta: { isFinal: false, sampleRate: 16000 },
          };
          return partial;
        }),
        flush: vi.fn(async () => ({
          kind: 'partial',
          revision: 10,
          text: 'flushed',
          committedText: 'flushed',
          previewText: '',
          warnings: [],
          meta: { isFinal: false },
        })),
        finalize: vi.fn(async () => ({
          kind: 'final',
          revision: 11,
          text: 'finalized-sentence',
          committedText: 'finalized-sentence',
          previewText: '',
          warnings: [],
          meta: { isFinal: true },
        })),
        reset: vi.fn(async () => undefined),
        dispose: vi.fn(async () => undefined),
        getState: vi.fn(() => ({
          revision: 1,
          bufferedDurationSeconds: 1,
          committedText: '',
          previewText: '',
          isFinalized: false,
        })),
      };

      const { starter, listeners } = createMockStarter();

      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        streamingTranscriber: mockTranscriber,
        onPartialTranscript: (p) => partialsEmitted.push(p),
        createStarter: () => starter,
        createMonitor: createMockMonitor,
        startCapture: async (opts) => {
          captureOnChunk = opts.onChunk;
          return {
            sampleRate: 16000,
            deviceSampleRate: 16000,
            contextSampleRate: 16000,
            chunkFrames: 256,
            chunkDurationMs: 16,
            stream: { getAudioTracks: () => [{ label: 'mic' }] } as any,
            stop: stopCapture,
          };
        },
      });

      await controller.start();

      expect(mockTranscriber.reset).toHaveBeenCalledOnce();
      vi.mocked(mockTranscriber.reset).mockClear();
      // 1. Silence chunk: speech is not active yet, so streamingTranscriber.pushAudio is NOT called
      const silenceChunk = new Float32Array([0.01, 0.02]);
      captureOnChunk({ pcm: silenceChunk, sampleRate: 16000, startFrame: 0, endFrame: 2 });
      await new Promise((r) => setTimeout(r, 20));
      expect(mockTranscriber.pushAudio).not.toHaveBeenCalled();

      // 2. Speech onset occurs! Detector emits speech-start
      listeners.forEach((l) => l({ type: 'speech-start', payload: { startFrame: 2, sampleRate: 16000 } }));

      // 3. Speech chunks arrive during speech
      const speechChunk1 = new Float32Array([0.5, 0.6, 0.7]);
      captureOnChunk({ pcm: speechChunk1, sampleRate: 16000, startFrame: 2, endFrame: 5 });

      const speechChunk2 = new Float32Array([0.8, 0.9, 1.0]);
      captureOnChunk({ pcm: speechChunk2, sampleRate: 16000, startFrame: 5, endFrame: 8 });

      await vi.waitFor(() => expect(partialsEmitted.length).toBe(2));
      expect(partialsEmitted[0]?.text).toBe('partial-1');
      expect(partialsEmitted[1]?.text).toBe('partial-2');
      expect(pushedChunks).toHaveLength(2);

      // 4. The detector emits speech-end before the accepted segment-ready event.
      listeners.forEach((l) => l({ type: 'speech-end', payload: {
        startFrame: 2, endFrame: 8, sampleRate: 16000,
      } }));
      listeners.forEach((l) =>
        l({
          type: 'segment-ready',
          payload: {
            readPcm: () => new Float32Array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            sampleRate: 16000,
            reason: 'silence',
            startFrame: 2,
            endFrame: 8,
            metadata: null,
          },
        }),
      );

      // 5. Finalize and reset must be called on the segment boundary
      await vi.waitFor(() => expect(mockTranscriber.finalize).toHaveBeenCalledOnce());
      expect(mockTranscriber.reset).toHaveBeenCalledOnce();

      // 6. Clean stop and dispose
      await controller.stop();
      await controller.dispose();
      expect(stopCapture).toHaveBeenCalled();
    });

    it('falls back to transcribeMonoPcm when streamingTranscriber is not provided', async () => {
      let captureOnChunk!: (chunk: any) => void;
      const transcribeMonoPcm = vi.fn(async () => ({ text: 'fallback' }));
      const { starter, listeners } = createMockStarter();

      const controller = createBrowserRealtimeMicrophoneController({
        micMode: 'speech-detect',
        transcribeMonoPcm,
        createStarter: () => starter,
        createMonitor: createMockMonitor,
        startCapture: async (opts) => {
          captureOnChunk = opts.onChunk;
          return {
            sampleRate: 16000,
            deviceSampleRate: 16000,
            contextSampleRate: 16000,
            chunkFrames: 256,
            chunkDurationMs: 16,
            stream: { getAudioTracks: () => [] } as any,
            stop: async () => undefined,
          };
        },
      });

      await controller.start();

      const pcmChunk = new Float32Array([0.1, 0.2]);
      captureOnChunk({ pcm: pcmChunk, sampleRate: 16000, startFrame: 0, endFrame: 2 });

      listeners.forEach((l) =>
        l({
          type: 'segment-ready',
          payload: {
            readPcm: () => pcmChunk,
            sampleRate: 16000,
            reason: 'silence',
            startFrame: 0,
            endFrame: 2,
            metadata: null,
          },
        }),
      );

      await vi.waitFor(() => expect(transcribeMonoPcm).toHaveBeenCalledWith(expect.any(Float32Array), 16000));
      await controller.dispose();
    });
  });
});
