import { describe, expect, it, vi } from 'vitest';
import {
  createBrowserRealtimeMicrophoneController,
  type BrowserRealtimeMonitor,
  type BrowserRealtimeStarter,
} from '@asrjs/speech-recognition/browser';

function createFakeStarter(): BrowserRealtimeStarter {
  const listeners = new Set<(event: any) => void>();
  return {
    detector: {} as never,
    tenVad: {} as never,
    vadBuffer: {} as never,
    controller: null,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async start() {
      listeners.forEach((listener) =>
        listener({
          type: 'metrics',
          payload: {
            tenVad: { state: 'ready' },
          },
        }),
      );
    },
    processChunk: vi.fn(),
    flush: vi.fn(),
    stop: vi.fn(async () => null),
    updateConfig: vi.fn(),
    getSnapshot() {
      return {
        tenVad: { state: 'ready' },
      } as never;
    },
    dispose: vi.fn(async () => undefined),
  };
}

function createFakeMonitor(): BrowserRealtimeMonitor {
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

describe('browser realtime microphone controller', () => {
  it('captures manual microphone audio and emits a flushed utterance', async () => {
    const utterances: Array<{ pcm: Float32Array; sampleRate: number; reason: string }> = [];
    let onChunk: ((chunk: any) => void) | null = null;
    const stop = vi.fn(async () => undefined);

    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'manual',
      createStarter: createFakeStarter,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        return {
          sampleRate: 16000,
          deviceSampleRate: 48000,
          contextSampleRate: 48000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {
            getAudioTracks() {
              return [{ label: 'Fake mic' }] as MediaStreamTrack[];
            },
          } as MediaStream,
          stop,
        };
      },
      onUtterance(utterance) {
        utterances.push({
          pcm: utterance.pcm,
          sampleRate: utterance.sampleRate,
          reason: utterance.reason,
        });
      },
    });

    await controller.start();
    expect(controller.getState().isMicActive).toBe(true);
    expect(controller.getState().captureInfo.deviceLabel).toBe('Fake mic');

    onChunk?.({
      pcm: new Float32Array([0.25, 0.5]),
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 2,
    });
    onChunk?.({
      pcm: new Float32Array([0.75]),
      sampleRate: 16000,
      startFrame: 2,
      endFrame: 3,
    });

    controller.flush('manual');

    expect(utterances).toHaveLength(1);
    expect(Array.from(utterances[0]!.pcm)).toEqual([0.25, 0.5, 0.75]);
    expect(utterances[0]!.sampleRate).toBe(16000);
    expect(utterances[0]!.reason).toBe('manual');

    await controller.stop();
    expect(stop).toHaveBeenCalled();
    expect(controller.getState().isMicActive).toBe(false);
  });

  it('routes speech-detect chunks through the realtime starter', async () => {
    const starter = createFakeStarter();
    let onChunk: ((chunk: any) => void) | null = null;

    const controller = createBrowserRealtimeMicrophoneController({
      micMode: 'speech-detect',
      createStarter: () => starter,
      createMonitor: createFakeMonitor,
      startCapture: async (options) => {
        onChunk = options.onChunk;
        return {
          sampleRate: 16000,
          deviceSampleRate: 48000,
          contextSampleRate: 48000,
          chunkFrames: 256,
          chunkDurationMs: 16,
          stream: {
            getAudioTracks() {
              return [] as MediaStreamTrack[];
            },
          } as MediaStream,
          stop: async () => undefined,
        };
      },
    });

    await controller.start();
    onChunk?.({
      pcm: new Float32Array([1, 2, 3]),
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 3,
    });

    expect(starter.processChunk).toHaveBeenCalledWith(
      expect.any(Float32Array),
      expect.objectContaining({ startFrame: 0, endFrame: 3 }),
    );

    await controller.dispose();
  });
});
