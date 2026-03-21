import {
  startMicrophoneCapture,
  startMicrophoneRingCapture,
} from '@asrjs/speech-recognition/browser';
import { describe, expect, it, vi } from 'vitest';

describe('microphone capture helpers', () => {
  it('emits mono PCM chunks and cleans up audio resources on stop', async () => {
    const sourceNode = {
      connect: vi.fn(),
      disconnect: vi.fn(),
    };
    const processorNode = {
      onaudioprocess: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
    };
    const close = vi.fn();
    const createAudioContext = vi.fn(() => ({
      sampleRate: 16000,
      destination: {},
      createMediaStreamSource: vi.fn(() => sourceNode),
      createScriptProcessor: vi.fn(() => processorNode),
      close,
    }));
    const stopTrack = vi.fn();
    const stream = {
      getTracks: () => [{ stop: stopTrack }],
    } as unknown as MediaStream;
    const onChunk = vi.fn();

    const handle = await startMicrophoneCapture({
      stream,
      createAudioContext,
      chunkFrames: 4,
      stopTracksOnStop: true,
      onChunk,
    });

    processorNode.onaudioprocess?.({
      inputBuffer: {
        numberOfChannels: 2,
        length: 4,
        sampleRate: 16000,
        getChannelData(channel: number) {
          return channel === 0
            ? new Float32Array([1, 0.5, 0, -0.5])
            : new Float32Array([0, 0.5, 1, -0.5]);
        },
      },
    });

    expect(onChunk).toHaveBeenCalledOnce();
    expect(onChunk.mock.calls[0]?.[0]).toMatchObject({
      sampleRate: 16000,
      startFrame: 0,
      endFrame: 4,
      startTimeSeconds: 0,
    });
    expect(Array.from(onChunk.mock.calls[0]?.[0].pcm ?? [])).toEqual([0.5, 0.5, 0.5, -0.5]);

    await handle.stop();

    expect(processorNode.disconnect).toHaveBeenCalledOnce();
    expect(sourceNode.disconnect).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
    expect(stopTrack).toHaveBeenCalledOnce();
  });

  it('can write resampled microphone chunks into an audio ring buffer', async () => {
    const sourceNode = {
      connect: vi.fn(),
      disconnect: vi.fn(),
    };
    const processorNode = {
      onaudioprocess: null,
      connect: vi.fn(),
      disconnect: vi.fn(),
    };
    const close = vi.fn();
    const createAudioContext = vi.fn(() => ({
      sampleRate: 16000,
      destination: {},
      createMediaStreamSource: vi.fn(() => sourceNode),
      createScriptProcessor: vi.fn(() => processorNode),
      close,
    }));
    const stopTrack = vi.fn();
    const stream = {
      getTracks: () => [{ stop: stopTrack }],
    } as unknown as MediaStream;

    const handle = await startMicrophoneRingCapture({
      stream,
      createAudioContext,
      chunkFrames: 4,
      ringBufferDurationSeconds: 1,
      stopTracksOnStop: true,
    });

    processorNode.onaudioprocess?.({
      inputBuffer: {
        numberOfChannels: 1,
        length: 4,
        sampleRate: 16000,
        getChannelData() {
          return new Float32Array([0.25, 0.5, -0.25, -0.5]);
        },
      },
    });

    expect(handle.ringBuffer.getCurrentFrame()).toBe(4);
    expect(Array.from(handle.ringBuffer.read(0, 4))).toEqual([0.25, 0.5, -0.25, -0.5]);

    await handle.stop();

    expect(close).toHaveBeenCalledOnce();
    expect(stopTrack).toHaveBeenCalledOnce();
  });
});
