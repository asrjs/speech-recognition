import { decodeAudioSourceToMonoPcm, mixAudioBufferChannelsToMono, resolveAudioSourceToBlob } from 'asr.js';
import { describe, expect, it, vi } from 'vitest';

describe('media helpers', () => {
  it('mixes multichannel audio buffers down to mono', () => {
    const mono = mixAudioBufferChannelsToMono({
      numberOfChannels: 2,
      length: 3,
      sampleRate: 16000,
      getChannelData(channel) {
        return channel === 0
          ? new Float32Array([1, 0, -1])
          : new Float32Array([0, 1, -1]);
      }
    });

    expect(Array.from(mono)).toEqual([0.5, 0.5, -1]);
  });

  it('resolves remote audio sources through fetch', async () => {
    const blob = new Blob(['audio']);
    const fetchImpl = vi.fn(async () => new Response(blob, { status: 200 }));

    const resolved = await resolveAudioSourceToBlob('https://example.test/audio.wav', { fetchImpl });

    expect(fetchImpl).toHaveBeenCalledOnce();
    expect(await resolved.text()).toBe('audio');
  });

  it('decodes audio sources into mono PCM with an injected AudioContext', async () => {
    const close = vi.fn();
    const createAudioContext = vi.fn(() => ({
      decodeAudioData: vi.fn(async () => ({
        numberOfChannels: 2,
        length: 4,
        sampleRate: 16000,
        getChannelData(channel: number) {
          return channel === 0
            ? new Float32Array([1, 0.5, 0, -0.5])
            : new Float32Array([0, 0.5, 1, -0.5]);
        }
      })),
      close
    }));

    const decoded = await decodeAudioSourceToMonoPcm(new Blob(['audio']), { createAudioContext });

    expect(createAudioContext).toHaveBeenCalledWith(16000);
    expect(close).toHaveBeenCalledOnce();
    expect(decoded.sampleRate).toBe(16000);
    expect(decoded.durationSec).toBeCloseTo(4 / 16000, 8);
    expect(Array.from(decoded.pcm)).toEqual([0.5, 0.5, 0.5, -0.5]);
  });
});
