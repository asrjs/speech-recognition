import {
  decodeAudioBufferToMonoPcm,
  decodeAudioSourceToMonoPcm,
  encodeMonoPcmToWavBlob,
  mixAudioBufferChannelsToMono,
  resolveAudioSourceToBlob,
} from '@asrjs/speech-recognition/browser';
import { describe, expect, it, vi } from 'vitest';

function createMonoPcm16Wav(samples: readonly number[], sampleRate: number): Blob {
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeAscii(36, 'data');
  view.setUint32(40, dataSize, true);

  samples.forEach((sample, index) => {
    view.setInt16(44 + index * 2, sample, true);
  });

  return new Blob([buffer], { type: 'audio/wav' });
}

describe('media helpers', () => {
  it('mixes multichannel audio buffers down to mono', () => {
    const mono = mixAudioBufferChannelsToMono({
      numberOfChannels: 2,
      length: 3,
      sampleRate: 16000,
      getChannelData(channel) {
        return channel === 0 ? new Float32Array([1, 0, -1]) : new Float32Array([0, 1, -1]);
      },
    });

    expect(Array.from(mono)).toEqual([0.5, 0.5, -1]);
  });

  it('resolves remote audio sources through fetch', async () => {
    const blob = new Blob(['audio']);
    const fetchImpl = vi.fn(async () => new Response(blob, { status: 200 }));

    const resolved = await resolveAudioSourceToBlob('https://example.test/audio.wav', {
      fetchImpl,
    });

    expect(fetchImpl).toHaveBeenCalledOnce();
    expect(await resolved.text()).toBe('audio');
  });

  it('encodes mono PCM into a WAV blob', async () => {
    const blob = encodeMonoPcmToWavBlob(new Float32Array([0, 1, -1]), { sampleRate: 8000 });
    const buffer = await blob.arrayBuffer();
    const view = new DataView(buffer);
    const readAscii = (offset: number, length: number) =>
      Array.from({ length }, (_, index) => String.fromCharCode(view.getUint8(offset + index))).join(
        '',
      );

    expect(blob.type).toBe('audio/wav');
    expect(readAscii(0, 4)).toBe('RIFF');
    expect(readAscii(8, 4)).toBe('WAVE');
    expect(view.getUint32(24, true)).toBe(8000);
    expect(view.getUint16(22, true)).toBe(1);
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
        },
      })),
      close,
    }));

    const decoded = await decodeAudioSourceToMonoPcm(new Blob(['audio']), { createAudioContext });

    expect(createAudioContext).toHaveBeenCalledWith(16000);
    expect(close).toHaveBeenCalledOnce();
    expect(decoded.sampleRate).toBe(16000);
    expect(decoded.durationSec).toBeCloseTo(4 / 16000, 8);
    expect(Array.from(decoded.pcm)).toEqual([0.5, 0.5, 0.5, -0.5]);
    expect(decoded.metrics.backend).toBe('browser');
    expect(decoded.metrics.strategy).toBe('audiocontext-target-rate');
    expect(decoded.metrics.outputSampleRate).toBe(16000);
    expect(decoded.metrics.decodeMs).toBeGreaterThanOrEqual(0);
    expect(decoded.metrics.downmixMs).toBeGreaterThanOrEqual(0);
    expect(decoded.metrics.wallMs).toBeGreaterThanOrEqual(decoded.metrics.totalMs);
    expect(decoded.metrics.audioDurationSec).toBeCloseTo(4 / 16000, 8);
  });

  it('supports native-rate decode with explicit linear resampling', async () => {
    const close = vi.fn();
    const createAudioContext = vi.fn(() => ({
      decodeAudioData: vi.fn(async () => ({
        numberOfChannels: 1,
        length: 4,
        sampleRate: 4,
        getChannelData() {
          return new Float32Array([0, 1, 0, -1]);
        },
      })),
      close,
    }));

    const decoded = await decodeAudioSourceToMonoPcm(new Blob(['audio']), {
      targetSampleRate: 3,
      strategy: 'native-rate',
      resamplerQuality: 'linear',
      createAudioContext,
    });

    expect(createAudioContext).toHaveBeenCalledWith();
    expect(close).toHaveBeenCalledOnce();
    expect(decoded.sampleRate).toBe(3);
    expect(decoded.pcm[0]).toBeCloseTo(0, 6);
    expect(decoded.pcm[1]).toBeCloseTo(2 / 3, 6);
    expect(decoded.pcm[2]).toBeCloseTo(-(2 / 3), 6);
    expect(decoded.metrics.strategy).toBe('audiocontext-native-rate');
    expect(decoded.metrics.inputSampleRate).toBe(4);
    expect(decoded.metrics.outputSampleRate).toBe(3);
    expect(decoded.metrics.resampler).toBe('linear-parity');
    expect(decoded.metrics.resamplerQuality).toBe('linear');
    expect(decoded.metrics.resampleMs).toBeGreaterThanOrEqual(0);
  });

  it('uses the deterministic WAV parser path for native-rate browser decoding', async () => {
    const createAudioContext = vi.fn();
    const wav = createMonoPcm16Wav([0, 32767, 0, -32768], 4);

    const decoded = await decodeAudioSourceToMonoPcm(wav, {
      targetSampleRate: 3,
      strategy: 'native-rate',
      createAudioContext,
    });

    expect(createAudioContext).not.toHaveBeenCalled();
    expect(decoded.sampleRate).toBe(3);
    expect(decoded.metrics.strategy).toBe('audiocontext-native-rate');
    expect(decoded.metrics.inputSampleRate).toBe(4);
    expect(decoded.metrics.resampler).toBe('linear-parity');
    expect(decoded.pcm[0]).toBeCloseTo(0, 4);
    expect(decoded.pcm[1]).toBeCloseTo(2 / 3, 3);
    expect(decoded.pcm[2]).toBeCloseTo(-(2 / 3), 3);
  });

  it('decodes ArrayBuffer audio sources through the shared wrapper', async () => {
    const createAudioContext = vi.fn(() => ({
      decodeAudioData: vi.fn(async () => ({
        numberOfChannels: 1,
        length: 2,
        sampleRate: 16000,
        getChannelData() {
          return new Float32Array([0.25, -0.25]);
        },
      })),
      close: vi.fn(),
    }));

    const decoded = await decodeAudioBufferToMonoPcm(await new Blob(['audio']).arrayBuffer(), {
      createAudioContext,
    });

    expect(createAudioContext).toHaveBeenCalledWith(16000);
    expect(Array.from(decoded.pcm)).toEqual([0.25, -0.25]);
  });
});
