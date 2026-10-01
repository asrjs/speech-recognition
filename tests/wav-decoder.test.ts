import { describe, expect, it, vi } from 'vitest';
import { decodeWavToMonoPcm } from '../src/audio/wav.js';
import { decodeAudioSourceToMonoPcm } from '../src/runtime/media.js';

function wav(
  values: readonly number[] = [0, 16384, -32768],
  { bits = 16, format = 1, channels = 1, rate = 16000, metadata = false } = {},
): Uint8Array<ArrayBuffer> {
  const width = bits / 8;
  const size = values.length * width;
  const prefix = metadata ? 10 : 0;
  const bytes = new Uint8Array(44 + prefix + size + (size % 2));
  const view = new DataView(bytes.buffer);
  const ascii = (offset: number, value: string) =>
    [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  ascii(0, 'RIFF');
  view.setUint32(4, bytes.length - 8, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * channels * width, true);
  view.setUint16(32, channels * width, true);
  view.setUint16(34, bits, true);
  if (metadata) {
    ascii(36, 'JUNK');
    view.setUint32(40, 1, true);
    view.setUint8(44, 99);
  }
  ascii(36 + prefix, 'data');
  view.setUint32(40 + prefix, size, true);
  values.forEach((value, index) => {
    const offset = 44 + prefix + index * width;
    if (format === 3 && bits === 32) view.setFloat32(offset, value, true);
    else if (format === 3 && bits === 64) view.setFloat64(offset, value, true);
    else if (bits === 8) view.setUint8(offset, value);
    else if (bits === 16) view.setInt16(offset, value, true);
    else if (bits === 24) {
      view.setUint8(offset, value & 255);
      view.setUint8(offset + 1, (value >> 8) & 255);
      view.setUint8(offset + 2, (value >> 16) & 255);
    } else if (bits === 32) view.setInt32(offset, value, true);
  });
  return bytes;
}

describe('portable WAV decoding', () => {
  it('exposes the same pure helper from opt-in native and browser-media entrypoints', async () => {
    const node = await import('../src/io-node.js');
    const browser = await import('../src/browser-media.js');
    expect(node.decodeWavToMonoPcm).toBe(decodeWavToMonoPcm);
    expect(browser.decodeWavToMonoPcm).toBe(decodeWavToMonoPcm);
  });

  it.each([
    [8, [0, 128, 255], [-1, 0, 127 / 128]],
    [16, [-32768, 0, 32767], [-1, 0, 32767 / 32768]],
    [24, [-8388608, 0, 8388607], [-1, 0, 8388607 / 8388608]],
    [32, [-2147483648, 0, 2147483647], [-1, 0, 2147483647 / 2147483648]],
  ])('decodes signed/unsigned PCM %i without resampling', (bits, input, expected) => {
    const result = decodeWavToMonoPcm(wav(input, { bits }));
    expect(result.pcm).toEqual(Float32Array.from(expected));
    expect(result.sampleRate).toBe(16000);
    expect(result.numberOfChannels).toBe(1);
  });

  it.each([32, 64])('decodes IEEE float %i', (bits) => {
    expect(decodeWavToMonoPcm(wav([0.25, -0.5], { bits, format: 3 })).pcm).toEqual(
      new Float32Array([0.25, -0.5]),
    );
  });

  it.each([32, 64])('preserves IEEE float %i signed zero in mono PCM', (bits) => {
    const decoded = decodeWavToMonoPcm(wav([-0, 0, -0.125], { bits, format: 3 }));
    expect(Object.is(decoded.pcm[0], -0)).toBe(true);
    expect(Object.is(decoded.pcm[1], 0)).toBe(true);
    expect(new Uint32Array(decoded.pcm.buffer)[0]).toBe(0x80000000);
  });

  it('honors a sliced Buffer byte range, odd metadata and channel mixing', () => {
    const bytes = wav([32767, -32768, 16384, 16384], { channels: 2, metadata: true });
    const storage = Buffer.alloc(bytes.length + 9, 255);
    storage.set(bytes, 3);
    const result = decodeWavToMonoPcm(storage.subarray(3, 3 + bytes.length));
    expect(result.pcm).toEqual(new Float32Array([-1 / 65536, 0.5]));
    expect(result.numberOfChannels).toBe(2);
    storage.fill(0);
    expect(result.pcm[1]).toBe(0.5);
  });

  it('preserves browser native-rate WAV output through the shared implementation', async () => {
    const bytes = wav([0.125, -0.25, 0.75, 0.5], { format: 3, bits: 32, channels: 2 });
    const createAudioContext = vi.fn();
    const browser = await decodeAudioSourceToMonoPcm(new Blob([bytes]), {
      strategy: 'native-rate',
      targetSampleRate: 16000,
      createAudioContext,
    });
    expect(browser.pcm).toEqual(decodeWavToMonoPcm(bytes).pcm);
    expect(createAudioContext).not.toHaveBeenCalled();
  });

  it('accepts an empty data chunk without inventing audio', () => {
    expect(decodeWavToMonoPcm(wav([])).pcm.length).toBe(0);
  });

  it.each([
    ['short header', (bytes: Uint8Array) => bytes.subarray(0, 10)],
    ['truncated RIFF', (bytes: Uint8Array) => bytes.subarray(0, bytes.length - 2)],
    [
      'truncated chunk',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint32(40, 999, true);
        return bytes;
      },
    ],
    [
      'short fmt',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint32(16, 8, true);
        return bytes;
      },
    ],
    [
      'zero channels',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint16(22, 0, true);
        return bytes;
      },
    ],
    [
      'zero rate',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint32(24, 0, true);
        return bytes;
      },
    ],
    [
      'unsupported encoding',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint16(20, 2, true);
        return bytes;
      },
    ],
    [
      'unsupported precision',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint16(34, 12, true);
        return bytes;
      },
    ],
    [
      'bad block alignment',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint16(32, 3, true);
        return bytes;
      },
    ],
    [
      'incomplete frame',
      (bytes: Uint8Array) => {
        new DataView(bytes.buffer).setUint32(40, 5, true);
        return bytes;
      },
    ],
    [
      'missing data',
      (bytes: Uint8Array) => {
        bytes.set([74, 85, 78, 75], 36);
        return bytes;
      },
    ],
  ])('fails closed for %s', (_name, corrupt) => {
    expect(() => decodeWavToMonoPcm(corrupt(wav()))).toThrow(/WAV/);
  });

  it.each([NaN, Infinity, -Infinity, 1e100])(
    'rejects nonfinite float32 results from %s',
    (value) => {
      expect(() => decodeWavToMonoPcm(wav([value], { format: 3, bits: 64 }))).toThrow(/finite/);
    },
  );
});
