export interface DecodedWavMonoAudio {
  /** Owned mono float32 samples. Decoding does not resample or normalize loudness. */
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  /** Original channel count before arithmetic-mean downmixing. */
  readonly numberOfChannels: number;
}

function readAscii(view: DataView, offset: number, length: number): string {
  let value = '';
  for (let index = 0; index < length; index += 1) {
    value += String.fromCharCode(view.getUint8(offset + index));
  }
  return value;
}

function readPcm(view: DataView, offset: number, bits: number): number {
  switch (bits) {
    case 8:
      return (view.getUint8(offset) - 128) / 128;
    case 16:
      return view.getInt16(offset, true) / 32768;
    case 24: {
      let value =
        view.getUint8(offset) |
        (view.getUint8(offset + 1) << 8) |
        (view.getUint8(offset + 2) << 16);
      if (value & 0x800000) value |= ~0xffffff;
      return value / 8388608;
    }
    case 32:
      return view.getInt32(offset, true) / 2147483648;
    default:
      throw new Error(`Unsupported PCM WAV precision: ${bits}.`);
  }
}

/**
 * Decode RIFF/WAVE bytes without AudioContext, filesystem access or an engine.
 * Supports PCM 8/16/24/32 and IEEE float 32/64; compressed, RF64, RIFX and
 * WAVE_FORMAT_EXTENSIBLE inputs are deliberately unsupported.
 * A Uint8Array/Buffer slice is restricted to its actual byte range.
 */
export function decodeWavToMonoPcm(input: ArrayBuffer | Uint8Array): DecodedWavMonoAudio {
  const view =
    input instanceof Uint8Array
      ? new DataView(input.buffer, input.byteOffset, input.byteLength)
      : new DataView(input);
  if (
    view.byteLength < 12 ||
    readAscii(view, 0, 4) !== 'RIFF' ||
    readAscii(view, 8, 4) !== 'WAVE'
  ) {
    throw new Error('Invalid WAV: expected a complete RIFF/WAVE header.');
  }
  const end = view.getUint32(4, true) + 8;
  if (end < 12 || end > view.byteLength) throw new Error('Invalid WAV: truncated RIFF payload.');

  let format:
    | { encoding: number; channels: number; rate: number; bits: number; align: number }
    | undefined;
  let data: { offset: number; size: number } | undefined;
  let offset = 12;
  while (offset < end) {
    if (offset + 8 > end) throw new Error('Invalid WAV: truncated chunk header.');
    const id = readAscii(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const start = offset + 8;
    const next = start + size + (size % 2);
    if (next > end) throw new Error('Invalid WAV: truncated chunk payload or padding.');
    if (id === 'fmt ') {
      if (format || size < 16) throw new Error('Invalid WAV: duplicate or incomplete fmt chunk.');
      format = {
        encoding: view.getUint16(start, true),
        channels: view.getUint16(start + 2, true),
        rate: view.getUint32(start + 4, true),
        align: view.getUint16(start + 12, true),
        bits: view.getUint16(start + 14, true),
      };
      const supported =
        format.encoding === 1
          ? [8, 16, 24, 32].includes(format.bits)
          : format.encoding === 3 && [32, 64].includes(format.bits);
      if (!supported) throw new Error('Unsupported WAV encoding or precision.');
      if (
        !format.channels ||
        !format.rate ||
        format.align !== format.channels * (format.bits / 8) ||
        view.getUint32(start + 8, true) !== format.rate * format.align
      ) {
        throw new Error('Invalid WAV: channel, sample-rate or block-alignment metadata.');
      }
    } else if (id === 'data') {
      if (data) throw new Error('Invalid WAV: duplicate data chunk.');
      data = { offset: start, size };
    }
    offset = next;
  }
  if (!format || !data) throw new Error('Invalid WAV: missing fmt or data chunk.');
  if (data.size % format.align) throw new Error('Invalid WAV: incomplete audio frame.');

  const pcm = new Float32Array(data.size / format.align);
  const width = format.bits / 8;
  let cursor = data.offset;
  for (let frame = 0; frame < pcm.length; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const sample =
        format.encoding === 3
          ? format.bits === 32
            ? view.getFloat32(cursor, true)
            : view.getFloat64(cursor, true)
          : readPcm(view, cursor, format.bits);
      if (!Number.isFinite(sample)) throw new Error('Invalid WAV: samples must be finite.');
      // Avoid adding a positive-zero seed: mono IEEE -0 must retain its bits.
      sum = channel === 0 ? sample : sum + sample;
      cursor += width;
    }
    pcm[frame] = sum / format.channels;
    if (!Number.isFinite(pcm[frame]))
      throw new Error('Invalid WAV: mono float32 samples must be finite.');
  }
  return { pcm, sampleRate: format.rate, numberOfChannels: format.channels };
}
