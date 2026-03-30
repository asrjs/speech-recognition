import { SAMPLE_RATE } from './constants.js';
import { loadPcm16Wav } from './wav.js';

export type AudioInput = string | Uint8Array | ArrayBuffer | Int16Array | Float32Array | number[];

export interface NormalizedAudioInput {
  readonly wavPath?: string;
  readonly sampleRate: number;
  readonly pcm16: Int16Array;
}

function floatToInt16(value: number): number {
  if (Math.abs(value) <= 1.0) {
    return Math.round(value * 32767);
  }
  return Math.round(value);
}

function arrayToInt16(input: Float32Array | number[]): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    out[i] = floatToInt16(input[i]!);
  }
  return out;
}

export async function normalizeAudioInput(input: AudioInput): Promise<NormalizedAudioInput> {
  if (typeof input === 'string') {
    const parsed = await loadPcm16Wav(input);
    return {
      wavPath: input,
      sampleRate: parsed.sampleRate,
      pcm16: parsed.samples,
    };
  }
  if (input instanceof Uint8Array || input instanceof ArrayBuffer) {
    const parsed = await loadPcm16Wav(input);
    return {
      sampleRate: parsed.sampleRate,
      pcm16: parsed.samples,
    };
  }
  if (input instanceof Int16Array) {
    return {
      sampleRate: SAMPLE_RATE,
      pcm16: input,
    };
  }
  if (input instanceof Float32Array) {
    return {
      sampleRate: SAMPLE_RATE,
      pcm16: arrayToInt16(input),
    };
  }
  if (Array.isArray(input)) {
    return {
      sampleRate: SAMPLE_RATE,
      pcm16: arrayToInt16(input),
    };
  }

  throw new TypeError('Unsupported audio input type.');
}
