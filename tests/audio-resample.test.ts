import { SincResampler, resampleSinc } from '../src/audio/resample.js';
import { resampleLinear } from '../src/runtime/media.js';
import { startMicrophoneCapture } from '@asrjs/speech-recognition/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

function tone(frequency: number, sampleRate: number, seconds = 1, amplitude = 0.5): Float32Array {
  const samples = new Float32Array(Math.round(sampleRate * seconds));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = amplitude * Math.sin((2 * Math.PI * frequency * index) / sampleRate);
  }
  return samples;
}

function mix(...signals: Float32Array[]): Float32Array {
  const out = new Float32Array(signals[0]!.length);
  for (const signal of signals) {
    for (let index = 0; index < out.length; index += 1) out[index]! += signal[index]!;
  }
  return out;
}

/** Amplitude of one frequency component (single-bin DFT), edges skipped. */
function amplitudeAt(samples: Float32Array, frequency: number, sampleRate: number, skip = 400) {
  const omega = (2 * Math.PI * frequency) / sampleRate;
  let real = 0;
  let imaginary = 0;
  let count = 0;
  for (let index = skip; index < samples.length - skip; index += 1) {
    real += samples[index]! * Math.cos(omega * index);
    imaginary -= samples[index]! * Math.sin(omega * index);
    count += 1;
  }
  return (2 * Math.hypot(real, imaginary)) / count;
}

const decibels = (ratio: number) => 20 * Math.log10(Math.max(ratio, 1e-12));

describe('resampleSinc', () => {
  it.each([48000, 44100])(
    'suppresses content above 8 kHz when converting %i Hz to 16 kHz',
    (from) => {
      for (const [frequency, alias] of [
        [9000, 7000],
        [14000, 2000],
      ] as const) {
        const input = tone(frequency, from);
        const sinc = resampleSinc(input, from, 16000);
        const linear = resampleLinear(input, from, 16000);
        expect(decibels(amplitudeAt(sinc, alias, 16000) / 0.5)).toBeLessThan(-60);
        // The linear path folds the same tone back almost unattenuated.
        expect(decibels(amplitudeAt(linear, alias, 16000) / 0.5)).toBeGreaterThan(-6);
      }
    },
  );

  it('keeps the speech band flat', () => {
    for (const frequency of [300, 1000, 4000, 6000]) {
      const output = resampleSinc(tone(frequency, 48000), 48000, 16000);
      expect(Math.abs(decibels(amplitudeAt(output, frequency, 16000) / 0.5))).toBeLessThan(0.1);
    }
  });

  it('matches the linear resampler output length and returns equal-rate input unchanged', () => {
    for (const [length, from, to] of [
      [48000, 48000, 16000],
      [44100, 44100, 16000],
      [1001, 22050, 16000],
      [8000, 8000, 16000],
      [1, 48000, 16000],
    ] as const) {
      const input = tone(440, from, length / from);
      expect(resampleSinc(input, from, to).length).toBe(resampleLinear(input, from, to).length);
    }
    const same = tone(440, 16000);
    expect(resampleSinc(same, 16000, 16000)).toBe(same);
  });

  it('rejects invalid sample rates', () => {
    expect(() => new SincResampler(0, 16000)).toThrow(RangeError);
    expect(() => new SincResampler(48000, Number.NaN)).toThrow(RangeError);
  });
});

describe('SincResampler streaming', () => {
  it('produces the same samples regardless of input chunking', () => {
    const input = mix(tone(1000, 44100, 0.5), tone(9000, 44100, 0.5, 0.2));
    const whole = new SincResampler(44100, 16000);
    const expected = [...whole.process(input), ...whole.flush()];

    const chunked = new SincResampler(44100, 16000);
    const actual: number[] = [];
    let offset = 0;
    for (const size of [1, 7, 128, 3, 4096, 999, 128]) {
      if (offset >= input.length) break;
      actual.push(...chunked.process(input.subarray(offset, offset + size)));
      offset += size;
    }
    actual.push(...chunked.process(input.subarray(offset)), ...chunked.flush());

    expect(actual.length).toBe(expected.length);
    for (let index = 0; index < expected.length; index += 1) {
      expect(actual[index]).toBeCloseTo(expected[index]!, 6);
    }
  });

  it('copies samples through unchanged at equal rates', () => {
    const resampler = new SincResampler(16000, 16000);
    const input = Float32Array.from([0.25, -0.5, 0.75]);
    const output = resampler.process(input);
    expect(Array.from(output)).toEqual(Array.from(input));
    expect(output).not.toBe(input);
    expect(resampler.flush().length).toBe(0);
  });
});

describe('microphone capture resampling', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function createContext(sampleRate: number, extra: Record<string, unknown> = {}) {
    const processorNode = { onaudioprocess: null, connect: vi.fn(), disconnect: vi.fn() };
    return {
      processorNode,
      context: {
        sampleRate,
        destination: {},
        createMediaStreamSource: vi.fn(() => ({ connect: vi.fn(), disconnect: vi.fn() })),
        createScriptProcessor: vi.fn(() => processorNode),
        close: vi.fn(),
        ...extra,
      },
    };
  }

  const stream = {
    getTracks: () => [{ stop: vi.fn() }],
    getAudioTracks: () => [{ getSettings: () => ({ sampleRate: 48000 }) }],
  } as unknown as MediaStream;

  it('band-limits the AudioWorklet capture path at 48 kHz', async () => {
    let moduleSource = '';
    vi.spyOn(URL, 'createObjectURL').mockImplementation(() => 'blob:capture-worklet');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
    const blobs: Blob[] = [];
    const OriginalBlob = globalThis.Blob;
    vi.stubGlobal(
      'Blob',
      class extends OriginalBlob {
        constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
          super(parts, options);
          blobs.push(this);
        }
      },
    );
    const nodes: { port: { onmessage: ((event: { data: unknown }) => void) | null } }[] = [];
    vi.stubGlobal(
      'AudioWorkletNode',
      class {
        port = { onmessage: null as ((event: { data: unknown }) => void) | null };
        onprocessorerror = null;
        connect = vi.fn();
        disconnect = vi.fn();
        constructor() {
          nodes.push(this);
        }
      },
    );
    const { context } = createContext(48000, {
      audioWorklet: {
        addModule: vi.fn(async () => {
          moduleSource = await blobs.at(-1)!.text();
        }),
      },
    });
    const chunks: { pcm: Float32Array; sampleRate: number }[] = [];

    const handle = await startMicrophoneCapture({
      stream,
      createAudioContext: () => context as never,
      chunkFrames: 512,
      onChunk: (chunk) => chunks.push(chunk),
    });

    // Run the generated module in a minimal AudioWorkletGlobalScope.
    let Processor: new (options: unknown) => {
      port: { postMessage(message: unknown): void };
      process(inputs: Float32Array[][]): boolean;
    } = undefined as never;
    new Function('AudioWorkletProcessor', 'registerProcessor', 'sampleRate', moduleSource)(
      class {
        port = { postMessage: (_message: unknown) => undefined };
      },
      (_name: string, ctor: typeof Processor) => {
        Processor = ctor;
      },
      48000,
    );
    const processor = new Processor({
      processorOptions: { targetSampleRate: 16000, targetChunkFrames: 512 },
    });
    processor.port.postMessage = (message) => nodes[0]!.port.onmessage?.({ data: message });

    const input = mix(tone(1000, 48000), tone(9000, 48000));
    for (let offset = 0; offset + 128 <= input.length; offset += 128) {
      processor.process([[input.subarray(offset, offset + 128)]]);
    }

    expect(chunks.length).toBeGreaterThan(25);
    expect(chunks.every((chunk) => chunk.pcm.length === 512 && chunk.sampleRate === 16000)).toBe(
      true,
    );
    const output = new Float32Array(chunks.length * 512);
    chunks.forEach((chunk, index) => output.set(chunk.pcm, index * 512));
    expect(Math.abs(decibels(amplitudeAt(output, 1000, 16000) / 0.5))).toBeLessThan(0.1);
    expect(decibels(amplitudeAt(output, 7000, 16000) / 0.5)).toBeLessThan(-60);

    await handle.stop();
  });

  it('band-limits the ScriptProcessor fallback at 48 kHz', async () => {
    const { context, processorNode } = createContext(48000);
    const chunks: Float32Array[] = [];
    const handle = await startMicrophoneCapture({
      stream,
      createAudioContext: () => context as never,
      chunkFrames: 400,
      onChunk: (chunk) => chunks.push(chunk.pcm),
    });

    const input = mix(tone(1000, 48000), tone(14000, 48000));
    for (let offset = 0; offset + 1024 <= input.length; offset += 1024) {
      const block = input.subarray(offset, offset + 1024);
      (processorNode.onaudioprocess as unknown as (event: unknown) => void)({
        inputBuffer: {
          numberOfChannels: 1,
          length: block.length,
          sampleRate: 48000,
          getChannelData: () => block,
        },
      });
    }

    expect(chunks.every((chunk) => chunk.length === 400)).toBe(true);
    const output = new Float32Array(chunks.length * 400);
    chunks.forEach((chunk, index) => output.set(chunk, index * 400));
    expect(Math.abs(decibels(amplitudeAt(output, 1000, 16000) / 0.5))).toBeLessThan(0.1);
    expect(decibels(amplitudeAt(output, 2000, 16000) / 0.5)).toBeLessThan(-60);

    await handle.stop();
  });
});
