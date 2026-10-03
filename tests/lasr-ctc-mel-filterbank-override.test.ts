import { describe, expect, it } from 'vitest';

import { MedAsrJsPreprocessor } from '../src/models/lasr-ctc/mel.js';

function deterministicSignal(seconds: number): Float32Array {
  const samples = 16000 * seconds;
  const audio = new Float32Array(samples);
  for (let index = 0; index < samples; index += 1) {
    audio[index] = Math.sin((2 * Math.PI * 220 * index) / 16000) * 0.5 + 0.1;
  }

  return audio;
}

const N_MELS = 8;
const N_FFT = 512;
const N_FREQ_BINS = N_FFT / 2 + 1;

function flatBank(fill: (mel: number, bin: number) => number): Float64Array {
  const bank = new Float64Array(N_MELS * N_FREQ_BINS);
  for (let mel = 0; mel < N_MELS; mel += 1) {
    for (let bin = 0; bin < N_FREQ_BINS; bin += 1) {
      bank[mel * N_FREQ_BINS + bin] = fill(mel, bin);
    }
  }

  return bank;
}

const baseOptions = {
  nMels: N_MELS,
  nFft: N_FFT,
  winLength: 400,
  hopLength: 160,
  center: true,
  preemphasis: 0,
  melScale: 'slaney' as const,
  slaneyNorm: true,
  logZeroGuard: 5.960464477539063e-8,
  logCombine: 'add' as const,
  windowKind: 'hann' as const,
};

describe('LasrCtc mel filterbank override', () => {
  it('uses a supplied bank instead of the built-in filterbank', () => {
    const audio = deterministicSignal(1);
    const builtIn = new MedAsrJsPreprocessor(baseOptions).process(audio);
    const singleBand = new MedAsrJsPreprocessor({
      ...baseOptions,
      melFilterbank: flatBank((mel, bin) => (mel === 3 && bin % 2 === 0 ? 1 : 0)),
    }).process(audio);

    const frames = builtIn.frameCount;
    const read = (result: typeof builtIn, mel: number): number =>
      result.features[mel * frames + 1] ?? Number.NaN;

    // Only band 3 is populated by the override, so every other band collapses
    // to the log guard floor while band 3 stays finite.
    expect(read(builtIn, 3)).toBeGreaterThan(Number.NEGATIVE_INFINITY);
    expect(read(builtIn, 3)).not.toBe(read(singleBand, 3));
    expect(read(singleBand, 3)).toBeGreaterThan(-16.6);
    for (const mel of [0, 1, 2, 4, 5, 6, 7]) {
      expect(read(singleBand, mel)).toBeCloseTo(Math.log(5.960464477539063e-8), 6);
    }
  });

  it('emits the trailing centered frame when the family opts in', () => {
    const audio = deterministicSignal(1);
    const clamped = new MedAsrJsPreprocessor(baseOptions).process(audio);
    const withTail = new MedAsrJsPreprocessor({
      ...baseOptions,
      includeCenteredTailFrame: true,
    }).process(audio);

    // torch.stft(center=True) yields floor((samples + nFft - winLength)/hop) + 1
    // frames; the default clamp stops one frame short of that.
    expect(withTail.frameCount).toBe(Math.floor((audio.length + N_FFT - 400) / 160) + 1);
    expect(withTail.frameCount).toBe(clamped.frameCount + 1);
  });

  it('rejects a bank whose length does not match nMels * (nFft / 2 + 1)', () => {
    expect(
      () =>
        new MedAsrJsPreprocessor({
          ...baseOptions,
          melFilterbank: new Float64Array(N_MELS * N_FREQ_BINS - 1),
        }),
    ).toThrow(/melFilterbank must hold nMels/);
    expect(
      () =>
        new MedAsrJsPreprocessor({
          ...baseOptions,
          melFilterbank: new Float64Array(N_MELS * N_FREQ_BINS + 1),
        }),
    ).toThrow(/melFilterbank must hold nMels/);
  });

  it('leaves existing families unchanged when no override is supplied', () => {
    const audio = deterministicSignal(1);
    const before = new MedAsrJsPreprocessor(baseOptions).process(audio);
    const after = new MedAsrJsPreprocessor({ ...baseOptions }).process(audio);
    expect(Array.from(after.features)).toEqual(Array.from(before.features));
  });
});
