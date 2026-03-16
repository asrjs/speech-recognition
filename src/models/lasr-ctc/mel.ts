import type { LasrCtcFeatureBatch, LasrCtcFeaturePreprocessor } from './types.js';

const SAMPLE_RATE = 16000;
const N_FFT = 512;
const WIN_LENGTH = 400;
const HOP_LENGTH = 160;
const PREEMPH = 0.97;
const LOG_ZERO_GUARD = 2 ** -24;
const N_FREQ_BINS = (N_FFT >> 1) + 1;

type MelScaleKind = 'slaney' | 'kaldi';

interface MelTwiddles {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly bitReverse: Uint32Array;
}

export interface MedAsrJsPreprocessorOptions {
  readonly nMels?: number;
  readonly center?: boolean;
  readonly preemphasis?: number;
  readonly melScale?: MelScaleKind;
  readonly slaneyNorm?: boolean;
  readonly logZeroGuard?: number;
  readonly normalizeFeatures?: boolean;
}

const MEL_FILTERBANK_CACHE = new Map<string, Float32Array>();
const FFT_TWIDDLE_CACHE = new Map<number, MelTwiddles>();
const HANN_WINDOW_CACHE = new Map<'center' | 'left', Float64Array>();

const F_SP = 200 / 3;
const MIN_LOG_HZ = 1000;
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;
const LOG_STEP = Math.log(6.4) / 27;

function hzToMel(frequencyHz: number): number {
  if (frequencyHz >= MIN_LOG_HZ) {
    return MIN_LOG_MEL + Math.log(frequencyHz / MIN_LOG_HZ) / LOG_STEP;
  }

  return frequencyHz / F_SP;
}

function melToHz(mel: number): number {
  if (mel >= MIN_LOG_MEL) {
    return MIN_LOG_HZ * Math.exp(LOG_STEP * (mel - MIN_LOG_MEL));
  }

  return mel * F_SP;
}

function createMelFilterbank(
  nMels: number,
  melScale: MelScaleKind,
  slaneyNorm: boolean,
): Float32Array {
  const frequencyMin = melScale === 'kaldi' ? 125 : 0;
  const frequencyMax = melScale === 'kaldi' ? 7500 : SAMPLE_RATE / 2;

  const toMel =
    melScale === 'kaldi'
      ? (frequency: number): number => 1127 * Math.log(1 + frequency / 700)
      : hzToMel;
  const toHz =
    melScale === 'kaldi' ? (mel: number): number => 700 * (Math.exp(mel / 1127) - 1) : melToHz;

  const allFrequencies = new Float64Array(N_FREQ_BINS);
  for (let index = 0; index < N_FREQ_BINS; index += 1) {
    allFrequencies[index] = ((SAMPLE_RATE / 2) * index) / (N_FREQ_BINS - 1);
  }

  const melMin = toMel(frequencyMin);
  const melMax = toMel(frequencyMax);
  const melPoints = nMels + 2;
  const melFrequencies = new Float64Array(melPoints);
  for (let index = 0; index < melPoints; index += 1) {
    melFrequencies[index] = toHz(melMin + ((melMax - melMin) * index) / (melPoints - 1));
  }

  const melDifferences = new Float64Array(melPoints - 1);
  for (let index = 0; index < melPoints - 1; index += 1) {
    const current = melFrequencies[index] ?? 0;
    const next = melFrequencies[index + 1] ?? current;
    melDifferences[index] = next - current;
  }

  const filterbank = new Float32Array(nMels * N_FREQ_BINS);
  for (let melIndex = 0; melIndex < nMels; melIndex += 1) {
    const centerLeft = melFrequencies[melIndex] ?? 0;
    const centerRight = melFrequencies[melIndex + 2] ?? centerLeft;
    const lowerDelta = melDifferences[melIndex] ?? 1;
    const upperDelta = melDifferences[melIndex + 1] ?? 1;
    const normalization = slaneyNorm ? 2 / Math.max(1e-12, centerRight - centerLeft) : 1;
    const rowOffset = melIndex * N_FREQ_BINS;

    for (let frequencyIndex = 0; frequencyIndex < N_FREQ_BINS; frequencyIndex += 1) {
      const frequency = allFrequencies[frequencyIndex] ?? 0;
      const downSlope = (frequency - centerLeft) / Math.max(1e-12, lowerDelta);
      const upSlope = (centerRight - frequency) / Math.max(1e-12, upperDelta);
      filterbank[rowOffset + frequencyIndex] =
        Math.max(0, Math.min(downSlope, upSlope)) * normalization;
    }
  }

  return filterbank;
}

function getCachedMelFilterbank(
  nMels: number,
  melScale: MelScaleKind,
  slaneyNorm: boolean,
): Float32Array {
  const cacheKey = `${nMels}:${melScale}:${slaneyNorm}`;
  const cached = MEL_FILTERBANK_CACHE.get(cacheKey);
  if (cached) {
    return cached;
  }

  const created = createMelFilterbank(nMels, melScale, slaneyNorm);
  MEL_FILTERBANK_CACHE.set(cacheKey, created);
  return created;
}

function createPaddedHannWindow(centerWindow: boolean): Float64Array {
  const window = new Float64Array(N_FFT);
  const leftPad = centerWindow ? (N_FFT - WIN_LENGTH) >> 1 : 0;

  for (let index = 0; index < WIN_LENGTH; index += 1) {
    window[leftPad + index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / (WIN_LENGTH - 1)));
  }

  return window;
}

function getCachedPaddedHannWindow(centerWindow: boolean): Float64Array {
  const key: 'center' | 'left' = centerWindow ? 'center' : 'left';
  const cached = HANN_WINDOW_CACHE.get(key);
  if (cached) {
    return cached;
  }

  const created = createPaddedHannWindow(centerWindow);
  HANN_WINDOW_CACHE.set(key, created);
  return created;
}

function precomputeFftTwiddles(size: number): MelTwiddles {
  const cached = FFT_TWIDDLE_CACHE.get(size);
  if (cached) {
    return cached;
  }

  const bits = Math.log2(size);
  if (1 << bits !== size) {
    throw new Error(`FFT size must be a power-of-two. Received: ${size}.`);
  }

  const half = size >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let index = 0; index < half; index += 1) {
    const angle = (-2 * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }

  const bitReverse = new Uint32Array(size);
  for (let index = 0; index < size; index += 1) {
    let value = index;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    bitReverse[index] = reversed;
  }

  const twiddles = { cos, sin, bitReverse };
  FFT_TWIDDLE_CACHE.set(size, twiddles);
  return twiddles;
}

function fft(
  real: Float64Array,
  imaginary: Float64Array,
  size: number,
  twiddles: MelTwiddles,
): void {
  for (let index = 0; index < size; index += 1) {
    const swappedIndex = twiddles.bitReverse[index] ?? index;
    if (index >= swappedIndex) {
      continue;
    }

    const realValue = real[index] ?? 0;
    real[index] = real[swappedIndex] ?? 0;
    real[swappedIndex] = realValue;

    const imaginaryValue = imaginary[index] ?? 0;
    imaginary[index] = imaginary[swappedIndex] ?? 0;
    imaginary[swappedIndex] = imaginaryValue;
  }

  for (let length = 2; length <= size; length <<= 1) {
    const halfLength = length >> 1;
    const twiddleStep = size / length;

    for (let segment = 0; segment < size; segment += length) {
      for (let offset = 0; offset < halfLength; offset += 1) {
        const twiddleIndex = offset * twiddleStep;
        const cosine = twiddles.cos[twiddleIndex] ?? 0;
        const sine = twiddles.sin[twiddleIndex] ?? 0;
        const first = segment + offset;
        const second = first + halfLength;

        const tReal = (real[second] ?? 0) * cosine - (imaginary[second] ?? 0) * sine;
        const tImaginary = (real[second] ?? 0) * sine + (imaginary[second] ?? 0) * cosine;
        const uReal = real[first] ?? 0;
        const uImaginary = imaginary[first] ?? 0;

        real[first] = uReal + tReal;
        imaginary[first] = uImaginary + tImaginary;
        real[second] = uReal - tReal;
        imaginary[second] = uImaginary - tImaginary;
      }
    }
  }
}

export function transposeMelToTxM(
  featuresMxT: Float32Array,
  nMels: number,
  frameCount: number,
): Float32Array {
  const transposed = new Float32Array(frameCount * nMels);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    for (let melIndex = 0; melIndex < nMels; melIndex += 1) {
      transposed[frameIndex * nMels + melIndex] =
        featuresMxT[melIndex * frameCount + frameIndex] ?? 0;
    }
  }

  return transposed;
}

interface RawMelOutput {
  readonly rawMel: Float32Array;
  readonly frameCount: number;
  readonly validFrameCount: number;
}

export class MedAsrJsPreprocessor implements LasrCtcFeaturePreprocessor {
  readonly nMels: number;
  private readonly center: boolean;
  private readonly preemphasis: number;
  private readonly melScale: MelScaleKind;
  private readonly slaneyNorm: boolean;
  private readonly logZeroGuard: number;
  private readonly normalizeFeatures: boolean;
  private readonly melFilterbank: Float32Array;
  private readonly hannWindow: Float64Array;
  private readonly fftTwiddles: MelTwiddles;
  private readonly fftReal = new Float64Array(N_FFT);
  private readonly fftImaginary = new Float64Array(N_FFT);
  private readonly powerBuffer = new Float32Array(N_FREQ_BINS);
  private readonly filterbankBounds: Int32Array;

  private emphasizedBuffer = new Float32Array(0);
  private paddedBuffer = new Float64Array(0);

  constructor(options: MedAsrJsPreprocessorOptions = {}) {
    this.nMels = options.nMels ?? 128;
    this.center = options.center ?? false;
    this.preemphasis = options.preemphasis ?? PREEMPH;
    this.melScale = options.melScale ?? 'kaldi';
    this.slaneyNorm = options.slaneyNorm ?? false;
    this.logZeroGuard = options.logZeroGuard ?? LOG_ZERO_GUARD;
    this.normalizeFeatures = options.normalizeFeatures ?? false;

    this.melFilterbank = getCachedMelFilterbank(this.nMels, this.melScale, this.slaneyNorm);
    this.hannWindow = getCachedPaddedHannWindow(this.center);
    this.fftTwiddles = precomputeFftTwiddles(N_FFT);

    this.filterbankBounds = new Int32Array(this.nMels * 2);
    for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
      const offset = melIndex * N_FREQ_BINS;
      let start = -1;
      let end = -1;
      for (let frequencyIndex = 0; frequencyIndex < N_FREQ_BINS; frequencyIndex += 1) {
        if ((this.melFilterbank[offset + frequencyIndex] ?? 0) > 0) {
          if (start < 0) {
            start = frequencyIndex;
          }
          end = frequencyIndex;
        }
      }

      this.filterbankBounds[melIndex * 2] = start >= 0 ? start : 0;
      this.filterbankBounds[melIndex * 2 + 1] = end >= 0 ? end + 1 : 0;
    }
  }

  process(audio: Float32Array): LasrCtcFeatureBatch {
    const { rawMel, frameCount, validFrameCount } = this.computeRawMel(audio);
    if (validFrameCount <= 0) {
      return {
        features: new Float32Array(0),
        frameCount: 0,
        featureSize: this.nMels,
      };
    }

    const features = this.normalizeFeatures
      ? this.normalize(rawMel, frameCount, validFrameCount)
      : this.copyWithoutNormalization(rawMel, frameCount, validFrameCount);

    return {
      features,
      frameCount: validFrameCount,
      featureSize: this.nMels,
    };
  }

  private computeRawMel(audio: Float32Array): RawMelOutput {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return {
        rawMel: new Float32Array(0),
        frameCount: 0,
        validFrameCount: 0,
      };
    }

    if (this.emphasizedBuffer.length < sampleCount) {
      this.emphasizedBuffer = new Float32Array(Math.max(this.emphasizedBuffer.length * 2, sampleCount));
    }
    const emphasized = this.emphasizedBuffer.subarray(0, sampleCount);

    emphasized[0] = audio[0] ?? 0;
    if (this.preemphasis > 0) {
      for (let index = 1; index < sampleCount; index += 1) {
        emphasized[index] = (audio[index] ?? 0) - this.preemphasis * (audio[index - 1] ?? 0);
      }
    } else {
      emphasized.set(audio);
    }

    const pad = this.center ? N_FFT >> 1 : 0;
    const paddedLength = sampleCount + pad * 2;

    if (this.paddedBuffer.length < paddedLength) {
      this.paddedBuffer = new Float64Array(Math.max(this.paddedBuffer.length * 2, paddedLength));
    } else {
      this.paddedBuffer.fill(0, 0, paddedLength); // Clear previously used buffer
    }
    const padded = this.paddedBuffer.subarray(0, paddedLength);

    for (let index = 0; index < sampleCount; index += 1) {
      padded[index + pad] = emphasized[index] ?? 0;
    }

    const frameCount = Math.floor((paddedLength - WIN_LENGTH) / HOP_LENGTH) + 1;
    const validFrameCount = this.center ? Math.floor(sampleCount / HOP_LENGTH) : frameCount;
    if (validFrameCount <= 0 || frameCount <= 0) {
      return {
        rawMel: new Float32Array(0),
        frameCount: 0,
        validFrameCount: 0,
      };
    }

    const rawMel = new Float32Array(this.nMels * frameCount);

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const frameOffset = frameIndex * HOP_LENGTH;
      for (let fftIndex = 0; fftIndex < N_FFT; fftIndex += 1) {
        const sourceIndex = frameOffset + fftIndex;
        const sample = sourceIndex < paddedLength ? (padded[sourceIndex] ?? 0) : 0;
        this.fftReal[fftIndex] = sample * (this.hannWindow[fftIndex] ?? 0);
        this.fftImaginary[fftIndex] = 0;
      }

      fft(this.fftReal, this.fftImaginary, N_FFT, this.fftTwiddles);

      for (let frequencyIndex = 0; frequencyIndex < N_FREQ_BINS; frequencyIndex += 1) {
        const realValue = this.fftReal[frequencyIndex] ?? 0;
        const imaginaryValue = this.fftImaginary[frequencyIndex] ?? 0;
        this.powerBuffer[frequencyIndex] = realValue * realValue + imaginaryValue * imaginaryValue;
      }

      for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
        const melOffset = melIndex * N_FREQ_BINS;
        const lower = this.filterbankBounds[melIndex * 2] ?? 0;
        const upper = this.filterbankBounds[melIndex * 2 + 1] ?? 0;

        let melValue = 0;
        for (let frequencyIndex = lower; frequencyIndex < upper; frequencyIndex += 1) {
          melValue +=
            (this.powerBuffer[frequencyIndex] ?? 0) *
            (this.melFilterbank[melOffset + frequencyIndex] ?? 0);
        }

        rawMel[melIndex * frameCount + frameIndex] =
          this.logZeroGuard === 1e-5
            ? Math.log(Math.max(melValue, 1e-5))
            : Math.log(melValue + this.logZeroGuard);
      }
    }

    return {
      rawMel,
      frameCount,
      validFrameCount,
    };
  }

  private copyWithoutNormalization(
    rawMel: Float32Array,
    frameCount: number,
    validFrameCount: number,
  ): Float32Array {
    const copied = new Float32Array(this.nMels * validFrameCount);
    for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
      const sourceBase = melIndex * frameCount;
      const destinationBase = melIndex * validFrameCount;
      for (let frameIndex = 0; frameIndex < validFrameCount; frameIndex += 1) {
        copied[destinationBase + frameIndex] = rawMel[sourceBase + frameIndex] ?? 0;
      }
    }

    return copied;
  }

  private normalize(
    rawMel: Float32Array,
    frameCount: number,
    validFrameCount: number,
  ): Float32Array {
    const normalized = new Float32Array(this.nMels * validFrameCount);

    for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
      const sourceBase = melIndex * frameCount;
      const destinationBase = melIndex * validFrameCount;

      let sum = 0;
      for (let frameIndex = 0; frameIndex < validFrameCount; frameIndex += 1) {
        sum += rawMel[sourceBase + frameIndex] ?? 0;
      }
      const mean = sum / validFrameCount;

      let varianceSum = 0;
      for (let frameIndex = 0; frameIndex < validFrameCount; frameIndex += 1) {
        const delta = (rawMel[sourceBase + frameIndex] ?? 0) - mean;
        varianceSum += delta * delta;
      }

      const inverseStdDev =
        validFrameCount > 1 ? 1 / (Math.sqrt(varianceSum / (validFrameCount - 1)) + 1e-5) : 0;

      for (let frameIndex = 0; frameIndex < validFrameCount; frameIndex += 1) {
        normalized[destinationBase + frameIndex] =
          ((rawMel[sourceBase + frameIndex] ?? 0) - mean) * inverseStdDev;
      }
    }

    return normalized;
  }
}
