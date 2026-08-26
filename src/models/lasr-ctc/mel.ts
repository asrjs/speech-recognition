import type { LasrCtcFeatureBatch, LasrCtcFeaturePreprocessor } from './types.js';

const SAMPLE_RATE = 16000;
const DEFAULT_N_FFT = 512;
const DEFAULT_WIN_LENGTH = 400;
const DEFAULT_HOP_LENGTH = 160;
const PREEMPH = 0.97;
const LOG_ZERO_GUARD = 2 ** -24;

type MelScaleKind = 'slaney' | 'kaldi' | 'htk';
type WindowKind = 'hann' | 'hann-periodic' | 'hamming';

interface MelTwiddles {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly bitReverse: Uint32Array;
}

export interface MedAsrJsPreprocessorOptions {
  readonly nFft?: number;
  readonly winLength?: number;
  readonly hopLength?: number;
  readonly nMels?: number;
  readonly center?: boolean;
  readonly preemphasis?: number;
  readonly melScale?: MelScaleKind;
  readonly slaneyNorm?: boolean;
  readonly logZeroGuard?: number;
  readonly normalizeFeatures?: boolean;
  /** Window used for each frame. Existing callers retain the symmetric Hann default. */
  readonly windowKind?: WindowKind;
  /** Remove each frame's DC offset before optional frame-local preemphasis. */
  readonly removeDcOffset?: boolean;
  /** Apply preemphasis independently inside every frame (Kaldi/Wespeaker). */
  readonly framePreemphasis?: boolean;
  /** Kaldi mel lower edge in Hz; existing callers retain 125 Hz. */
  readonly melLowHz?: number;
  /** Kaldi mel upper edge in Hz; negative values are relative to Nyquist. */
  readonly melHighHz?: number;
}

const MEL_FILTERBANK_CACHE = new Map<string, Float32Array>();
const FFT_TWIDDLE_CACHE = new Map<number, MelTwiddles>();
const WINDOW_CACHE = new Map<string, Float64Array>();
const INVERSE_FFT_TWIDDLE_CACHE = new Map<number, MelTwiddles>();

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

function hzToMelHtk(frequencyHz: number): number {
  return 2595 * Math.log10(1 + frequencyHz / 700);
}

function melToHzHtk(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function createMelFilterbank(
  nMels: number,
  melScale: MelScaleKind,
  slaneyNorm: boolean,
  nFreqBins: number,
  melLowHz?: number,
  melHighHz?: number,
): Float32Array {
  const frequencyMin = melScale === 'kaldi' ? melLowHz ?? 125 : 0;
  const requestedMax = melHighHz ?? 7500;
  const frequencyMax =
    melScale === 'kaldi'
      ? requestedMax < 0
        ? SAMPLE_RATE / 2 + requestedMax
        : requestedMax
      : SAMPLE_RATE / 2;

  const toMel =
    melScale === 'kaldi'
      ? (frequency: number): number => 1127 * Math.log(1 + frequency / 700)
      : melScale === 'htk'
        ? hzToMelHtk
        : hzToMel;
  const toHz =
    melScale === 'kaldi'
      ? (mel: number): number => 700 * (Math.exp(mel / 1127) - 1)
      : melScale === 'htk'
        ? melToHzHtk
        : melToHz;

  const allFrequencies = new Float64Array(nFreqBins);
  for (let index = 0; index < nFreqBins; index += 1) {
    allFrequencies[index] = ((SAMPLE_RATE / 2) * index) / (nFreqBins - 1);
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

  const filterbank = new Float32Array(nMels * nFreqBins);
  for (let melIndex = 0; melIndex < nMels; melIndex += 1) {
    const centerLeft = melFrequencies[melIndex] ?? 0;
    const centerRight = melFrequencies[melIndex + 2] ?? centerLeft;
    const lowerDelta = melDifferences[melIndex] ?? 1;
    const upperDelta = melDifferences[melIndex + 1] ?? 1;
    const normalization = slaneyNorm ? 2 / Math.max(1e-12, centerRight - centerLeft) : 1;
    const rowOffset = melIndex * nFreqBins;

    for (let frequencyIndex = 0; frequencyIndex < nFreqBins; frequencyIndex += 1) {
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
  nFreqBins: number,
  melLowHz?: number,
  melHighHz?: number,
): Float32Array {
  const cacheKey = `${nMels}:${melScale}:${slaneyNorm}:${nFreqBins}:${melLowHz ?? ''}:${melHighHz ?? ''}`;
  const cached = MEL_FILTERBANK_CACHE.get(cacheKey);
  if (cached) {
    return cached;
  }

  const created = createMelFilterbank(nMels, melScale, slaneyNorm, nFreqBins, melLowHz, melHighHz);
  MEL_FILTERBANK_CACHE.set(cacheKey, created);
  return created;
}

function createPaddedWindow(
  centerWindow: boolean,
  windowKind: WindowKind,
  nFft: number,
  winLength: number,
): Float64Array {
  const window = new Float64Array(nFft);
  const leftPad = centerWindow ? (nFft - winLength) >> 1 : 0;

  for (let index = 0; index < winLength; index += 1) {
    const cosine = Math.cos((2 * Math.PI * index) / (windowKind === 'hann-periodic' ? winLength : winLength - 1));
    window[leftPad + index] = windowKind === 'hamming' ? 0.54 - 0.46 * cosine : 0.5 * (1 - cosine);
  }

  return window;
}

function getCachedWindow(centerWindow: boolean, windowKind: WindowKind, nFft: number, winLength: number): Float64Array {
  const key = `${centerWindow ? 'center' : 'left'}:${windowKind}:${nFft}:${winLength}`;
  const cached = WINDOW_CACHE.get(key);
  if (cached) {
    return cached;
  }

  const created = createPaddedWindow(centerWindow, windowKind, nFft, winLength);
  WINDOW_CACHE.set(key, created);
  return created;
}

function assertPowerOfTwo(size: number): void {
  if (!Number.isInteger(size) || size < 1 || (size & (size - 1)) !== 0) {
    throw new RangeError(`Radix-2 FFT requires a power-of-two size. Received ${size}.`);
  }
}

function createBitReverseTable(size: number): Uint32Array {
  const bits = Math.log2(size);
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
  return bitReverse;
}

function createFftTwiddles(size: number, inverse: boolean): MelTwiddles {
  const half = size >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let index = 0; index < half; index += 1) {
    const angle = ((inverse ? 2 : -2) * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }
  return { cos, sin, bitReverse: createBitReverseTable(size) };
}

function precomputeFftTwiddles(size: number): MelTwiddles {
  const cached = FFT_TWIDDLE_CACHE.get(size);
  if (cached) {
    return cached;
  }

  assertPowerOfTwo(size);
  const twiddles = createFftTwiddles(size, false);
  FFT_TWIDDLE_CACHE.set(size, twiddles);
  return twiddles;
}

function precomputeInverseFftTwiddles(size: number): MelTwiddles {
  const cached = INVERSE_FFT_TWIDDLE_CACHE.get(size);
  if (cached) {
    return cached;
  }

  assertPowerOfTwo(size);
  const twiddles = createFftTwiddles(size, true);
  INVERSE_FFT_TWIDDLE_CACHE.set(size, twiddles);
  return twiddles;
}

function fft(
  real: Float64Array,
  imaginary: Float64Array,
  size: number,
  twiddles: MelTwiddles,
): void {
  assertPowerOfTwo(size);

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

function nextPowerOfTwo(size: number): number {
  let power = 1;
  while (power < size) {
    power <<= 1;
  }
  return power;
}

/**
 * Bluestein (chirp-z) FFT for arbitrary, non-power-of-two transform sizes.
 *
 * The radix-2 loop above only accepts power-of-two sizes, while model
 * geometries such as GigaAM's n_fft=320 do not. A naive O(N^2) DFT is exact
 * but becomes the frontend bottleneck for long-form audio, so this class
 * expresses the N-point DFT as a cyclic convolution of length
 * nextPowerOfTwo(2N - 1) and reuses the fast radix-2 kernel. The result is a
 * full N-bin complex spectrum, matching the previous direct-DFT fallback.
 */
export class CompositeFft {
  private readonly fftSize: number;
  private readonly forwardTwiddles: MelTwiddles;
  private readonly inverseTwiddles: MelTwiddles;
  private readonly chirpCos: Float64Array;
  private readonly chirpSin: Float64Array;
  private readonly kernelReal: Float64Array;
  private readonly kernelImaginary: Float64Array;
  private readonly workReal: Float64Array;
  private readonly workImaginary: Float64Array;

  constructor(readonly size: number) {
    this.fftSize = nextPowerOfTwo(size * 2 - 1);
    this.forwardTwiddles = precomputeFftTwiddles(this.fftSize);
    this.inverseTwiddles = precomputeInverseFftTwiddles(this.fftSize);
    this.chirpCos = new Float64Array(size);
    this.chirpSin = new Float64Array(size);
    this.kernelReal = new Float64Array(this.fftSize);
    this.kernelImaginary = new Float64Array(this.fftSize);
    this.workReal = new Float64Array(this.fftSize);
    this.workImaginary = new Float64Array(this.fftSize);

    this.kernelReal[0] = 1;
    for (let index = 0; index < size; index += 1) {
      const angle = (Math.PI * index * index) / size;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      this.chirpCos[index] = cosine;
      this.chirpSin[index] = sine;
      if (index > 0) {
        this.kernelReal[index] = cosine;
        this.kernelImaginary[index] = sine;
        this.kernelReal[this.fftSize - index] = cosine;
        this.kernelImaginary[this.fftSize - index] = sine;
      }
    }

    fft(this.kernelReal, this.kernelImaginary, this.fftSize, this.forwardTwiddles);
  }

  /** In-place N-point complex DFT over the first `size` entries of the buffers. */
  transform(real: Float64Array, imaginary: Float64Array): void {
    this.workReal.fill(0);
    this.workImaginary.fill(0);

    // Multiply the input by the chirp exp(+i*pi*n^2/N) and zero-pad to fftSize.
    for (let index = 0; index < this.size; index += 1) {
      const realValue = real[index] ?? 0;
      const imaginaryValue = imaginary[index] ?? 0;
      const cosine = this.chirpCos[index] ?? 0;
      const sine = this.chirpSin[index] ?? 0;
      this.workReal[index] = realValue * cosine + imaginaryValue * sine;
      this.workImaginary[index] = imaginaryValue * cosine - realValue * sine;
    }

    fft(this.workReal, this.workImaginary, this.fftSize, this.forwardTwiddles);

    for (let index = 0; index < this.fftSize; index += 1) {
      const leftReal = this.workReal[index] ?? 0;
      const leftImaginary = this.workImaginary[index] ?? 0;
      const rightReal = this.kernelReal[index] ?? 0;
      const rightImaginary = this.kernelImaginary[index] ?? 0;
      this.workReal[index] = leftReal * rightReal - leftImaginary * rightImaginary;
      this.workImaginary[index] = leftReal * rightImaginary + leftImaginary * rightReal;
    }

    fft(this.workReal, this.workImaginary, this.fftSize, this.inverseTwiddles);

    // The inverse radix-2 pass is unnormalized, so divide by fftSize while
    // demodulating with the conjugate chirp exp(-i*pi*k^2/N).
    const scale = 1 / this.fftSize;
    for (let bin = 0; bin < this.size; bin += 1) {
      const convolutionReal = (this.workReal[bin] ?? 0) * scale;
      const convolutionImaginary = (this.workImaginary[bin] ?? 0) * scale;
      const cosine = this.chirpCos[bin] ?? 0;
      const sine = this.chirpSin[bin] ?? 0;
      real[bin] = convolutionReal * cosine + convolutionImaginary * sine;
      imaginary[bin] = convolutionImaginary * cosine - convolutionReal * sine;
    }
  }
}

export function transposeMelToTxM(
  featuresMxT: Float32Array,
  nMels: number,
  frameCount: number,
  output?: Float32Array,
): Float32Array {
  const size = frameCount * nMels;
  const transposed = !output || output.length < size ? new Float32Array(size) : output;

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    const frameOffset = frameIndex * nMels;
    for (let melIndex = 0; melIndex < nMels; melIndex += 1) {
      transposed[frameOffset + melIndex] = featuresMxT[melIndex * frameCount + frameIndex] ?? 0;
    }
  }

  return transposed.length === size ? transposed : transposed.subarray(0, size);
}

interface RawMelOutput {
  readonly rawMel: Float32Array;
  readonly frameCount: number;
  readonly validFrameCount: number;
}

export class MedAsrJsPreprocessor implements LasrCtcFeaturePreprocessor {
  readonly nMels: number;
  readonly nFft: number;
  readonly winLength: number;
  readonly hopLength: number;
  private readonly nFreqBins: number;
  private readonly center: boolean;
  private readonly preemphasis: number;
  private readonly melScale: MelScaleKind;
  private readonly slaneyNorm: boolean;
  private readonly logZeroGuard: number;
  private readonly normalizeFeatures: boolean;
  private readonly windowKind: WindowKind;
  private readonly removeDcOffset: boolean;
  private readonly framePreemphasis: boolean;
  private readonly melLowHz?: number;
  private readonly melHighHz?: number;
  private readonly melFilterbank: Float32Array;
  private readonly hannWindow: Float64Array;
  private readonly fftTwiddles: MelTwiddles | null;
  private readonly compositeFft: CompositeFft | null;
  private readonly fftReal: Float64Array;
  private readonly fftImaginary: Float64Array;
  private readonly powerBuffer: Float32Array;
  private readonly filterbankBounds: Int32Array;
  private emphasizedBuffer: Float32Array | null = null;
  private paddedBuffer: Float64Array | null = null;
  private rawMelBuffer: Float32Array | null = null;

  constructor(options: MedAsrJsPreprocessorOptions = {}) {
    this.nMels = options.nMels ?? 128;
    this.nFft = options.nFft ?? DEFAULT_N_FFT;
    this.winLength = options.winLength ?? DEFAULT_WIN_LENGTH;
    this.hopLength = options.hopLength ?? DEFAULT_HOP_LENGTH;
    if (!Number.isInteger(this.nFft) || this.nFft <= 0) {
      throw new RangeError(`nFft must be positive. Received ${this.nFft}.`);
    }
    if (!Number.isInteger(this.winLength) || this.winLength <= 0 || this.winLength > this.nFft) {
      throw new RangeError(`winLength must be between 1 and nFft. Received ${this.winLength}.`);
    }
    if (!Number.isInteger(this.hopLength) || this.hopLength <= 0) {
      throw new RangeError(`hopLength must be positive. Received ${this.hopLength}.`);
    }
    this.nFreqBins = (this.nFft >> 1) + 1;
    this.center = options.center ?? false;
    this.preemphasis = options.preemphasis ?? PREEMPH;
    this.melScale = options.melScale ?? 'kaldi';
    this.slaneyNorm = options.slaneyNorm ?? false;
    this.logZeroGuard = options.logZeroGuard ?? LOG_ZERO_GUARD;
    this.normalizeFeatures = options.normalizeFeatures ?? false;
    this.windowKind = options.windowKind ?? 'hann';
    this.removeDcOffset = options.removeDcOffset ?? false;
    this.framePreemphasis = options.framePreemphasis ?? false;
    this.melLowHz = options.melLowHz;
    this.melHighHz = options.melHighHz;

    this.melFilterbank = getCachedMelFilterbank(
      this.nMels,
      this.melScale,
      this.slaneyNorm,
      this.nFreqBins,
      this.melLowHz,
      this.melHighHz,
    );
    this.hannWindow = getCachedWindow(this.center, this.windowKind, this.nFft, this.winLength);
    const isPowerOfTwo = (this.nFft & (this.nFft - 1)) === 0;
    this.fftTwiddles = isPowerOfTwo ? precomputeFftTwiddles(this.nFft) : null;
    this.compositeFft = isPowerOfTwo ? null : new CompositeFft(this.nFft);
    this.fftReal = new Float64Array(this.nFft);
    this.fftImaginary = new Float64Array(this.nFft);
    this.powerBuffer = new Float32Array(this.nFreqBins);

    this.filterbankBounds = new Int32Array(this.nMels * 2);
    for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
      const offset = melIndex * this.nFreqBins;
      let start = -1;
      let end = -1;
      for (let frequencyIndex = 0; frequencyIndex < this.nFreqBins; frequencyIndex += 1) {
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

    if (!this.emphasizedBuffer || this.emphasizedBuffer.length < sampleCount) {
      this.emphasizedBuffer = new Float32Array(Math.ceil(sampleCount * 1.2));
    }
    const emphasized = this.emphasizedBuffer;
    emphasized[0] = audio[0] ?? 0;
    if (this.preemphasis > 0 && !this.framePreemphasis) {
      for (let index = 1; index < sampleCount; index += 1) {
        emphasized[index] = (audio[index] ?? 0) - this.preemphasis * (audio[index - 1] ?? 0);
      }
    } else {
      emphasized.set(audio);
    }

    const pad = this.center ? this.nFft >> 1 : 0;
    const paddedLength = sampleCount + pad * 2;
    if (!this.paddedBuffer || this.paddedBuffer.length < paddedLength) {
      this.paddedBuffer = new Float64Array(Math.ceil(paddedLength * 1.2));
    }
    const padded = this.paddedBuffer;
    padded.fill(0, 0, paddedLength);
    for (let index = 0; index < sampleCount; index += 1) {
      padded[index + pad] = emphasized[index] ?? 0;
    }

    const frameCount = Math.floor((paddedLength - this.winLength) / this.hopLength) + 1;
    const validFrameCount = this.center ? Math.floor(sampleCount / this.hopLength) : frameCount;
    if (validFrameCount <= 0 || frameCount <= 0) {
      return {
        rawMel: new Float32Array(0),
        frameCount: 0,
        validFrameCount: 0,
      };
    }

    const requiredRawMelSize = this.nMels * frameCount;
    if (!this.rawMelBuffer || this.rawMelBuffer.length < requiredRawMelSize) {
      this.rawMelBuffer = new Float32Array(Math.ceil(requiredRawMelSize * 1.2));
    }
    const rawMel = this.rawMelBuffer.subarray(0, requiredRawMelSize);

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
      const frameOffset = frameIndex * this.hopLength;
      let frameMean = 0;
      if (this.removeDcOffset) {
        for (let fftIndex = 0; fftIndex < this.winLength; fftIndex += 1) {
          const sourceIndex = frameOffset + fftIndex;
          frameMean += sourceIndex < paddedLength ? (padded[sourceIndex] ?? 0) : 0;
        }
        frameMean /= this.winLength;
      }
      let previousSample = 0;
      for (let fftIndex = 0; fftIndex < this.nFft; fftIndex += 1) {
        const sourceIndex = frameOffset + fftIndex;
        let sample = sourceIndex < paddedLength ? (padded[sourceIndex] ?? 0) : 0;
        if (fftIndex < this.winLength && this.removeDcOffset) sample -= frameMean;
        if (fftIndex < this.winLength && this.framePreemphasis && this.preemphasis > 0) {
          sample -= this.preemphasis * (fftIndex === 0 ? sample : previousSample);
          previousSample = sourceIndex < paddedLength ? (padded[sourceIndex] ?? 0) - frameMean : -frameMean;
        }
        this.fftReal[fftIndex] = sample * (this.hannWindow[fftIndex] ?? 0);
        this.fftImaginary[fftIndex] = 0;
      }

      if (this.compositeFft) {
        this.compositeFft.transform(this.fftReal, this.fftImaginary);
      } else {
        fft(this.fftReal, this.fftImaginary, this.nFft, this.fftTwiddles!);
      }

      for (let frequencyIndex = 0; frequencyIndex < this.nFreqBins; frequencyIndex += 1) {
        const realValue = this.fftReal[frequencyIndex] ?? 0;
        const imaginaryValue = this.fftImaginary[frequencyIndex] ?? 0;
        this.powerBuffer[frequencyIndex] = realValue * realValue + imaginaryValue * imaginaryValue;
      }

      for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
        const melOffset = melIndex * this.nFreqBins;
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
