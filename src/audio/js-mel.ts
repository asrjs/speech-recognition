/**
 * Shared NeMo-compatible JavaScript mel frontend.
 *
 * Ported from the proven `parakeet.js` / `meljs` implementations so multiple
 * NeMo model families in this repo can reuse one optimized browser-safe
 * frontend instead of shipping per-model ONNX preprocessors.
 */

export interface JsNemoMelProcessorOptions {
  readonly nMels?: number;
  readonly boundaryFrames?: number;
  readonly validLengthMode?: 'onnx' | 'centered';
  readonly normalization?: 'per_feature' | 'none';
}

export interface JsNemoMelProcessResult {
  readonly features: Float32Array;
  readonly frameCount: number;
  readonly length: number;
}

export interface IncrementalJsNemoMelProcessResult extends JsNemoMelProcessResult {
  readonly cached: boolean;
  readonly cachedFrames: number;
  readonly newFrames: number;
}

export const MEL_CONSTANTS = Object.freeze({
  SAMPLE_RATE: 16000,
  N_FFT: 512,
  WIN_LENGTH: 400,
  HOP_LENGTH: 160,
  PREEMPH: 0.97,
  LOG_ZERO_GUARD: 2 ** -24,
  N_FREQ_BINS: (512 >> 1) + 1,
});

const SAMPLE_RATE = MEL_CONSTANTS.SAMPLE_RATE;
const N_FFT = MEL_CONSTANTS.N_FFT;
const WIN_LENGTH = MEL_CONSTANTS.WIN_LENGTH;
const HOP_LENGTH = MEL_CONSTANTS.HOP_LENGTH;
const PREEMPH = MEL_CONSTANTS.PREEMPH;
const LOG_ZERO_GUARD = MEL_CONSTANTS.LOG_ZERO_GUARD;
const N_FREQ_BINS = MEL_CONSTANTS.N_FREQ_BINS;
const INV_SQRT2 = Math.SQRT1_2;

const F_SP = 200.0 / 3;
const MIN_LOG_HZ = 1000.0;
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;
const LOG_STEP = Math.log(6.4) / 27.0;

const MEL_FILTERBANK_CACHE = new Map<number, Float32Array>();
const FFT_TWIDDLE_CACHE = new Map<number, FFTTwiddles>();
let SHARED_HANN_WINDOW: Float64Array | null = null;

interface FFTTwiddles {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly bitrev: Uint32Array;
}

export function hzToMel(freq: number): number {
  return freq >= MIN_LOG_HZ ? MIN_LOG_MEL + Math.log(freq / MIN_LOG_HZ) / LOG_STEP : freq / F_SP;
}

export function melToHz(mel: number): number {
  return mel >= MIN_LOG_MEL ? MIN_LOG_HZ * Math.exp(LOG_STEP * (mel - MIN_LOG_MEL)) : mel * F_SP;
}

export function createMelFilterbank(nMels = 128): Float32Array {
  const fMax = SAMPLE_RATE / 2;
  const allFreqs = new Float64Array(N_FREQ_BINS);
  for (let index = 0; index < N_FREQ_BINS; index += 1) {
    allFreqs[index] = (fMax * index) / (N_FREQ_BINS - 1);
  }

  const melMin = hzToMel(0);
  const melMax = hzToMel(fMax);
  const nPoints = nMels + 2;
  const fPts = new Float64Array(nPoints);
  for (let index = 0; index < nPoints; index += 1) {
    fPts[index] = melToHz(melMin + ((melMax - melMin) * index) / (nPoints - 1));
  }

  const fDiff = new Float64Array(nPoints - 1);
  for (let index = 0; index < nPoints - 1; index += 1) {
    fDiff[index] = fPts[index + 1]! - fPts[index]!;
  }

  const filterbank = new Float32Array(nMels * N_FREQ_BINS);
  for (let melIndex = 0; melIndex < nMels; melIndex += 1) {
    const lower = fPts[melIndex]!;
    const upper = fPts[melIndex + 2]!;
    const lowerWidth = fDiff[melIndex]!;
    const upperWidth = fDiff[melIndex + 1]!;
    const enorm = 2.0 / (upper - lower);
    const offset = melIndex * N_FREQ_BINS;
    for (let freqIndex = 0; freqIndex < N_FREQ_BINS; freqIndex += 1) {
      const freq = allFreqs[freqIndex]!;
      const downSlope = (freq - lower) / lowerWidth;
      const upSlope = (upper - freq) / upperWidth;
      filterbank[offset + freqIndex] = Math.max(0, Math.min(downSlope, upSlope)) * enorm;
    }
  }

  return filterbank;
}

function getCachedMelFilterbank(nMels: number): Float32Array {
  let filterbank = MEL_FILTERBANK_CACHE.get(nMels);
  if (!filterbank) {
    filterbank = createMelFilterbank(nMels);
    MEL_FILTERBANK_CACHE.set(nMels, filterbank);
  }
  return filterbank;
}

export function createPaddedHannWindow(): Float64Array {
  const window = new Float64Array(N_FFT);
  const padLeft = (N_FFT - WIN_LENGTH) >> 1;
  for (let index = 0; index < WIN_LENGTH; index += 1) {
    window[padLeft + index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / (WIN_LENGTH - 1)));
  }
  return window;
}

function getCachedPaddedHannWindow(): Float64Array {
  if (!SHARED_HANN_WINDOW) {
    SHARED_HANN_WINDOW = createPaddedHannWindow();
  }
  return SHARED_HANN_WINDOW;
}

export function precomputeTwiddles(size: number): FFTTwiddles {
  const cached = FFT_TWIDDLE_CACHE.get(size);
  if (cached) {
    return cached;
  }

  const bits = Math.log2(size);
  if (1 << bits !== size) {
    throw new Error(`FFT size must be a power of two. Received: ${size}`);
  }

  const half = size >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let index = 0; index < half; index += 1) {
    const angle = (-2 * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }

  const bitrev = new Uint32Array(size);
  for (let index = 0; index < size; index += 1) {
    let value = index;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    bitrev[index] = reversed;
  }

  const twiddles: FFTTwiddles = { cos, sin, bitrev };
  FFT_TWIDDLE_CACHE.set(size, twiddles);
  return twiddles;
}

export function fft(re: Float64Array, im: Float64Array, size: number, twiddles: FFTTwiddles): void {
  const bitrev = twiddles.bitrev;
  if (bitrev.length === size) {
    for (let index = 0; index < size; index += 1) {
      const reversed = bitrev[index]!;
      if (index < reversed) {
        let tmp = re[index]!;
        re[index] = re[reversed]!;
        re[reversed] = tmp;
        tmp = im[index]!;
        im[index] = im[reversed]!;
        im[reversed] = tmp;
      }
    }
  }

  if (size >= 2) {
    for (let index = 0; index < size; index += 2) {
      const q = index + 1;
      const tRe = re[q]!;
      const tIm = im[q]!;
      re[q] = re[index]! - tRe;
      im[q] = im[index]! - tIm;
      re[index] = re[index]! + tRe;
      im[index] = im[index]! + tIm;
    }
  }

  if (size >= 4) {
    for (let index = 0; index < size; index += 4) {
      const q0 = index + 2;
      const tRe0 = re[q0]!;
      const tIm0 = im[q0]!;
      re[q0] = re[index]! - tRe0;
      im[q0] = im[index]! - tIm0;
      re[index] = re[index]! + tRe0;
      im[index] = im[index]! + tIm0;

      const p1 = index + 1;
      const q1 = index + 3;
      const tRe1 = im[q1]!;
      const tIm1 = -re[q1]!;
      re[q1] = re[p1]! - tRe1;
      im[q1] = im[p1]! - tIm1;
      re[p1] = re[p1]! + tRe1;
      im[p1] = im[p1]! + tIm1;
    }
  }

  if (size >= 8) {
    for (let index = 0; index < size; index += 8) {
      {
        const q = index + 4;
        const tRe = re[q]!;
        const tIm = im[q]!;
        re[q] = re[index]! - tRe;
        im[q] = im[index]! - tIm;
        re[index] = re[index]! + tRe;
        im[index] = im[index]! + tIm;
      }
      {
        const wCos = INV_SQRT2;
        const wSin = -INV_SQRT2;
        const p = index + 1;
        const q = index + 5;
        const tRe = re[q]! * wCos - im[q]! * wSin;
        const tIm = re[q]! * wSin + im[q]! * wCos;
        re[q] = re[p]! - tRe;
        im[q] = im[p]! - tIm;
        re[p] = re[p]! + tRe;
        im[p] = im[p]! + tIm;
      }
      {
        const p = index + 2;
        const q = index + 6;
        const tRe = im[q]!;
        const tIm = -re[q]!;
        re[q] = re[p]! - tRe;
        im[q] = im[p]! - tIm;
        re[p] = re[p]! + tRe;
        im[p] = im[p]! + tIm;
      }
      {
        const wCos = -INV_SQRT2;
        const wSin = -INV_SQRT2;
        const p = index + 3;
        const q = index + 7;
        const tRe = re[q]! * wCos - im[q]! * wSin;
        const tIm = re[q]! * wSin + im[q]! * wCos;
        re[q] = re[p]! - tRe;
        im[q] = im[p]! - tIm;
        re[p] = re[p]! + tRe;
        im[p] = im[p]! + tIm;
      }
    }
  }

  for (let len = 16; len <= size; len <<= 1) {
    const halfLen = len >> 1;
    const step = size / len;
    for (let index = 0; index < size; index += len) {
      for (let k = 0; k < halfLen; k += 1) {
        const twiddleIndex = k * step;
        const wCos = twiddles.cos[twiddleIndex]!;
        const wSin = twiddles.sin[twiddleIndex]!;
        const p = index + k;
        const q = p + halfLen;
        const tRe = re[q]! * wCos - im[q]! * wSin;
        const tIm = re[q]! * wSin + im[q]! * wCos;
        re[q] = re[p]! - tRe;
        im[q] = im[p]! - tIm;
        re[p] = re[p]! + tRe;
        im[p] = im[p]! + tIm;
      }
    }
  }
}

export class JSMelProcessor {
  readonly nMels: number;
  readonly validLengthMode: 'onnx' | 'centered';
  readonly normalization: 'per_feature' | 'none';
  private readonly melFilterbank: Float32Array;
  private readonly hannWindow: Float64Array;
  private readonly twiddles: FFTTwiddles;
  private readonly twiddlesHalf: FFTTwiddles;
  private readonly fftRe: Float64Array;
  private readonly fftIm: Float64Array;
  private readonly powerBuf: Float32Array;
  private paddedBuffer: Float64Array | null = null;
  private processRawBuffer: Float32Array | null = null;
  private readonly fbBounds: Int32Array;

  constructor(options: JsNemoMelProcessorOptions = {}) {
    this.nMels = options.nMels ?? 128;
    this.validLengthMode = options.validLengthMode ?? 'onnx';
    this.normalization = options.normalization ?? 'per_feature';
    this.melFilterbank = getCachedMelFilterbank(this.nMels);
    this.hannWindow = getCachedPaddedHannWindow();
    this.twiddles = precomputeTwiddles(N_FFT);
    this.twiddlesHalf = precomputeTwiddles(N_FFT >> 1);
    this.fftRe = new Float64Array(N_FFT >> 1);
    this.fftIm = new Float64Array(N_FFT >> 1);
    this.powerBuf = new Float32Array(N_FREQ_BINS);
    this.fbBounds = new Int32Array(this.nMels * 2);

    for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
      const offset = melIndex * N_FREQ_BINS;
      let start = -1;
      let end = -1;
      for (let freqIndex = 0; freqIndex < N_FREQ_BINS; freqIndex += 1) {
        if (this.melFilterbank[offset + freqIndex]! > 0) {
          if (start === -1) {
            start = freqIndex;
          }
          end = freqIndex;
        }
      }
      this.fbBounds[melIndex * 2] = start === -1 ? 0 : start;
      this.fbBounds[melIndex * 2 + 1] = end + 1;
    }
  }

  process(audio: Float32Array): JsNemoMelProcessResult {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return { features: new Float32Array(0), frameCount: 0, length: 0 };
    }

    const pad = N_FFT >> 1;
    const paddedLen = sampleCount + 2 * pad;
    const nFrames = Math.floor((paddedLen - N_FFT) / HOP_LENGTH) + 1;
    const validLength =
      this.validLengthMode === 'centered' ? nFrames : Math.floor(sampleCount / HOP_LENGTH);
    if (validLength === 0) {
      return { features: new Float32Array(0), frameCount: 0, length: 0 };
    }

    const requiredRawSize = this.nMels * nFrames;
    if (!this.processRawBuffer || this.processRawBuffer.length < requiredRawSize) {
      this.processRawBuffer = new Float32Array(Math.ceil(requiredRawSize * 1.2));
    }

    const {
      rawMel,
      nFrames: computedNFrames,
      validLength: computedValidLength,
    } = this.computeRawMel(audio, 0, this.processRawBuffer);

    return {
      features: this.finalizeFeatures(rawMel, computedNFrames, computedValidLength),
      frameCount: computedNFrames,
      length: computedValidLength,
    };
  }

  computeRawMel(
    audio: Float32Array,
    startFrame = 0,
    outBuffer: Float32Array | null = null,
  ): { rawMel: Float32Array; nFrames: number; validLength: number } {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return {
        rawMel: outBuffer ? outBuffer.subarray(0, 0) : new Float32Array(0),
        nFrames: 0,
        validLength: 0,
      };
    }

    const pad = N_FFT >> 1;
    const paddedLen = sampleCount + 2 * pad;
    let paddedReallocated = false;
    if (!this.paddedBuffer || this.paddedBuffer.length < paddedLen) {
      this.paddedBuffer = new Float64Array(Math.ceil(paddedLen * 1.2));
      paddedReallocated = true;
    }
    const padded = this.paddedBuffer;

    padded[pad] = Math.fround(audio[0]!);
    for (let index = 1; index < sampleCount; index += 1) {
      padded[pad + index] = Math.fround(audio[index]! - PREEMPH * audio[index - 1]!);
    }
    if (!paddedReallocated) {
      padded.fill(0, pad + sampleCount, paddedLen);
    }

    const nFrames = Math.floor((paddedLen - N_FFT) / HOP_LENGTH) + 1;
    const validLength =
      this.validLengthMode === 'centered' ? nFrames : Math.floor(sampleCount / HOP_LENGTH);
    if (validLength === 0) {
      return { rawMel: new Float32Array(0), nFrames: 0, validLength: 0 };
    }

    const requiredSize = this.nMels * nFrames;
    let rawMel: Float32Array;
    if (outBuffer && outBuffer.length >= requiredSize) {
      rawMel = outBuffer.subarray(0, requiredSize);
      if (startFrame > 0) {
        for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
          rawMel.fill(0, melIndex * nFrames, melIndex * nFrames + startFrame);
        }
      }
    } else {
      rawMel = new Float32Array(requiredSize);
    }

    const halfN = N_FFT >> 1;
    const quarterN = halfN >> 1;
    for (let frameIndex = startFrame; frameIndex < nFrames; frameIndex += 1) {
      const offset = frameIndex * HOP_LENGTH;
      for (let k = 0; k < halfN; k += 1) {
        const sampleIndex = k << 1;
        this.fftRe[k] = padded[offset + sampleIndex]! * this.hannWindow[sampleIndex]!;
        this.fftIm[k] = padded[offset + sampleIndex + 1]! * this.hannWindow[sampleIndex + 1]!;
      }

      fft(this.fftRe, this.fftIm, halfN, this.twiddlesHalf);

      const z0r = this.fftRe[0]!;
      const z0i = this.fftIm[0]!;
      this.powerBuf[0] = (z0r + z0i) * (z0r + z0i);
      this.powerBuf[halfN] = (z0r - z0i) * (z0r - z0i);

      for (let k = 1; k < quarterN; k += 1) {
        const rk = this.fftRe[k]!;
        const ik = this.fftIm[k]!;
        const rnk = this.fftRe[halfN - k]!;
        const ink = this.fftIm[halfN - k]!;

        const xeR = 0.5 * (rk + rnk);
        const xeI = 0.5 * (ik - ink);
        const xoR = 0.5 * (ik + ink);
        const xoI = -0.5 * (rk - rnk);

        const wc = this.twiddles.cos[k]!;
        const ws = this.twiddles.sin[k]!;
        const tr = xoR * wc - xoI * ws;
        const ti = xoR * ws + xoI * wc;

        const xkR = xeR + tr;
        const xkI = xeI + ti;
        this.powerBuf[k] = xkR * xkR + xkI * xkI;

        const xnkR = xeR - tr;
        const xnkI = xeI - ti;
        this.powerBuf[halfN - k] = xnkR * xnkR + xnkI * xnkI;
      }

      const quarterRe = this.fftRe[quarterN]!;
      const quarterIm = this.fftIm[quarterN]!;
      this.powerBuf[quarterN] = quarterRe * quarterRe + quarterIm * quarterIm;

      for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
        let melValue = 0;
        const filterbankOffset = melIndex * N_FREQ_BINS;
        const start = this.fbBounds[melIndex * 2]!;
        const end = this.fbBounds[melIndex * 2 + 1]!;
        for (let freqIndex = start; freqIndex < end; freqIndex += 1) {
          melValue += this.powerBuf[freqIndex]! * this.melFilterbank[filterbankOffset + freqIndex]!;
        }
        rawMel[melIndex * nFrames + frameIndex] = Math.log(melValue + LOG_ZERO_GUARD);
      }
    }

    return { rawMel, nFrames, validLength };
  }

  normalizeFeatures(
    rawMel: Float32Array,
    nFrames: number,
    validLength: number,
    outBuffer: Float32Array | null = null,
  ): Float32Array {
    const outputFrames = nFrames;
    const requiredSize = this.nMels * outputFrames;
    const features =
      outBuffer && outBuffer.length >= requiredSize
        ? outBuffer.subarray(0, requiredSize)
        : new Float32Array(requiredSize);

    for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
      const srcBase = melIndex * nFrames;
      const dstBase = melIndex * outputFrames;

      let sum = 0;
      for (let frameIndex = 0; frameIndex < validLength; frameIndex += 1) {
        sum += rawMel[srcBase + frameIndex]!;
      }
      const mean = sum / validLength;

      let varianceSum = 0;
      for (let frameIndex = 0; frameIndex < validLength; frameIndex += 1) {
        const delta = rawMel[srcBase + frameIndex]! - mean;
        varianceSum += delta * delta;
      }
      const invStd =
        validLength > 1 ? 1.0 / (Math.sqrt(varianceSum / (validLength - 1)) + 1e-5) : 0;

      for (let frameIndex = 0; frameIndex < validLength; frameIndex += 1) {
        features[dstBase + frameIndex] = (rawMel[srcBase + frameIndex]! - mean) * invStd;
      }
      if (validLength < outputFrames) {
        features.fill(0, dstBase + validLength, dstBase + outputFrames);
      }
    }

    return features;
  }

  finalizeFeatures(
    rawMel: Float32Array,
    nFrames: number,
    validLength: number,
    outBuffer: Float32Array | null = null,
  ): Float32Array {
    if (this.normalization === 'none') {
      const requiredSize = this.nMels * nFrames;
      const features =
        outBuffer && outBuffer.length >= requiredSize
          ? outBuffer.subarray(0, requiredSize)
          : new Float32Array(requiredSize);

      features.set(rawMel);
      if (validLength < nFrames) {
        for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
          const base = melIndex * nFrames;
          features.fill(0, base + validLength, base + nFrames);
        }
      }

      return features;
    }

    return this.normalizeFeatures(rawMel, nFrames, validLength, outBuffer);
  }
}

export class IncrementalJSMelProcessor {
  readonly nMels: number;
  readonly validLengthMode: 'onnx' | 'centered';
  readonly normalization: 'per_feature' | 'none';
  private readonly preprocessor: JSMelProcessor;
  private readonly boundaryFrames: number;
  private cachedRawMel: Float32Array | null = null;
  private cachedNFrames = 0;
  private cachedAudioLen = 0;
  private cachedFeaturesLen = 0;
  private readonly rawBuffers: [Float32Array | null, Float32Array | null] = [null, null];
  private currentBufferIndex = 0;
  private featuresBuffer: Float32Array | null = null;

  constructor(options: JsNemoMelProcessorOptions = {}) {
    this.preprocessor = new JSMelProcessor(options);
    this.nMels = this.preprocessor.nMels;
    this.validLengthMode = this.preprocessor.validLengthMode;
    this.normalization = this.preprocessor.normalization;
    this.boundaryFrames = options.boundaryFrames ?? 3;
  }

  process(audio: Float32Array, prefixSamples = 0): IncrementalJsNemoMelProcessResult {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return {
        features: new Float32Array(0),
        frameCount: 0,
        length: 0,
        cached: false,
        cachedFrames: 0,
        newFrames: 0,
      };
    }

    const canReuse =
      prefixSamples > 0 && this.cachedRawMel !== null && prefixSamples <= this.cachedAudioLen;

    const predictedFrames = Math.floor(sampleCount / HOP_LENGTH) + 1;
    const requiredRawSize = this.nMels * predictedFrames;
    let currentRawBuffer = this.rawBuffers[this.currentBufferIndex];
    if (!currentRawBuffer || currentRawBuffer.length < requiredRawSize) {
      currentRawBuffer = new Float32Array(Math.ceil(requiredRawSize * 1.2));
      this.rawBuffers[this.currentBufferIndex] = currentRawBuffer;
    }

    const requiredFeatureSize = this.nMels * predictedFrames;
    if (!this.featuresBuffer || this.featuresBuffer.length < requiredFeatureSize) {
      this.featuresBuffer = new Float32Array(Math.ceil(requiredFeatureSize * 1.2));
    }

    if (!canReuse) {
      const { rawMel, nFrames, validLength } = this.preprocessor.computeRawMel(
        audio,
        0,
        currentRawBuffer,
      );
      const features = this.preprocessor.finalizeFeatures(
        rawMel,
        nFrames,
        validLength,
        this.featuresBuffer,
      );

      this.cachedRawMel = rawMel;
      this.cachedNFrames = nFrames;
      this.cachedAudioLen = sampleCount;
      this.cachedFeaturesLen = validLength;
      this.currentBufferIndex ^= 1;

      return {
        features: new Float32Array(features),
        frameCount: nFrames,
        length: validLength,
        cached: false,
        cachedFrames: 0,
        newFrames: validLength,
      };
    }

    const prefixFrames = Math.floor(prefixSamples / HOP_LENGTH);
    const safeFrames = Math.max(
      0,
      Math.min(prefixFrames - this.boundaryFrames, this.cachedFeaturesLen),
    );

    const { rawMel, nFrames, validLength } = this.preprocessor.computeRawMel(
      audio,
      safeFrames,
      currentRawBuffer,
    );
    if (safeFrames > 0 && this.cachedRawMel) {
      for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {
        const srcBase = melIndex * this.cachedNFrames;
        const dstBase = melIndex * nFrames;
        rawMel.set(this.cachedRawMel.subarray(srcBase, srcBase + safeFrames), dstBase);
      }
    }

    const features = this.preprocessor.finalizeFeatures(
      rawMel,
      nFrames,
      validLength,
      this.featuresBuffer,
    );

    this.cachedRawMel = rawMel;
    this.cachedNFrames = nFrames;
    this.cachedAudioLen = sampleCount;
    this.cachedFeaturesLen = validLength;
    this.currentBufferIndex ^= 1;

    return {
      features: new Float32Array(features),
      frameCount: nFrames,
      length: validLength,
      cached: true,
      cachedFrames: safeFrames,
      newFrames: validLength - safeFrames,
    };
  }

  reset(): void {
    this.cachedRawMel = null;
    this.cachedNFrames = 0;
    this.cachedAudioLen = 0;
    this.cachedFeaturesLen = 0;
    this.currentBufferIndex = 0;
  }

  clear(): void {
    this.reset();
  }
}
