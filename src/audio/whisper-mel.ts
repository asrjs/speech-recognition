/**
 * Whisper-compatible log-mel frontend.
 *
 * Computes 80-bin (or 128-bin for large-v3) log-mel spectrograms
 * from 16 kHz mono PCM, matching OpenAI Whisper's preprocessing.
 */

const WHISPER_SAMPLE_RATE = 16000;
const WHISPER_N_FFT = 400;
const WHISPER_HOP_LENGTH = 160;
const WHISPER_WIN_LENGTH = 400;
const WHISPER_N_FREQS = (WHISPER_N_FFT >> 1) + 1;

interface FftTwiddles {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly bitrev: Uint32Array;
  readonly inverse: boolean;
}

const FFT_TWIDDLE_CACHE = new Map<string, FftTwiddles>();

function nextPowerOfTwo(size: number): number {
  let value = 1;
  while (value < size) value <<= 1;
  return value;
}

function precomputeTwiddles(size: number, inverse: boolean): FftTwiddles {
  const key = `${size}:${inverse ? 'inverse' : 'forward'}`;
  const cached = FFT_TWIDDLE_CACHE.get(key);
  if (cached) return cached;

  const bits = Math.log2(size);
  if (!Number.isInteger(bits)) {
    throw new Error(`FFT size must be power-of-two. Received: ${size}`);
  }

  const half = size >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  const sign = inverse ? 2 : -2;
  for (let i = 0; i < half; i++) {
    const angle = (sign * Math.PI * i) / size;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }

  const bitrev = new Uint32Array(size);
  for (let i = 0; i < size; i++) {
    let x = i;
    let r = 0;
    for (let bit = 0; bit < bits; bit++) {
      r = (r << 1) | (x & 1);
      x >>= 1;
    }
    bitrev[i] = r;
  }

  const twiddles: FftTwiddles = { cos, sin, bitrev, inverse };
  FFT_TWIDDLE_CACHE.set(key, twiddles);
  return twiddles;
}

function fft(re: Float64Array, im: Float64Array, size: number, twiddles: FftTwiddles): void {
  const bitrev = twiddles.bitrev;
  for (let i = 0; i < size; i++) {
    const j = bitrev[i] as number;
    if (i < j) {
      const tmpRe = re[i] as number;
      re[i] = re[j] as number;
      re[j] = tmpRe;
      const tmpIm = im[i] as number;
      im[i] = im[j] as number;
      im[j] = tmpIm;
    }
  }

  const tCos = twiddles.cos;
  const tSin = twiddles.sin;

  for (let len = 2; len <= size; len <<= 1) {
    const halfLen = len >> 1;
    const step = size / len;
    for (let k = 0; k < halfLen; k++) {
      const twiddleIndex = k * step;
      const wCos = tCos[twiddleIndex] as number;
      const wSin = tSin[twiddleIndex] as number;
      for (let offset = 0; offset < size; offset += len) {
        const even = offset + k;
        const odd = even + halfLen;
        const oddRe = re[odd] as number;
        const oddIm = im[odd] as number;
        const tRe = oddRe * wCos - oddIm * wSin;
        const tIm = oddRe * wSin + oddIm * wCos;
        const evenRe = re[even] as number;
        const evenIm = im[even] as number;
        re[odd] = evenRe - tRe;
        im[odd] = evenIm - tIm;
        re[even] = evenRe + tRe;
        im[even] = evenIm + tIm;
      }
    }
  }

  if (twiddles.inverse) {
    const invSize = 1 / size;
    for (let i = 0; i < size; i++) {
      re[i] = (re[i] as number) * invSize;
      im[i] = (im[i] as number) * invSize;
    }
  }
}

class BluesteinRfft {
  private readonly fftSize: number;
  private readonly forwardTwiddles: FftTwiddles;
  private readonly inverseTwiddles: FftTwiddles;
  private readonly chirpCos: Float64Array;
  private readonly chirpSin: Float64Array;
  private readonly kernelRe: Float64Array;
  private readonly kernelIm: Float64Array;
  private readonly workRe: Float64Array;
  private readonly workIm: Float64Array;

  constructor(private readonly size: number) {
    this.fftSize = nextPowerOfTwo(size * 2 - 1);
    this.forwardTwiddles = precomputeTwiddles(this.fftSize, false);
    this.inverseTwiddles = precomputeTwiddles(this.fftSize, true);
    this.chirpCos = new Float64Array(size);
    this.chirpSin = new Float64Array(size);
    this.kernelRe = new Float64Array(this.fftSize);
    this.kernelIm = new Float64Array(this.fftSize);
    this.workRe = new Float64Array(this.fftSize);
    this.workIm = new Float64Array(this.fftSize);

    this.kernelRe[0] = 1;
    for (let i = 0; i < size; i++) {
      const angle = (Math.PI * i * i) / size;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      this.chirpCos[i] = cos;
      this.chirpSin[i] = sin;
      if (i > 0) {
        this.kernelRe[i] = cos;
        this.kernelIm[i] = sin;
        this.kernelRe[this.fftSize - i] = cos;
        this.kernelIm[this.fftSize - i] = sin;
      }
    }

    fft(this.kernelRe, this.kernelIm, this.fftSize, this.forwardTwiddles);
  }

  transform(input: Float32Array, outRe: Float64Array, outIm: Float64Array): void {
    const size = this.size;
    const workRe = this.workRe;
    const workIm = this.workIm;
    const chirpCos = this.chirpCos;
    const chirpSin = this.chirpSin;

    workRe.fill(0);
    workIm.fill(0);

    for (let i = 0; i < size; i++) {
      const value = input[i] as number;
      workRe[i] = value * (chirpCos[i] as number);
      workIm[i] = -value * (chirpSin[i] as number);
    }

    const fftSize = this.fftSize;
    fft(workRe, workIm, fftSize, this.forwardTwiddles);

    const kernelRe = this.kernelRe;
    const kernelIm = this.kernelIm;
    for (let i = 0; i < fftSize; i++) {
      const aRe = workRe[i] as number;
      const aIm = workIm[i] as number;
      const bRe = kernelRe[i] as number;
      const bIm = kernelIm[i] as number;
      workRe[i] = aRe * bRe - aIm * bIm;
      workIm[i] = aRe * bIm + aIm * bRe;
    }

    fft(workRe, workIm, fftSize, this.inverseTwiddles);

    for (let k = 0; k < outRe.length; k++) {
      const cRe = workRe[k] as number;
      const cIm = workIm[k] as number;
      const cos = chirpCos[k] as number;
      const sin = chirpSin[k] as number;
      outRe[k] = cRe * cos + cIm * sin;
      outIm[k] = cIm * cos - cRe * sin;
    }
  }
}

// Hann window, periodic=True (matches torch.hann_window default)
function createHannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size);
  }
  return window;
}

// Slaney-style mel scale (librosa compatible)
function hzToMelSlaney(hz: number): number {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  if (hz >= minLogHz) {
    return (Math.log(hz / minLogHz) / Math.log(6.4)) * 27 + minLogHz / fSp;
  }
  return hz / fSp;
}

function melToHzSlaney(mel: number): number {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  if (mel >= minLogMel) {
    return minLogHz * Math.exp((Math.log(6.4) * (mel - minLogMel)) / 27);
  }
  return mel * fSp;
}

function createMelFilterbank(
  nMels: number,
  sampleRate = WHISPER_SAMPLE_RATE,
  nFft = WHISPER_N_FFT,
): Float32Array {
  const fMax = sampleRate / 2;
  const nFreqs = (nFft >> 1) + 1;
  const melMin = hzToMelSlaney(0);
  const melMax = hzToMelSlaney(fMax);
  const nPoints = nMels + 2;

  const melPts = new Float64Array(nPoints);
  for (let i = 0; i < nPoints; i++) {
    melPts[i] = melToHzSlaney(melMin + ((melMax - melMin) * i) / (nPoints - 1));
  }

  const filterbank = new Float32Array(nMels * nFreqs);
  for (let melIndex = 0; melIndex < nMels; melIndex++) {
    const lower = melPts[melIndex] as number;
    const center = melPts[melIndex + 1] as number;
    const upper = melPts[melIndex + 2] as number;
    const offset = melIndex * nFreqs;
    for (let freqIndex = 0; freqIndex < nFreqs; freqIndex++) {
      const freq = (fMax * freqIndex) / (nFreqs - 1);
      if (freq >= lower && freq <= center && center !== lower) {
        filterbank[offset + freqIndex] = (freq - lower) / (center - lower);
      } else if (freq > center && freq <= upper && upper !== center) {
        filterbank[offset + freqIndex] = (upper - freq) / (upper - center);
      }
    }
    // Slaney-style normalization: divide by bandwidth so each filter has
    // approximately constant energy per channel (matches librosa norm='slaney')
    const bandwidth = upper - lower;
    if (bandwidth > 0) {
      const scale = 2.0 / bandwidth;
      for (let freqIndex = 0; freqIndex < nFreqs; freqIndex++) {
        const existing = filterbank[offset + freqIndex] as number;
        filterbank[offset + freqIndex] = existing * scale;
      }
    }
  }
  return filterbank;
}

function createMelFilterBounds(
  filterbank: Float32Array,
  nMels: number,
  nFreqs: number,
): Int32Array {
  const bounds = new Int32Array(nMels * 2);
  for (let melIndex = 0; melIndex < nMels; melIndex++) {
    const fbOffset = melIndex * nFreqs;
    let start = -1;
    let end = -1;
    for (let freqIndex = 0; freqIndex < nFreqs; freqIndex++) {
      if ((filterbank[fbOffset + freqIndex] as number) > 0) {
        if (start === -1) start = freqIndex;
        end = freqIndex;
      }
    }
    bounds[melIndex * 2] = start === -1 ? 0 : start;
    bounds[melIndex * 2 + 1] = end === -1 ? 0 : end + 1;
  }
  return bounds;
}

export interface WhisperMelProcessResult {
  readonly features: Float32Array;
  readonly frameCount: number;
  readonly nMels: number;
}

export class WhisperMelProcessor {
  readonly sampleRate: number;
  readonly nMels: number;
  readonly nFft: number;
  readonly hopLength: number;
  readonly winLength: number;
  private readonly window: Float32Array;
  private readonly melFilterbank: Float32Array;
  private readonly melFilterBounds: Int32Array;
  private readonly dftWindowed: Float32Array;
  private readonly rfft: BluesteinRfft;
  private readonly fftRe: Float64Array;
  private readonly fftIm: Float64Array;
  private readonly powerBuf: Float32Array;

  constructor(options: { readonly nMels?: number; readonly sampleRate?: number } = {}) {
    this.sampleRate = options.sampleRate ?? WHISPER_SAMPLE_RATE;
    this.nMels = options.nMels ?? 80;
    this.nFft = WHISPER_N_FFT;
    this.hopLength = WHISPER_HOP_LENGTH;
    this.winLength = WHISPER_WIN_LENGTH;
    this.window = createHannWindow(this.winLength);
    this.melFilterbank = createMelFilterbank(this.nMels, this.sampleRate, this.nFft);
    this.melFilterBounds = createMelFilterBounds(this.melFilterbank, this.nMels, WHISPER_N_FREQS);
    this.dftWindowed = new Float32Array(this.winLength);
    this.rfft = new BluesteinRfft(this.nFft);
    this.fftRe = new Float64Array(WHISPER_N_FREQS);
    this.fftIm = new Float64Array(WHISPER_N_FREQS);
    this.powerBuf = new Float32Array(WHISPER_N_FREQS);
  }

  process(audio: Float32Array): WhisperMelProcessResult {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return { features: new Float32Array(0), frameCount: 0, nMels: this.nMels };
    }

    const nFrames = Math.floor(sampleCount / this.hopLength);
    const pad = this.nFft >> 1; // 200

    const features = new Float32Array(this.nMels * nFrames);
    const nFreqs = WHISPER_N_FREQS;

    // Cache instance variables to local scope
    const fftRe = this.fftRe;
    const fftIm = this.fftIm;
    const powerBuf = this.powerBuf;
    const window = this.window;
    const dftWindowed = this.dftWindowed;
    const melFilterbank = this.melFilterbank;
    const melFilterBounds = this.melFilterBounds;
    const rfft = this.rfft;
    const hopLength = this.hopLength;
    const winLength = this.winLength;
    const nMels = this.nMels;

    const originalStart = pad;
    const originalEnd = pad + sampleCount - 1;

    for (let frameIndex = 0; frameIndex < nFrames; frameIndex++) {
      const offset = frameIndex * hopLength;

      // Determine phase boundaries bounded by [0, winLength]
      const endPhase1 = Math.min(winLength, Math.max(0, originalStart - offset));
      const endPhase2 = Math.min(winLength, Math.max(0, originalEnd - offset + 1));

      // Window samples with reflect padding (matches torch.stft center=True pad_mode='reflect')

      // Phase 1: Left padding
      for (let i = 0; i < endPhase1; i++) {
        const paddedIdx = offset + i;
        const sample = audio[originalStart - paddedIdx] as number;
        dftWindowed[i] = sample * (window[i] as number);
      }

      // Phase 2: Main audio body
      for (let i = endPhase1; i < endPhase2; i++) {
        const paddedIdx = offset + i;
        const sample = audio[paddedIdx - pad] as number;
        dftWindowed[i] = sample * (window[i] as number);
      }

      // Phase 3: Right padding
      for (let i = endPhase2; i < winLength; i++) {
        const paddedIdx = offset + i;
        const sample = audio[sampleCount - 1 - (paddedIdx - originalEnd)] as number;
        dftWindowed[i] = sample * (window[i] as number);
      }

      rfft.transform(dftWindowed, fftRe, fftIm);

      // Power spectrum
      for (let k = 0; k < nFreqs; k++) {
        const re = fftRe[k] as number;
        const im = fftIm[k] as number;
        powerBuf[k] = re * re + im * im;
      }

      // Mel filterbank + log10
      for (let melIndex = 0; melIndex < nMels; melIndex++) {
        let melPower = 0;
        const fbOffset = melIndex * nFreqs;
        const start = melFilterBounds[melIndex * 2] as number;
        const end = melFilterBounds[melIndex * 2 + 1] as number;
        for (let freqIndex = start; freqIndex < end; freqIndex++) {
          melPower +=
            (powerBuf[freqIndex] as number) * (melFilterbank[fbOffset + freqIndex] as number);
        }
        features[melIndex * nFrames + frameIndex] = melPower > 0 ? Math.log10(melPower) : -10;
      }
    }

    // Post-processing matching OpenAI whisper/audio.py:
    let globalMax = -Infinity;
    for (let i = 0; i < features.length; i++) {
      const v = features[i] as number;
      if (v > globalMax) globalMax = v;
    }
    const clipMin = globalMax - 8.0;
    for (let i = 0; i < features.length; i++) {
      const v = features[i] as number;
      const clipped = v > clipMin ? v : clipMin;
      features[i] = (clipped + 4.0) / 4.0;
    }

    return { features, frameCount: nFrames, nMels: this.nMels };
  }

  /**
   * Pad or truncate mel features to a target frame count.
   * Whisper encoder expects [batch, n_mels, max_frames].
   */
  static padToFrames(result: WhisperMelProcessResult, targetFrames: number): Float32Array {
    const { features, frameCount, nMels } = result;
    if (frameCount === targetFrames) {
      return features;
    }
    const out = new Float32Array(nMels * targetFrames);
    const copyFrames = Math.min(frameCount, targetFrames);
    for (let mel = 0; mel < nMels; mel++) {
      const srcOffset = mel * frameCount;
      const dstOffset = mel * targetFrames;
      for (let f = 0; f < copyFrames; f++) {
        out[dstOffset + f] = features[srcOffset + f] as number;
      }
    }
    return out;
  }
}
