/**
 * Whisper-compatible log-mel frontend.
 *
 * Computes 80-bin (or 128-bin for large-v3) log10 mel spectrograms
 * from 16 kHz mono PCM, matching OpenAI Whisper's preprocessing.
 */

const WHISPER_SAMPLE_RATE = 16000;
const WHISPER_N_FFT = 400;
const WHISPER_HOP_LENGTH = 160;
const WHISPER_WIN_LENGTH = 400;
const WHISPER_CLIP_MIN = -4.0;

// Hann window for 400 samples
function createHannWindow(size: number): Float32Array {
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
  }
  return window;
}

// Slaney-style mel scale (librosa compatible)
function hzToMelSlaney(hz: number): number {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  if (hz >= minLogHz) {
    return (Math.log(hz / minLogHz) / Math.log(6.4)) * 27 + (minLogHz / fSp);
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

function createMelFilterbank(nMels: number, sampleRate = WHISPER_SAMPLE_RATE, nFft = WHISPER_N_FFT): Float32Array {
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
  }
  return filterbank;
}

// Cooley-Tukey FFT (power-of-2 only)
function bitReverse(index: number, bits: number): number {
  let reversed = 0;
  for (let b = 0; b < bits; b++) {
    reversed = (reversed << 1) | (index & 1);
    index >>= 1;
  }
  return reversed;
}

function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  const bits = Math.log2(n);
  // Bit-reverse permutation
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (i < j) {
      let tmp = re[i] as number;
      re[i] = re[j] as number;
      re[j] = tmp;
      tmp = im[i] as number;
      im[i] = im[j] as number;
      im[j] = tmp;
    }
  }
  // Butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angleStep = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < halfLen; j++) {
        const wCos = Math.cos(angleStep * j);
        const wSin = Math.sin(angleStep * j);
        const u = i + j;
        const v = u + halfLen;
        const tRe = (re[v] as number) * wCos - (im[v] as number) * wSin;
        const tIm = (re[v] as number) * wSin + (im[v] as number) * wCos;
        re[v] = (re[u] as number) - tRe;
        im[v] = (im[u] as number) - tIm;
        re[u] = (re[u] as number) + tRe;
        im[u] = (im[u] as number) + tIm;
      }
    }
  }
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
    this.fftRe = new Float64Array(this.nFft);
    this.fftIm = new Float64Array(this.nFft);
    this.powerBuf = new Float32Array((this.nFft >> 1) + 1);
  }

  process(audio: Float32Array): WhisperMelProcessResult {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return { features: new Float32Array(0), frameCount: 0, nMels: this.nMels };
    }

    const pad = this.nFft >> 1; // 200
    const paddedLen = sampleCount + 2 * pad;
    const nFrames = Math.floor((paddedLen - this.nFft) / this.hopLength) + 1;

    const features = new Float32Array(this.nMels * nFrames);
    const nFreqs = (this.nFft >> 1) + 1;

    for (let frameIndex = 0; frameIndex < nFrames; frameIndex++) {
      const offset = frameIndex * this.hopLength;

      // Fill FFT buffer with windowed samples
      this.fftRe.fill(0);
      this.fftIm.fill(0);
      for (let i = 0; i < this.winLength; i++) {
        const sampleIdx = offset + i - pad;
        const sample = sampleIdx >= 0 && sampleIdx < sampleCount ? (audio[sampleIdx] as number) : 0;
        this.fftRe[i] = sample * (this.window[i] as number);
      }

      fft(this.fftRe, this.fftIm);

      // Power spectrum
      this.powerBuf[0] = (this.fftRe[0] as number) * (this.fftRe[0] as number) + (this.fftIm[0] as number) * (this.fftIm[0] as number);
      for (let k = 1; k < nFreqs - 1; k++) {
        this.powerBuf[k] = (this.fftRe[k] as number) * (this.fftRe[k] as number) + (this.fftIm[k] as number) * (this.fftIm[k] as number);
      }
      const nyquist = nFreqs - 1;
      this.powerBuf[nyquist] = (this.fftRe[nyquist] as number) * (this.fftRe[nyquist] as number) + (this.fftIm[nyquist] as number) * (this.fftIm[nyquist] as number);

      // Mel filterbank + log10
      for (let melIndex = 0; melIndex < this.nMels; melIndex++) {
        let melPower = 0;
        const fbOffset = melIndex * nFreqs;
        for (let freqIndex = 0; freqIndex < nFreqs; freqIndex++) {
          melPower += (this.powerBuf[freqIndex] as number) * (this.melFilterbank[fbOffset + freqIndex] as number);
        }
        const logValue = melPower > 0 ? Math.log10(melPower) : WHISPER_CLIP_MIN;
        features[melIndex * nFrames + frameIndex] = Math.max(logValue, WHISPER_CLIP_MIN);
      }
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
