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

// Real-input DFT returning first nFreqs bins (slow but correct for any n)
function rfftDirect(input: Float32Array, outRe: Float64Array, outIm: Float64Array): void {
  const n = input.length;
  const nFreqs = outRe.length;
  for (let k = 0; k < nFreqs; k++) {
    let sumRe = 0;
    let sumIm = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      const v = input[t] as number;
      sumRe += v * Math.cos(angle);
      sumIm -= v * Math.sin(angle);
    }
    outRe[k] = sumRe;
    outIm[k] = sumIm;
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
  private readonly dftWindowed: Float32Array;

  constructor(options: { readonly nMels?: number; readonly sampleRate?: number } = {}) {
    this.sampleRate = options.sampleRate ?? WHISPER_SAMPLE_RATE;
    this.nMels = options.nMels ?? 80;
    this.nFft = WHISPER_N_FFT;
    this.hopLength = WHISPER_HOP_LENGTH;
    this.winLength = WHISPER_WIN_LENGTH;
    this.window = createHannWindow(this.winLength);
    this.melFilterbank = createMelFilterbank(this.nMels, this.sampleRate, this.nFft);
    this.dftWindowed = new Float32Array(this.winLength);
  }

  process(audio: Float32Array): WhisperMelProcessResult {
    const sampleCount = audio.length;
    if (sampleCount === 0) {
      return { features: new Float32Array(0), frameCount: 0, nMels: this.nMels };
    }

    // OpenAI whisper drops the last STFT frame: nFrames = floor(sampleCount / hopLength)
    const nFrames = Math.floor(sampleCount / this.hopLength);
    const pad = this.nFft >> 1; // 200

    const features = new Float32Array(this.nMels * nFrames);
    const nFreqs = (this.nFft >> 1) + 1; // 201
    const fftRe = new Float64Array(nFreqs);
    const fftIm = new Float64Array(nFreqs);

    for (let frameIndex = 0; frameIndex < nFrames; frameIndex++) {
      const offset = frameIndex * this.hopLength;

      // Window samples with reflect padding (matches torch.stft center=True pad_mode='reflect')
      for (let i = 0; i < this.winLength; i++) {
        const paddedIdx = offset + i;
        const sample = this.getReflectPaddedSample(audio, paddedIdx, pad);
        this.dftWindowed[i] = sample * (this.window[i] as number);
      }

      rfftDirect(this.dftWindowed, fftRe, fftIm);

      // Power spectrum
      const powerBuf = new Float32Array(nFreqs);
      for (let k = 0; k < nFreqs; k++) {
        powerBuf[k] = (fftRe[k] as number) * (fftRe[k] as number) + (fftIm[k] as number) * (fftIm[k] as number);
      }

      // Mel filterbank + log10
      for (let melIndex = 0; melIndex < this.nMels; melIndex++) {
        let melPower = 0;
        const fbOffset = melIndex * nFreqs;
        for (let freqIndex = 0; freqIndex < nFreqs; freqIndex++) {
          melPower += (powerBuf[freqIndex] as number) * (this.melFilterbank[fbOffset + freqIndex] as number);
        }
        const logValue = melPower > 0 ? Math.log10(melPower) : -10;
        features[melIndex * nFrames + frameIndex] = logValue;
      }
    }

    // Post-processing matching OpenAI whisper/audio.py:
    // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    // log_spec = (log_spec + 4.0) / 4.0
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

  private getReflectPaddedSample(audio: Float32Array, paddedIdx: number, pad: number): number {
    const sampleCount = audio.length;
    const originalStart = pad;
    const originalEnd = pad + sampleCount - 1;

    if (paddedIdx < originalStart) {
      const dist = originalStart - paddedIdx;
      return audio[dist] as number;
    } else if (paddedIdx > originalEnd) {
      const dist = paddedIdx - originalEnd;
      return audio[sampleCount - 1 - dist] as number;
    } else {
      return audio[paddedIdx - pad] as number;
    }
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
