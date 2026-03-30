import { FEATURE_DIM, FRAME_LENGTH_SAMPLE, FRAME_SHIFT_SAMPLE, SAMPLE_RATE } from './constants.js';
import { createPoveyWindow, fft, makeBitrev, makeSintbl, upperPowerOfTwo } from './fft.js';

export interface FbankOptions {
  readonly num_bins?: number;
  readonly sample_rate?: number;
  readonly frame_length?: number;
  readonly frame_shift?: number;
  readonly remove_dc_offset?: boolean;
  readonly pre_emphasis?: boolean;
  readonly use_log?: boolean;
  readonly stateful_pre_emphasis?: boolean;
}

function melScale(freq: number): number {
  return 1127.0 * Math.log(1.0 + freq / 700.0);
}

interface MelBin {
  readonly start: number;
  readonly weights: Float32Array;
}

function initMelBins(numBins: number, sampleRate: number, fftPoints: number): MelBin[] {
  const numFftBins = Math.floor(fftPoints / 2);
  const fftBinWidth = sampleRate / fftPoints;
  const melLow = melScale(20.0);
  const melHigh = melScale(sampleRate / 2);
  const melDelta = (melHigh - melLow) / (numBins + 1);
  const bins: MelBin[] = [];
  for (let bin = 0; bin < numBins; bin += 1) {
    const leftMel = melLow + bin * melDelta;
    const centerMel = melLow + (bin + 1) * melDelta;
    const rightMel = melLow + (bin + 2) * melDelta;
    const weights = new Float32Array(numFftBins);
    let firstIndex = -1;
    let lastIndex = -1;

    for (let i = 0; i < numFftBins; i += 1) {
      const freq = fftBinWidth * i;
      const mel = melScale(freq);
      if (mel > leftMel && mel < rightMel) {
        const weight =
          mel <= centerMel
            ? (mel - leftMel) / (centerMel - leftMel)
            : (rightMel - mel) / (rightMel - centerMel);
        weights[i] = weight;
        if (firstIndex < 0) {
          firstIndex = i;
        }
        lastIndex = i;
      }
    }

    if (firstIndex < 0 || lastIndex < firstIndex) {
      bins.push({
        start: 0,
        weights: new Float32Array(1),
      });
      continue;
    }

    bins.push({
      start: firstIndex,
      weights: weights.slice(firstIndex, lastIndex + 1),
    });
  }
  return bins;
}

export class FireRedFbank {
  readonly numBins: number;
  readonly sampleRate: number;
  readonly frameLength: number;
  readonly frameShift: number;
  readonly removeDcOffset: boolean;
  readonly preEmphasis: boolean;
  readonly useLog: boolean;
  readonly statefulPreEmphasis: boolean;

  private readonly fftPoints: number;
  private readonly bitrev: Int32Array;
  private readonly sintbl: Float32Array;
  private readonly window: Float32Array;
  private readonly bins: MelBin[];
  private preEmphasisState = 0.0;

  constructor(options: FbankOptions = {}) {
    this.numBins = options.num_bins ?? FEATURE_DIM;
    this.sampleRate = options.sample_rate ?? SAMPLE_RATE;
    this.frameLength = options.frame_length ?? FRAME_LENGTH_SAMPLE;
    this.frameShift = options.frame_shift ?? FRAME_SHIFT_SAMPLE;
    this.removeDcOffset = options.remove_dc_offset ?? true;
    this.preEmphasis = options.pre_emphasis ?? true;
    this.useLog = options.use_log ?? true;
    this.statefulPreEmphasis = options.stateful_pre_emphasis ?? false;

    this.fftPoints = upperPowerOfTwo(this.frameLength);
    this.bitrev = makeBitrev(this.fftPoints);
    this.sintbl = makeSintbl(this.fftPoints);
    this.window = createPoveyWindow(this.frameLength);
    this.bins = initMelBins(this.numBins, this.sampleRate, this.fftPoints);
  }

  reset(): void {
    this.preEmphasisState = 0.0;
  }

  compute(input: ArrayLike<number>): Float32Array[] {
    const samples = Float32Array.from(input);
    if (samples.length < this.frameLength) {
      return [];
    }
    const numFrames = 1 + Math.floor((samples.length - this.frameLength) / this.frameShift);
    const frames: Float32Array[] = new Array(numFrames);

    const fftReal = new Float32Array(this.fftPoints);
    const fftImag = new Float32Array(this.fftPoints);
    const power = new Float32Array(this.fftPoints / 2);
    let preState = this.statefulPreEmphasis ? this.preEmphasisState : 0.0;

    for (let frameIndex = 0; frameIndex < numFrames; frameIndex += 1) {
      const offset = frameIndex * this.frameShift;
      const frame = samples.slice(offset, offset + this.frameLength);

      if (this.removeDcOffset) {
        let mean = 0;
        for (let i = 0; i < frame.length; i += 1) {
          mean += frame[i]!;
        }
        mean /= frame.length;
        for (let i = 0; i < frame.length; i += 1) {
          frame[i] = frame[i]! - mean;
        }
      }

      if (this.preEmphasis) {
        let prev = preState;
        for (let i = 0; i < frame.length; i += 1) {
          const current = frame[i]!;
          frame[i] = current - 0.97 * prev;
          prev = current;
        }
        preState = frame[frame.length - 1] ?? preState;
      }

      for (let i = 0; i < this.frameLength; i += 1) {
        frame[i] = frame[i]! * this.window[i]!;
      }

      fftReal.fill(0);
      fftImag.fill(0);
      fftReal.set(frame, 0);
      fft(this.bitrev, this.sintbl, fftReal, fftImag);

      for (let i = 0; i < power.length; i += 1) {
        power[i] = fftReal[i]! * fftReal[i]! + fftImag[i]! * fftImag[i]!;
      }

      const feat = new Float32Array(this.numBins);
      for (let i = 0; i < this.numBins; i += 1) {
        const bin = this.bins[i]!;
        let melEnergy = 0;
        for (let j = 0; j < bin.weights.length; j += 1) {
          melEnergy += bin.weights[j]! * power[bin.start + j]!;
        }
        if (this.useLog) {
          melEnergy = Math.log(Math.max(melEnergy, 1e-20));
        }
        feat[i] = melEnergy;
      }
      frames[frameIndex] = feat;
    }

    if (this.statefulPreEmphasis) {
      this.preEmphasisState = preState;
    }
    return frames;
  }
}
