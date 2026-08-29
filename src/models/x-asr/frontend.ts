import { createPoveyWindow, fft, makeBitrev, makeSintbl, upperPowerOfTwo } from '../../runtime/firered-vad/core/fft.js';

/** float32 epsilon; knf log-fbank floors energy with std::numeric_limits<float>::epsilon(). */
const FLOAT32_EPSILON = 1.1920928955078125e-7;

function melScale(freq: number): number {
  return 1127.0 * Math.log(1.0 + freq / 700.0);
}

interface MelBin {
  readonly start: number;
  readonly weights: Float32Array;
}

function resolveHighFreq(sampleRate: number, highFreq: number): number {
  const nyquist = sampleRate / 2;
  return highFreq > 0 ? highFreq : nyquist + highFreq;
}

function initKaldiMelBins(
  numBins: number,
  sampleRate: number,
  fftPoints: number,
  lowFreq: number,
  highFreq: number,
): MelBin[] {
  const numFftBins = Math.floor(fftPoints / 2);
  const fftBinWidth = sampleRate / fftPoints;
  const resolvedHigh = resolveHighFreq(sampleRate, highFreq);
  const melLow = melScale(lowFreq);
  const melHigh = melScale(resolvedHigh);
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
      const mel = melScale(fftBinWidth * i);
      if (mel > leftMel && mel < rightMel) {
        weights[i] =
          mel <= centerMel
            ? (mel - leftMel) / (centerMel - leftMel)
            : (rightMel - mel) / (rightMel - centerMel);
        if (firstIndex < 0) firstIndex = i;
        lastIndex = i;
      }
    }
    if (firstIndex < 0 || lastIndex < firstIndex) {
      bins.push({ start: 0, weights: new Float32Array(1) });
      continue;
    }
    bins.push({ start: firstIndex, weights: weights.slice(firstIndex, lastIndex + 1) });
  }
  return bins;
}

function reflectIndex(index: number, length: number): number {
  let sample = index;
  while (sample < 0 || sample >= length) {
    if (sample < 0) sample = -sample - 1;
    else sample = 2 * length - 1 - sample;
  }
  return sample;
}

function numFrames(numSamples: number, frameShift: number, frameLength: number, snipEdges: boolean): number {
  if (snipEdges) {
    if (numSamples < frameLength) return 0;
    return 1 + Math.floor((numSamples - frameLength) / frameShift);
  }
  return Math.floor((numSamples + Math.floor(frameShift / 2)) / frameShift);
}

function firstSampleOfFrame(frame: number, frameShift: number, frameLength: number, snipEdges: boolean): number {
  if (snipEdges) return frame * frameShift;
  return frameShift * frame + Math.floor(frameShift / 2) - Math.floor(frameLength / 2);
}

/**
 * Sherpa-onnx / kaldi-native-fbank contract used by X-ASR zipformer2:
 * 16 kHz, 80 bins, 25/10 ms, povey, dither 0, snip_edges false, high_freq -400.
 */
export class XAsrJsFrontend {
  private readonly numBins = 80;
  private readonly sampleRate = 16000;
  private readonly frameLength = 400;
  private readonly frameShift = 160;
  private readonly snipEdges = false;
  private readonly fftPoints: number;
  private readonly bitrev: Int32Array;
  private readonly sintbl: Float32Array;
  private readonly window: Float32Array;
  private readonly bins: MelBin[];

  constructor() {
    this.fftPoints = upperPowerOfTwo(this.frameLength);
    this.bitrev = makeBitrev(this.fftPoints);
    this.sintbl = makeSintbl(this.fftPoints);
    this.window = createPoveyWindow(this.frameLength);
    this.bins = initKaldiMelBins(this.numBins, this.sampleRate, this.fftPoints, 20, -400);
  }

  private processFrames(
    frameStart: number,
    frameCount: number,
    sampleCount: number,
    sampleAt: (index: number) => number,
    directSamples?: Float32Array,
  ): Float32Array {
    const output = new Float32Array(frameCount * this.numBins);
    const fftReal = new Float32Array(this.fftPoints);
    const fftImag = new Float32Array(this.fftPoints);
    const power = new Float32Array(this.fftPoints / 2);
    const windowBuf = new Float32Array(this.frameLength);

    for (let outputFrame = 0; outputFrame < frameCount; outputFrame += 1) {
      const frameIndex = frameStart + outputFrame;
      const start = firstSampleOfFrame(frameIndex, this.frameShift, this.frameLength, this.snipEdges);
      for (let i = 0; i < this.frameLength; i += 1) {
        const source = start + i;
        const reflected = source >= 0 && source < sampleCount
          ? source
          : reflectIndex(source, sampleCount);
        windowBuf[i] = directSamples ? directSamples[reflected] ?? 0 : sampleAt(reflected);
      }

      let mean = 0;
      for (let i = 0; i < this.frameLength; i += 1) mean += windowBuf[i]!;
      mean /= this.frameLength;
      for (let i = 0; i < this.frameLength; i += 1) windowBuf[i] = windowBuf[i]! - mean;

      for (let i = this.frameLength - 1; i > 0; i -= 1) {
        windowBuf[i] = windowBuf[i]! - 0.97 * windowBuf[i - 1]!;
      }
      windowBuf[0] = windowBuf[0]! - 0.97 * windowBuf[0]!;

      for (let i = 0; i < this.frameLength; i += 1) {
        windowBuf[i] = windowBuf[i]! * this.window[i]!;
      }

      fftReal.fill(0);
      fftImag.fill(0);
      fftReal.set(windowBuf, 0);
      fft(this.bitrev, this.sintbl, fftReal, fftImag);
      for (let i = 0; i < power.length; i += 1) {
        power[i] = fftReal[i]! * fftReal[i]! + fftImag[i]! * fftImag[i]!;
      }

      const dest = outputFrame * this.numBins;
      for (let bin = 0; bin < this.numBins; bin += 1) {
        const spec = this.bins[bin]!;
        let energy = 0;
        for (let j = 0; j < spec.weights.length; j += 1) {
          energy += spec.weights[j]! * power[spec.start + j]!;
        }
        output[dest + bin] = Math.log(Math.max(energy, FLOAT32_EPSILON));
      }
    }
    return output;
  }

  process(audio: Float32Array): Float32Array {
    const frames = numFrames(audio.length, this.frameShift, this.frameLength, this.snipEdges);
    if (frames <= 0) return new Float32Array(0);
    return this.processFrames(0, frames, audio.length, () => 0, audio);
  }

  /**
   * Process only the feature frames made available by an appended audio chunk.
   * The caller supplies the previous 400-sample tail so frame reflection at a
   * chunk boundary remains identical to a single full-buffer `process()` call.
   */
  processIncremental(
    previousTail: Float32Array,
    previousSampleCount: number,
    appendedAudio: Float32Array,
    previousFrameCount: number,
    final = false,
  ): { readonly features: Float32Array; readonly frameCount: number; readonly tail: Float32Array } {
    const priorSamples = Math.max(0, Math.floor(previousSampleCount));
    const totalSamples = priorSamples + appendedAudio.length;
    const firstFrame = Math.max(0, Math.floor(previousFrameCount));
    const totalFrames = numFrames(totalSamples, this.frameShift, this.frameLength, this.snipEdges);
    // With snip_edges=false, the last frame(s) use reflection at the right
    // edge. Their values change when another chunk arrives, so keep them
    // pending until they are fully sample-backed or the stream is final.
    const stableFrames = this.snipEdges
      ? totalFrames
      : Math.max(
          0,
          Math.floor(
            (totalSamples - firstSampleOfFrame(0, this.frameShift, this.frameLength, false) - this.frameLength) /
              this.frameShift,
          ) + 1,
        );
    const availableFrames = final ? totalFrames : Math.min(totalFrames, stableFrames);
    const frameCount = Math.max(0, availableFrames - firstFrame);
    const historyLength = Math.min(previousTail.length, priorSamples);
    const historyStart = priorSamples - historyLength;
    const context = new Float32Array(historyLength + appendedAudio.length);
    context.set(previousTail.subarray(previousTail.length - historyLength), 0);
    context.set(appendedAudio, historyLength);
    const features = frameCount <= 0
      ? new Float32Array(0)
      : this.processFrames(firstFrame, frameCount, totalSamples, (index) => {
          const localIndex = index - historyStart;
          return localIndex >= 0 && localIndex < context.length ? context[localIndex] ?? 0 : 0;
        });
    const tailLength = Math.min(this.frameLength, totalSamples);
    const tail = tailLength <= 0 ? new Float32Array(0) : context.slice(context.length - tailLength);
    return { features, frameCount, tail };
  }
}
