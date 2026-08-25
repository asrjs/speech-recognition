import { WhisperMelProcessor } from '../../audio/whisper-mel.js';
import type { AudioBufferLike } from '../../types/index.js';
import type { Qwen3AsrFeatureResult, Qwen3AsrModelConfig } from './types.js';

function resampleLinear(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input;
  if (!Number.isFinite(fromRate) || fromRate <= 0) {
    throw new RangeError(`Qwen3-ASR requires a positive input sample rate; received ${fromRate}.`);
  }
  const outputLength = Math.max(0, Math.round((input.length * toRate) / fromRate));
  const output = new Float32Array(outputLength);
  if (input.length === 0 || output.length === 0) return output;
  const scale = fromRate / toRate;
  for (let index = 0; index < output.length; index += 1) {
    const source = index * scale;
    const left = Math.min(input.length - 1, Math.floor(source));
    const right = Math.min(input.length - 1, left + 1);
    const fraction = source - left;
    output[index] = (input[left] as number) * (1 - fraction) + (input[right] as number) * fraction;
  }
  return output;
}

function padFeatures(features: Float32Array, nMels: number, fromFrames: number, toFrames: number): Float32Array {
  if (fromFrames === toFrames) return features;
  const output = new Float32Array(nMels * toFrames);
  const copyFrames = Math.min(fromFrames, toFrames);
  for (let mel = 0; mel < nMels; mel += 1) {
    const sourceOffset = mel * fromFrames;
    const targetOffset = mel * toFrames;
    output.set(features.subarray(sourceOffset, sourceOffset + copyFrames), targetOffset);
  }
  return output;
}

/** Exact 128-bin Qwen/Whisper-compatible frontend plus graph padding/masking. */
export class Qwen3AsrFeatureProcessor {
  private readonly mel: WhisperMelProcessor;

  constructor(private readonly config: Qwen3AsrModelConfig) {
    this.mel = new WhisperMelProcessor({
      nMels: config.melBins,
      sampleRate: config.sampleRate,
      fastFft: false,
    });
  }

  process(audio: AudioBufferLike): Qwen3AsrFeatureResult {
    const source = audio.channels?.[0] ??
      (audio.data instanceof Float32Array || audio.data instanceof Float64Array
        ? Float32Array.from(audio.data)
        : new Float32Array(0));
    const resampled = resampleLinear(source, audio.sampleRate, this.config.sampleRate);
    const originalSampleCount = resampled.length;
    const paddedSampleCount = Math.max(originalSampleCount, this.config.minInputSamples);
    const waveform = new Float32Array(paddedSampleCount);
    waveform.set(resampled);

    const mel = this.mel.process(waveform);
    const validFrameCount = Math.min(mel.frameCount, Math.floor(originalSampleCount / this.config.hopLength));
    const frameMultiple = this.config.graph.audioFramesMultiple;
    const frameCount = Math.max(frameMultiple, Math.ceil(mel.frameCount / frameMultiple) * frameMultiple);
    const features = padFeatures(mel.features, this.config.melBins, mel.frameCount, frameCount);
    const inputFeaturesMask = new Int32Array(frameCount);
    inputFeaturesMask.fill(1, 0, Math.max(0, Math.min(validFrameCount, frameCount)));

    return {
      features,
      inputFeaturesMask,
      nMels: this.config.melBins,
      frameCount,
      validFrameCount,
      durationSeconds: originalSampleCount / this.config.sampleRate,
      sampleRate: this.config.sampleRate,
    };
  }
}

export function getQwenAudioTokenCount(inputLengthFrames: number): number {
  const inputLengths = Math.max(0, Math.floor(inputLengthFrames));
  const inputLengthsLeave = inputLengths % 100;
  const featLengths = Math.floor((inputLengthsLeave - 1) / 2) + 1;
  return Math.floor((Math.floor((featLengths - 1) / 2) + 1 - 1) / 2) + 1 +
    Math.floor(inputLengths / 100) * 13;
}
