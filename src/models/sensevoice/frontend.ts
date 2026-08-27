import { MedAsrJsPreprocessor, transposeMelToTxM } from '../lasr-ctc/mel.js';
import type { SenseVoiceFeatureBatch } from './types.js';

export const SENSEVOICE_LFR_M = 7;
export const SENSEVOICE_LFR_N = 6;
export const SENSEVOICE_LFR_DIM = 80 * SENSEVOICE_LFR_M;

export interface SenseVoiceCmvn {
  readonly means: Float32Array;
  readonly scales: Float32Array;
}

/**
 * SenseVoice's official FunASR ONNX export keeps Kaldi fbank, LFR, and CMVN
 * outside the graph and feeds `speech` `[B,T,560]`.
 *
 * Some third-party graphs fold LFR/CMVN in and take raw `[T,80]` fbank. The
 * preprocessor still exposes that 80-bin path; the executor chooses which
 * contract to feed from the session input names.
 */
export class SenseVoiceJsPreprocessor {
  private readonly delegate = new MedAsrJsPreprocessor({
    nMels: 80,
    center: false,
    preemphasis: 0.97,
    melScale: 'kaldi',
    slaneyNorm: false,
    logZeroGuard: 1e-5,
    normalizeFeatures: false,
    windowKind: 'hamming',
    removeDcOffset: true,
    framePreemphasis: true,
    melLowHz: 20,
    melHighHz: -400,
  });

  process(audio: Float32Array): SenseVoiceFeatureBatch {
    const result = this.delegate.process(audio);
    const features = transposeMelToTxM(result.features, result.featureSize, result.frameCount);
    return {
      ...result,
      features,
      validFrameCount: result.frameCount,
    };
  }

  processOfficial(audio: Float32Array, cmvn: SenseVoiceCmvn): SenseVoiceFeatureBatch {
    const fbank = this.process(audio);
    const lfr = applySenseVoiceLfr(fbank.features, fbank.frameCount, 80);
    const features = applySenseVoiceCmvn(lfr.features, lfr.frameCount, SENSEVOICE_LFR_DIM, cmvn);
    return {
      ...fbank,
      features,
      featureSize: SENSEVOICE_LFR_DIM,
      frameCount: lfr.frameCount,
      validFrameCount: lfr.frameCount,
    };
  }
}

export function parseSenseVoiceCmvn(text: string): SenseVoiceCmvn {
  const means = extractKaldiVector(text, 'AddShift');
  const scales = extractKaldiVector(text, 'Rescale');
  if (means.length !== SENSEVOICE_LFR_DIM || scales.length !== SENSEVOICE_LFR_DIM) {
    throw new Error(
      `SenseVoice CMVN expected ${SENSEVOICE_LFR_DIM} means/scales, got ${means.length}/${scales.length}.`,
    );
  }
  return { means, scales };
}

function extractKaldiVector(text: string, section: 'AddShift' | 'Rescale'): Float32Array {
  const marker = `<${section}>`;
  const start = text.indexOf(marker);
  if (start < 0) throw new Error(`SenseVoice CMVN is missing ${marker}.`);
  const learn = text.indexOf('<LearnRateCoef>', start);
  if (learn < 0) throw new Error(`SenseVoice CMVN is missing <LearnRateCoef> after ${marker}.`);
  const bracket = text.indexOf('[', learn);
  const end = text.indexOf(']', bracket);
  if (bracket < 0 || end < 0) throw new Error(`SenseVoice CMVN ${section} vector is malformed.`);
  const values = text
    .slice(bracket + 1, end)
    .trim()
    .split(/\s+/)
    .filter((item) => item.length > 0)
    .map((item) => Number.parseFloat(item));
  return Float32Array.from(values);
}

export function applySenseVoiceLfr(
  frames: Float32Array,
  frameCount: number,
  nMels = 80,
  lfrM = SENSEVOICE_LFR_M,
  lfrN = SENSEVOICE_LFR_N,
): { readonly features: Float32Array; readonly frameCount: number } {
  if (frameCount <= 0) return { features: new Float32Array(0), frameCount: 0 };
  const tLfr = Math.ceil(frameCount / lfrN);
  const leftPad = Math.floor((lfrM - 1) / 2);
  const paddedCount = frameCount + leftPad;
  const padded = new Float32Array(paddedCount * nMels);
  const first = frames.subarray(0, nMels);
  for (let index = 0; index < leftPad; index += 1) padded.set(first, index * nMels);
  padded.set(frames.subarray(0, frameCount * nMels), leftPad * nMels);

  const out = new Float32Array(tLfr * lfrM * nMels);
  const lastFrame = padded.subarray((paddedCount - 1) * nMels, paddedCount * nMels);
  for (let index = 0; index < tLfr; index += 1) {
    const start = index * lfrN;
    const dest = index * lfrM * nMels;
    if (lfrM <= paddedCount - start) {
      out.set(padded.subarray(start * nMels, (start + lfrM) * nMels), dest);
    } else {
      const available = paddedCount - start;
      out.set(padded.subarray(start * nMels, paddedCount * nMels), dest);
      let offset = dest + available * nMels;
      for (let pad = 0; pad < lfrM - available; pad += 1) {
        out.set(lastFrame, offset);
        offset += nMels;
      }
    }
  }
  return { features: out, frameCount: tLfr };
}

export function applySenseVoiceCmvn(
  frames: Float32Array,
  frameCount: number,
  dim: number,
  cmvn: SenseVoiceCmvn,
): Float32Array {
  const out = new Float32Array(frameCount * dim);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * dim;
    for (let index = 0; index < dim; index += 1) {
      out[offset + index] = ((frames[offset + index] ?? 0) + (cmvn.means[index] ?? 0)) * (cmvn.scales[index] ?? 1);
    }
  }
  return out;
}
