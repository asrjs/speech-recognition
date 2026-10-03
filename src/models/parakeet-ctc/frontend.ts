import { MedAsrJsPreprocessor, type MedAsrJsPreprocessorOptions } from '../lasr-ctc/mel.js';
import type { LasrCtcFeatureBatch, LasrCtcFeaturePreprocessor } from '../lasr-ctc/types.js';

/** Matches transformers' `LOG_ZERO_GUARD_VALUE` (2^-24), used as log(mel + guard). */
const LOG_ZERO_GUARD = 5.960464477539063e-8;

export interface ParakeetCtcJsPreprocessorOptions {
  readonly nMels?: number;
  readonly nFft?: number;
  readonly winLength?: number;
  readonly hopLength?: number;
  readonly preemphasis?: number;
}

/**
 * ParakeetForCTC frontend.
 *
 * The official `ParakeetFeatureExtractor` is a numpy STFT that must match
 * `torch.stft`: a periodic-false Hann window, centered STFT with constant
 * padding, `librosa.filters.mel(norm='slaney')` on the HTK scale, squared
 * magnitudes, then `log(mel + 2^-24)`, laid out `[frames, mels]`. It differs
 * from the shared nemo128 frontend, so this family owns its own preprocessor
 * rather than changing the NeMo path.
 */
export class ParakeetCtcJsPreprocessor implements LasrCtcFeaturePreprocessor {
  private readonly delegate: MedAsrJsPreprocessor;

  constructor(options: ParakeetCtcJsPreprocessorOptions = {}) {
    this.delegate = new MedAsrJsPreprocessor({
      nMels: options.nMels ?? 80,
      nFft: options.nFft ?? 512,
      winLength: options.winLength ?? 400,
      hopLength: options.hopLength ?? 160,
      center: true,
      preemphasis: options.preemphasis ?? 0.97,
      melScale: 'htk',
      slaneyNorm: true,
      normalizeFeatures: false,
      logZeroGuard: LOG_ZERO_GUARD,
      logCombine: 'add',
      windowKind: 'hann',
    } satisfies MedAsrJsPreprocessorOptions);
  }

  process(audio: Float32Array): LasrCtcFeatureBatch {
    return this.delegate.process(audio);
  }
}
