import { MedAsrJsPreprocessor, type MedAsrJsPreprocessorOptions } from '../lasr-ctc/mel.js';
import type { LasrCtcFeatureBatch, LasrCtcFeaturePreprocessor } from '../lasr-ctc/types.js';

export interface GigaAmJsPreprocessorOptions {
  readonly nMels?: number;
  readonly nFft?: number;
  readonly winLength?: number;
  readonly hopLength?: number;
  readonly center?: boolean;
  readonly melScale?: MedAsrJsPreprocessorOptions['melScale'];
  readonly slaneyNorm?: boolean;
  readonly melLowHz?: number;
  readonly melHighHz?: number;
}

/**
 * GigaAM CTC frontend. Geometry defaults match the official multilingual CTC
 * export (16 kHz, 64 bins, 320/320/160). Log compression matches official
 * `SpecScaler` (`log(clamp(x, 1e-9, 1e9))`), not an additive floor. `center`
 * and the torchaudio mel scale/norm must follow the checkpoint preprocessor,
 * not a third-party ONNX card. The tensor layout stays feature-major
 * `[mel, frame]` for the official `[batch, mel, frame]` graph feed.
 */
export class GigaAmJsPreprocessor implements LasrCtcFeaturePreprocessor {
  private readonly delegate: MedAsrJsPreprocessor;

  constructor(options: GigaAmJsPreprocessorOptions = {}) {
    this.delegate = new MedAsrJsPreprocessor({
      nMels: options.nMels ?? 64,
      nFft: options.nFft ?? 320,
      winLength: options.winLength ?? 320,
      hopLength: options.hopLength ?? 160,
      center: options.center ?? false,
      preemphasis: 0,
      melScale: options.melScale ?? 'htk',
      slaneyNorm: options.slaneyNorm ?? false,
      normalizeFeatures: false,
      logZeroGuard: 1e-9,
      logCombine: 'clamp',
      windowKind: 'hann-periodic',
      melLowHz: options.melLowHz,
      melHighHz: options.melHighHz,
    });
  }

  process(audio: Float32Array): LasrCtcFeatureBatch {
    return this.delegate.process(audio);
  }
}
