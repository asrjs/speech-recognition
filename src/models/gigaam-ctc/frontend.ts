import { MedAsrJsPreprocessor } from '../lasr-ctc/mel.js';
import type { LasrCtcFeatureBatch, LasrCtcFeaturePreprocessor } from '../lasr-ctc/types.js';

/**
 * GigaAM Multilingual's exported feature contract: 16 kHz audio, 64 mel
 * bins, 320-sample Hann windows, 160-sample hop, and no centering. The
 * preprocessor intentionally returns feature-major [mel, frame] data because
 * the ONNX graph consumes [batch, mel, frame].
 */
export class GigaAmJsPreprocessor implements LasrCtcFeaturePreprocessor {
  private readonly delegate = new MedAsrJsPreprocessor({
    nMels: 64,
    nFft: 320,
    winLength: 320,
    hopLength: 160,
    center: false,
    preemphasis: 0,
    melScale: 'slaney',
    slaneyNorm: false,
    normalizeFeatures: false,
    logZeroGuard: 1e-9,
  });

  process(audio: Float32Array): LasrCtcFeatureBatch {
    return this.delegate.process(audio);
  }
}
