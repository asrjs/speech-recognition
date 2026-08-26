import { MedAsrJsPreprocessor, transposeMelToTxM } from '../lasr-ctc/mel.js';
import type { SenseVoiceFeatureBatch } from './types.js';

/**
 * SenseVoice's ONNX export keeps the Kaldi fbank outside the graph.
 *
 * The low-frame-rate stack and CMVN are folded into the exported model, so
 * this adapter intentionally returns raw `[T, 80]` fbank frames. Keeping the
 * implementation as a thin, explicitly configured adapter avoids silently
 * reusing the 128-bin MedASR contract.
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
}
