import type { AudioBufferLike } from '../../types/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from './ort.js';

export interface OnnxPreprocessorResult {
  readonly features: Float32Array;
  readonly frameCount: number;
  readonly validLength: number;
}

export class OnnxNemoPreprocessor {
  private sessionPromise?: Promise<OrtSessionLike>;

  constructor(
    private readonly ort: OrtModuleLike,
    private readonly modelUrl: string,
    private readonly enableProfiling = false
  ) {}

  private async getSession(): Promise<OrtSessionLike> {
    this.sessionPromise ??= this.ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ['wasm'],
      enableProfiling: this.enableProfiling
    });

    return this.sessionPromise;
  }

  async process(audio: AudioBufferLike): Promise<OnnxPreprocessorResult> {
    const mono = audio.channels?.[0];
    if (!mono) {
      throw new Error('OnnxNemoPreprocessor expected mono planar audio.');
    }

    const session = await this.getSession();
    const waveformTensor = new this.ort.Tensor('float32', mono, [1, mono.length]);
    const lengthTensor = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(mono.length)]), [1]);

    try {
      const outputs = await session.run({
        waveforms: waveformTensor,
        waveforms_lens: lengthTensor
      });
      const featuresTensor = outputs.features as OrtTensorLike<Float32Array>;
      const lengthsTensor = outputs.features_lens as OrtTensorLike<BigInt64Array | Int32Array>;
      const melBins = Number(featuresTensor.dims[1] ?? 0);
      const features = new Float32Array(featuresTensor.data);

      return {
        features,
        frameCount: melBins > 0 ? Math.floor(features.length / melBins) : 0,
        validLength: Number(lengthsTensor.data[0] ?? 0)
      };
    } finally {
      waveformTensor.dispose?.();
      lengthTensor.dispose?.();
    }
  }
}
