import type { AudioBufferLike } from '../../types/index.js';
import { JSMelProcessor } from '../../audio/js-mel.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from './ort.js';
import { importNodeModule, isNodeLikeRuntime } from '../../io/node.js';

export interface NemoPreprocessorResult {
  readonly features: Float32Array;
  readonly frameCount: number;
  readonly validLength: number;
}

export interface NemoPreprocessor {
  process(audio: AudioBufferLike): Promise<NemoPreprocessorResult> | NemoPreprocessorResult;
}

export class OnnxNemoPreprocessor implements NemoPreprocessor {
  private sessionPromise?: Promise<OrtSessionLike>;

  constructor(
    private readonly ort: OrtModuleLike,
    private readonly modelUrl: string,
    private readonly enableProfiling = false,
  ) {}

  private async getSession(): Promise<OrtSessionLike> {
    this.sessionPromise ??= (async () => {
      let modelUrl = this.modelUrl;
      if (isNodeLikeRuntime() && /^file:/i.test(modelUrl)) {
        const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
        modelUrl = fileURLToPath(modelUrl);
      }

      return this.ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
        enableProfiling: this.enableProfiling,
      });
    })();

    return this.sessionPromise;
  }

  async process(audio: AudioBufferLike): Promise<NemoPreprocessorResult> {
    const mono = audio.channels?.[0];
    if (!mono) {
      throw new Error('OnnxNemoPreprocessor expected mono planar audio.');
    }

    const session = await this.getSession();
    const waveformTensor = new this.ort.Tensor('float32', mono, [1, mono.length]);
    const lengthTensor = new this.ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(mono.length)]),
      [1],
    );

    try {
      const outputs = await session.run({
        waveforms: waveformTensor,
        waveforms_lens: lengthTensor,
      });
      const featuresTensor = outputs.features as OrtTensorLike<Float32Array>;
      const lengthsTensor = outputs.features_lens as OrtTensorLike<BigInt64Array | Int32Array>;
      const melBins = Number(featuresTensor.dims[1] ?? 0);
      const features = new Float32Array(featuresTensor.data);

      return {
        features,
        frameCount: melBins > 0 ? Math.floor(features.length / melBins) : 0,
        validLength: Number(lengthsTensor.data[0] ?? 0),
      };
    } finally {
      waveformTensor.dispose?.();
      lengthTensor.dispose?.();
    }
  }
}

export class JsNemoPreprocessor implements NemoPreprocessor {
  private readonly processor: JSMelProcessor;

  constructor(
    options: {
      readonly melBins?: number;
      readonly validLengthMode?: 'onnx' | 'centered';
    } = {},
  ) {
    this.processor = new JSMelProcessor({
      nMels: options.melBins,
      validLengthMode: options.validLengthMode,
    });
  }

  process(audio: AudioBufferLike): NemoPreprocessorResult {
    const mono = audio.channels?.[0];
    if (!mono) {
      throw new Error('JsNemoPreprocessor expected mono planar audio.');
    }

    const result = this.processor.process(mono);
    const melBins = this.processor.nMels;
    const frameCount = melBins > 0 ? Math.floor(result.features.length / melBins) : 0;

    return {
      features: result.features,
      frameCount: result.frameCount || frameCount,
      validLength: result.length,
    };
  }
}
