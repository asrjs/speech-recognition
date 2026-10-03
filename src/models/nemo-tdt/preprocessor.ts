import type { AudioBufferLike } from '../../types/index.js';
import { JSMelProcessor } from '../../audio/js-mel.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from './ort.js';
import { disposeOrtOutputs, releaseOrtSession } from './ort.js';
import { honorAbortAfterCreate, withNativeAbortSignalOption } from '../../io/abort.js';
import { importNodeModule, isNodeLikeRuntime } from '../../io/node-compat.js';

export interface NemoPreprocessorResult {
  readonly features: Float32Array;
  readonly frameCount: number;
  readonly validLength: number;
}

export interface NemoPreprocessor {
  process(audio: AudioBufferLike): Promise<NemoPreprocessorResult> | NemoPreprocessorResult;
  dispose?(): Promise<void> | void;
}

export class OnnxNemoPreprocessor implements NemoPreprocessor {
  private sessionPromise?: Promise<OrtSessionLike>;
  private disposePromise?: Promise<void>;
  private disposed = false;

  constructor(
    private readonly ort: OrtModuleLike,
    private readonly modelUrl: string,
    private readonly enableProfiling = false,
    private readonly signal?: { readonly aborted: boolean } | null,
    private readonly executionProvider: 'wasm' | 'cpu' = 'wasm',
    private readonly cpuThreads?: number,
  ) {}

  private async getSession(): Promise<OrtSessionLike> {
    if (this.disposed) {
      throw new Error('NeMo preprocessor is disposed.');
    }
    this.sessionPromise ??= (async () => {
      let modelUrl = this.modelUrl;
      if (isNodeLikeRuntime() && /^file:/i.test(modelUrl)) {
        const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
        modelUrl = fileURLToPath(modelUrl);
      }

      const sessionOptions = {
        executionProviders: [this.executionProvider],
        enableProfiling: this.enableProfiling,
        ...(this.executionProvider === 'cpu'
          ? { intraOpNumThreads: this.cpuThreads ?? 4, interOpNumThreads: 1 }
          : {}),
      };
      const session = await honorAbortAfterCreate(
        () =>
          this.ort.InferenceSession.create(
            modelUrl,
            withNativeAbortSignalOption(sessionOptions, this.signal) ?? sessionOptions,
          ),
        this.signal,
        (value) => releaseOrtSession(value),
      );
      if (this.disposed) {
        await releaseOrtSession(session);
        throw new Error('NeMo preprocessor is disposed.');
      }
      return session;
    })();

    return this.sessionPromise;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    this.disposePromise ??= this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    let session: OrtSessionLike | undefined;
    try {
      session = await this.sessionPromise;
    } catch {
      // Keep the original load error; preprocessor session may never have opened.
    }
    this.sessionPromise = undefined;
    await releaseOrtSession(session);
  }

  async process(audio: AudioBufferLike): Promise<NemoPreprocessorResult> {
    const mono = audio.channels?.[0];
    if (!mono) {
      throw new Error('OnnxNemoPreprocessor expected mono planar audio.');
    }

    const session = await this.getSession();
    let waveformTensor: OrtTensorLike<Float32Array> | undefined;
    let lengthTensor: OrtTensorLike<BigInt64Array> | undefined;
    let outputs: Record<string, OrtTensorLike> | undefined;
    try {
      waveformTensor = new this.ort.Tensor('float32', mono, [1, mono.length]);
      lengthTensor = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(mono.length)]), [1]);
      outputs = await session.run({
        waveforms: waveformTensor,
        waveforms_lens: lengthTensor,
      });
      const featuresTensor = outputs.features;
      const lengthsTensor = outputs.features_lens;
      const melBins = featuresTensor?.dims[1];
      const frameCount = featuresTensor?.dims[2];
      if (
        !featuresTensor ||
        !(featuresTensor.data instanceof Float32Array) ||
        featuresTensor.dims.length !== 3 ||
        featuresTensor.dims[0] !== 1 ||
        !Number.isSafeInteger(melBins) ||
        !Number.isSafeInteger(frameCount) ||
        melBins === undefined ||
        melBins <= 0 ||
        frameCount === undefined ||
        frameCount < 0 ||
        featuresTensor.data.length !== melBins * frameCount
      ) {
        throw new Error(
          'NeMo preprocessor output contract: expected float32 features [1, melBins, frames].',
        );
      }
      if (
        !lengthsTensor ||
        !(
          lengthsTensor.data instanceof BigInt64Array || lengthsTensor.data instanceof Int32Array
        ) ||
        lengthsTensor.dims.length !== 1 ||
        lengthsTensor.dims[0] !== 1 ||
        lengthsTensor.data.length !== 1
      ) {
        throw new Error(
          'NeMo preprocessor output contract: expected int64/int32 features_lens [1].',
        );
      }
      const validLength = Number(lengthsTensor.data[0]);
      if (!Number.isSafeInteger(validLength) || validLength < 0 || validLength > frameCount) {
        throw new Error(
          'NeMo preprocessor output contract: features_lens must be within the frame count.',
        );
      }
      for (const value of featuresTensor.data) {
        if (!Number.isFinite(value)) {
          throw new Error('NeMo preprocessor output contract: features contain nonfinite values.');
        }
      }
      const features = new Float32Array(featuresTensor.data);

      return {
        features,
        frameCount,
        validLength,
      };
    } finally {
      disposeOrtOutputs(outputs);
      waveformTensor?.dispose?.();
      lengthTensor?.dispose?.();
    }
  }
}

export class JsNemoPreprocessor implements NemoPreprocessor {
  private readonly processor: JSMelProcessor;

  constructor(
    options: {
      readonly melBins?: number;
      readonly validLengthMode?: 'onnx' | 'centered';
      readonly normalization?: 'per_feature' | 'none';
    } = {},
  ) {
    this.processor = new JSMelProcessor({
      nMels: options.melBins,
      validLengthMode: options.validLengthMode,
      normalization: options.normalization,
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
