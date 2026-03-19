import type {
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
  TranscriptionProgressEvent,
} from '../../types/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import {
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  buildSentenceTimings,
  buildUtteranceTiming,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
} from './ctc.js';
import { MedAsrJsPreprocessor, transposeMelToTxM } from './mel.js';
import {
  createOrtSession,
  initOrt,
  resolveLasrCtcArtifacts,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
  type ResolvedLasrCtcArtifacts,
} from './ort.js';
import { MedAsrTextTokenizer } from './tokenizer.js';
import type {
  LasrCtcArtifactSource,
  LasrCtcExecutor,
  LasrCtcFeaturePreprocessor,
  LasrCtcModelConfig,
  LasrCtcModelOptions,
  LasrCtcNativeToken,
  LasrCtcNativeTranscript,
  LasrCtcSentenceTiming,
  LasrCtcTokenSpan,
  LasrCtcTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly session: OrtSessionLike;
  readonly tokenizer: MedAsrTextTokenizer;
  readonly preprocessor: LasrCtcFeaturePreprocessor;
  readonly warnings: readonly TranscriptWarning[];
}

const FLOAT32_BITS_VIEW = new Float32Array(1);
const UINT32_BITS_VIEW = new Uint32Array(FLOAT32_BITS_VIEW.buffer);

function clampProgress(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function estimateRemainingMs(elapsedMs: number, progress: number): number | undefined {
  if (progress <= 0 || progress >= 1) {
    return undefined;
  }

  return roundMetric((elapsedMs / progress) * (1 - progress), 2);
}

function emitTranscriptionProgress(
  options: LasrCtcTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

function roundMiB(bytes: number | undefined): number | undefined {
  if (!Number.isFinite(bytes)) {
    return undefined;
  }

  return roundMetric((bytes as number) / (1024 * 1024), 2);
}

function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: {
    readonly loaded: number;
    readonly total?: number;
    readonly done?: boolean;
  },
): RuntimeProgressEvent {
  const percent =
    event.total && event.total > 0
      ? Math.min(100, Math.round((event.loaded / event.total) * 100))
      : event.done
        ? 100
        : undefined;

  return {
    phase: 'asset:download',
    modelId,
    file,
    loaded: event.loaded,
    total: event.total,
    percent,
    loadedMiB: roundMiB(event.loaded),
    totalMiB: roundMiB(event.total),
    isComplete: event.done,
    message: event.done ? `Prepared ${file}.` : `Downloading ${file}.`,
  };
}

function average(values: readonly number[]): number | undefined {
  if (values.length === 0) {
    return undefined;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function ensureFloat32Buffer(length: number, buffer?: Float32Array): Float32Array {
  return !buffer || buffer.length < length ? new Float32Array(length) : buffer;
}

function prepareMonoBuffer(frames: number, destination?: Float32Array): Float32Array {
  const mono = destination
    ? (destination.length === frames ? destination : destination.subarray(0, frames))
    : new Float32Array(frames);
  mono.fill(0);
  return mono;
}

function toMonoPcm(audio: AudioBufferLike, destination?: Float32Array): Float32Array {
  if (audio.channels && audio.channels.length > 0) {
    if (audio.channels.length === 1) {
      return audio.channels[0] ?? new Float32Array(0);
    }

    const mono = prepareMonoBuffer(audio.numberOfFrames, destination);
    const channelCount = audio.channels.length;

    if (channelCount === 2) {
      const left = audio.channels[0];
      const right = audio.channels[1];
      if (left && right) {
        for (let frameIndex = 0; frameIndex < audio.numberOfFrames; frameIndex += 1) {
          mono[frameIndex] = ((left[frameIndex] ?? 0) + (right[frameIndex] ?? 0)) * 0.5;
        }
        return mono;
      }
    }

    const invChannels = 1 / channelCount;
    for (let frameIndex = 0; frameIndex < audio.numberOfFrames; frameIndex += 1) {
      let sampleSum = 0;
      for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
        sampleSum += audio.channels[channelIndex]?.[frameIndex] ?? 0;
      }
      mono[frameIndex] = sampleSum * invChannels;
    }

    return mono;
  }

  const numberOfChannels = Math.max(1, audio.numberOfChannels || 1);
  if (audio.data instanceof Float32Array || audio.data instanceof Float64Array) {
    const frames = Math.floor(audio.data.length / numberOfChannels);
    if (numberOfChannels === 1) {
      return Float32Array.from(audio.data.subarray(0, frames));
    }

    const mono = prepareMonoBuffer(frames, destination);
    const data = audio.data;

    if (numberOfChannels === 2) {
      for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
        const baseIndex = frameIndex * 2;
        mono[frameIndex] = ((data[baseIndex] ?? 0) + (data[baseIndex + 1] ?? 0)) * 0.5;
      }
      return mono;
    }

    const invChannels = 1 / numberOfChannels;
    for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
      let sampleSum = 0;
      const baseIndex = frameIndex * numberOfChannels;
      for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
        sampleSum += data[baseIndex + channelIndex] ?? 0;
      }
      mono[frameIndex] = sampleSum * invChannels;
    }
    return mono;
  }

  if (audio.data instanceof Int16Array) {
    const frames = Math.floor(audio.data.length / numberOfChannels);
    const mono = prepareMonoBuffer(frames, destination);
    const data = audio.data;
    const int16Scale = 1 / 32768;

    if (numberOfChannels === 2) {
      for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
        const baseIndex = frameIndex * 2;
        mono[frameIndex] = ((data[baseIndex] ?? 0) + (data[baseIndex + 1] ?? 0)) * 0.5 * int16Scale;
      }
      return mono;
    }

    const sampleScale = int16Scale / numberOfChannels;
    for (let frameIndex = 0; frameIndex < frames; frameIndex += 1) {
      let sampleSum = 0;
      const baseIndex = frameIndex * numberOfChannels;
      for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
        sampleSum += data[baseIndex + channelIndex] ?? 0;
      }
      mono[frameIndex] = sampleSum * sampleScale;
    }
    return mono;
  }

  throw new Error('Unsupported audio buffer shape for LASR CTC executor.');
}

function parseOrtTensorElementType(ortType: string | undefined, fallback: string): string {
  if (!ortType) {
    return fallback;
  }

  const match = /^tensor\((.+)\)$/.exec(ortType.trim());
  const elementType = (match?.[1] ?? ortType).trim();
  if (elementType === 'float') {
    return 'float32';
  }

  return elementType;
}

function getInputElementType(session: OrtSessionLike, inputName: string, fallback: string): string {
  return parseOrtTensorElementType(session.inputMetadata?.[inputName]?.type, fallback);
}

function float32ToFloat16Bits(value: number): number {
  FLOAT32_BITS_VIEW[0] = value;
  const bits = UINT32_BITS_VIEW[0] ?? 0;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x007fffff;

  if (exponent === 0xff) {
    return mantissa !== 0 ? sign | 0x7e00 : sign | 0x7c00;
  }
  if (exponent > 142) {
    return sign | 0x7c00;
  }
  if (exponent < 113) {
    if (exponent < 103) {
      return sign;
    }
    const shifted = (0x00800000 | mantissa) >> (114 - exponent);
    return sign | ((shifted + 1) >> 1);
  }

  const halfExponent = exponent - 112;
  const halfMantissa = mantissa + 0x00001000;
  return sign | (halfExponent << 10) | (halfMantissa >> 13);
}

function float16BitsToFloat32(value: number): number {
  const sign = (value & 0x8000) << 16;
  const exponent = (value >>> 10) & 0x1f;
  const mantissa = value & 0x03ff;

  if (exponent === 0) {
    if (mantissa === 0) {
      UINT32_BITS_VIEW[0] = sign;
      return FLOAT32_BITS_VIEW[0] ?? 0;
    }

    let normalizedMantissa = mantissa;
    let shift = -1;
    do {
      shift += 1;
      normalizedMantissa <<= 1;
    } while ((normalizedMantissa & 0x0400) === 0);
    normalizedMantissa &= 0x03ff;
    UINT32_BITS_VIEW[0] = sign | ((127 - 15 - shift) << 23) | (normalizedMantissa << 13);
    return FLOAT32_BITS_VIEW[0] ?? 0;
  }

  if (exponent === 0x1f) {
    UINT32_BITS_VIEW[0] = sign | 0x7f800000 | (mantissa << 13);
    return FLOAT32_BITS_VIEW[0] ?? 0;
  }

  UINT32_BITS_VIEW[0] = sign | ((exponent + 112) << 23) | (mantissa << 13);
  return FLOAT32_BITS_VIEW[0] ?? 0;
}

function float32ToFloat16Array(values: Float32Array): Uint16Array {
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = float32ToFloat16Bits(values[index] ?? 0);
  }
  return output;
}

function createInputFeaturesTensor(
  ort: OrtModuleLike,
  session: OrtSessionLike,
  featuresTxM: Float32Array,
  frames: number,
  nMels: number,
): OrtTensorLike {
  const expectedType = getInputElementType(session, 'input_features', 'float32');
  if (expectedType === 'float16') {
    return new ort.Tensor('float16', float32ToFloat16Array(featuresTxM), [1, frames, nMels]);
  }

  return new ort.Tensor('float32', featuresTxM, [1, frames, nMels]);
}

function createAttentionMaskTensor(
  ort: OrtModuleLike,
  session: OrtSessionLike,
  frames: number,
): OrtTensorLike {
  const expectedMaskType = getInputElementType(session, 'attention_mask', 'int32');
  if (expectedMaskType === 'int64') {
    const data = new BigInt64Array(frames);
    for (let index = 0; index < frames; index += 1) {
      data[index] = 1n;
    }
    return new ort.Tensor('int64', data, [1, frames]);
  }

  const data = new Int32Array(frames);
  data.fill(1);
  return new ort.Tensor('int32', data, [1, frames]);
}

function normalizeLogitsData(logitsTensor: OrtTensorLike): Float32Array {
  const tensorType = logitsTensor.type ?? 'float32';
  if (tensorType !== 'float16') {
    const source = logitsTensor.data as Float32Array;
    return source instanceof Float32Array ? source : Float32Array.from(source);
  }

  const source = logitsTensor.data as Uint16Array;
  const normalized = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    normalized[index] = float16BitsToFloat32(source[index] ?? 0);
  }

  return normalized;
}

function findLogitsTensor(outputs: Record<string, OrtTensorLike>): OrtTensorLike {
  if (outputs.logits) {
    return outputs.logits;
  }

  const first = Object.values(outputs)[0];
  if (!first) {
    throw new Error('LASR CTC encoder run returned no output tensors.');
  }

  return first;
}

export class OrtLasrCtcExecutor implements LasrCtcExecutor {
  private readonly sourceOptions: LasrCtcModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly preprocessor?: LasrCtcFeaturePreprocessor;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private sharedMonoBuffer?: Float32Array;
  private featuresTxMBuffer?: Float32Array;

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: LasrCtcModelConfig,
    private readonly backendId: string,
    loadOptions: LasrCtcModelOptions | undefined,
    dependencies: {
      readonly assetProvider?: AssetProvider;
      readonly runtimeHooks?: SpeechRuntimeHooks;
      readonly preprocessor?: LasrCtcFeaturePreprocessor;
    } = {},
  ) {
    this.sourceOptions = loadOptions?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    this.preprocessor = dependencies.preprocessor;

    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize();
    }
  }

  private async materializeHuggingFaceArtifacts(
    source: Extract<LasrCtcArtifactSource, { kind: 'huggingface' }>,
    artifacts: ResolvedLasrCtcArtifacts['artifacts'],
  ): Promise<{
    readonly artifacts: ResolvedLasrCtcArtifacts['artifacts'];
    readonly warnings: readonly TranscriptWarning[];
  }> {
    if (!this.assetProvider) {
      return {
        artifacts,
        warnings: [],
      };
    }

    const warnings: TranscriptWarning[] = [];
    const revision = source.revision ?? 'main';
    const subfolder = source.subfolder ?? '';
    const resolveFile = async (filename: string, optional = false): Promise<string | undefined> => {
      try {
        const handle = await this.assetProvider!.resolve({
          id: `huggingface:${source.repoId}:${revision}:${subfolder}:${filename}`,
          provider: 'huggingface',
          repoId: source.repoId,
          revision,
          subfolder,
          filename,
          cacheKey: `huggingface:${source.repoId}:${revision}:${subfolder}:${filename}`,
          onProgress: (event) => {
            this.runtimeHooks?.onProgress?.(
              createAssetProgressEvent(this.modelId, filename, {
                loaded: event.loaded,
                total: event.total,
                done: event.done,
              }),
            );
          },
        });
        this.assetHandles.push(handle);
        const locator = await handle.getLocator('url');
        if (!locator) {
          throw new Error(`Could not create a URL locator for "${filename}".`);
        }
        return locator;
      } catch (error) {
        if (optional) {
          warnings.push({
            code: 'lasr-ctc.optional-asset-missing',
            message: `Optional asset "${filename}" was not found for ${this.modelId}.`,
            recoverable: true,
          });
          return undefined;
        }
        throw error;
      }
    };

    const modelFilename = source.modelFilename ?? 'model.onnx';
    const tokenizerFilename = source.tokenizerFilename ?? 'tokens.txt';
    const modelDataFilename = source.modelDataFilename ?? artifacts.modelDataFilename;

    let tokenizerUrl = await resolveFile(tokenizerFilename, tokenizerFilename !== 'vocab.txt');
    if (!tokenizerUrl) {
      tokenizerUrl = await resolveFile('vocab.txt');
      warnings.push({
        code: 'lasr-ctc.tokenizer-fallback',
        message: `Tokenizer "${tokenizerFilename}" was unavailable. Falling back to "vocab.txt".`,
        recoverable: true,
      });
    }

    const modelDataUrl = modelDataFilename ? await resolveFile(modelDataFilename, true) : undefined;

    return {
      artifacts: {
        ...artifacts,
        modelUrl: (await resolveFile(modelFilename)) ?? artifacts.modelUrl,
        modelDataUrl,
        modelDataFilename,
        tokenizerUrl: tokenizerUrl ?? artifacts.tokenizerUrl,
      },
      warnings,
    };
  }

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = resolveLasrCtcArtifacts(this.sourceOptions, this.backendId);
    let artifacts = resolved.artifacts;
    const warnings: TranscriptWarning[] = [];

    if (this.sourceOptions.kind === 'huggingface') {
      const materialized = await this.materializeHuggingFaceArtifacts(
        this.sourceOptions,
        artifacts,
      );
      artifacts = materialized.artifacts;
      warnings.push(...materialized.warnings);
    }

    const ort = await initOrt(this.backendId, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
    });

    let session: OrtSessionLike;
    try {
      session = await createOrtSession(ort, artifacts.modelUrl, {
        backendId: resolved.backendForOrt,
        enableProfiling: resolved.enableProfiling,
        externalDataUrl: artifacts.modelDataUrl,
        externalDataPath: artifacts.modelDataFilename,
      });
    } catch (error) {
      if (!artifacts.modelDataUrl) {
        throw error;
      }

      session = await createOrtSession(ort, artifacts.modelUrl, {
        backendId: resolved.backendForOrt,
        enableProfiling: resolved.enableProfiling,
      });
      warnings.push({
        code: 'lasr-ctc.external-data-fallback',
        message:
          'Failed to initialize optional external ONNX data file. Retrying with single-file model load.',
        recoverable: true,
      });
    }

    let tokenizer: MedAsrTextTokenizer;
    try {
      tokenizer = await MedAsrTextTokenizer.fromUrl(artifacts.tokenizerUrl);
    } catch (error) {
      if (!artifacts.tokenizerFallbackUrl) {
        throw error;
      }

      tokenizer = await MedAsrTextTokenizer.fromUrl(artifacts.tokenizerFallbackUrl);
      warnings.push({
        code: 'lasr-ctc.tokenizer-fallback',
        message:
          'Tokenizer URL from primary source could not be loaded. Falling back to vocab.txt decoder.',
        recoverable: true,
      });
    }

    const preprocessor =
      this.preprocessor ??
      new MedAsrJsPreprocessor({
        nMels: this.config.nMels,
        center: false,
        preemphasis: 0,
        melScale: 'kaldi',
        slaneyNorm: false,
        logZeroGuard: 1e-5,
        normalizeFeatures: false,
      });

    return {
      ort,
      session,
      tokenizer,
      preprocessor,
      warnings,
    };
  }

  private async getLoadedState(): Promise<LoadedExecutorState> {
    if (!this.loadStatePromise) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    return this.loadStatePromise;
  }

  async ready(): Promise<void> {
    await this.getLoadedState();
  }

  async transcribe(
    audio: AudioBufferLike,
    options: LasrCtcTranscriptionOptions,
  ): Promise<LasrCtcNativeTranscript> {
    const state = await this.getLoadedState();
    const warnings: TranscriptWarning[] = [...state.warnings];
    const transcriptionStart = nowMs();

    emitTranscriptionProgress(options, {
      stage: 'start',
      progress: 0,
      elapsedMs: 0,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Starting transcription for ${this.modelId}.`,
    });

    if (audio.sampleRate !== this.config.sampleRate) {
      warnings.push({
        code: 'lasr-ctc.sample-rate-mismatch',
        message: `Expected ${this.config.sampleRate} Hz audio but received ${audio.sampleRate} Hz. No resampler is active on this path.`,
        recoverable: true,
      });
    }

    this.sharedMonoBuffer = ensureFloat32Buffer(audio.numberOfFrames, this.sharedMonoBuffer);
    const mono = toMonoPcm(audio, this.sharedMonoBuffer);

    const preprocessStart = nowMs();
    const features = state.preprocessor.process(mono);
    const transposeStart = nowMs();
    this.featuresTxMBuffer = ensureFloat32Buffer(
      features.featureSize * features.frameCount,
      this.featuresTxMBuffer,
    );
    const featuresTxM = transposeMelToTxM(
      features.features,
      features.featureSize,
      features.frameCount,
      this.featuresTxMBuffer,
    );
    const transposeMs = nowMs() - transposeStart;
    const preprocessMs = nowMs() - preprocessStart;
    const totalPreprocessMs = preprocessMs + transposeMs;
    const preprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'preprocess',
      progress: 0.25,
      elapsedMs: roundMetric(preprocessElapsedMs),
      remainingMs: estimateRemainingMs(preprocessElapsedMs, 0.25),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Prepared features for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(totalPreprocessMs),
      },
    });

    if (features.frameCount <= 0) {
      const totalMs = roundMetric(nowMs() - transcriptionStart);
      emitTranscriptionProgress(options, {
        stage: 'complete',
        progress: 1,
        elapsedMs: totalMs,
        modelId: this.modelId,
        backendId: this.backendId,
        message: `Finished transcription for ${this.modelId}.`,
        metrics: {
          preprocessMs: roundMetric(totalPreprocessMs),
          totalMs,
        },
      });

      return {
        utteranceText: '',
        isFinal: true,
        warnings,
        metrics: {
          preprocessMs: roundMetric(totalPreprocessMs),
          totalMs,
          wallMs: totalMs,
          audioDurationSec: roundMetric(audio.durationSeconds, 4),
          rtf: 0,
          rtfx: undefined,
          preprocessorBackend: 'js',
          encoderFrameCount: 0,
          decodeIterations: 0,
          emittedTokenCount: 0,
          emittedWordCount: 0,
        },
        ctc: {
          frameIds: options.returnFrameIds ? [] : undefined,
          collapsedIds: options.returnTokenIds ? [] : undefined,
          secondsPerFrame: 0,
          utterance: {
            hasSpeech: false,
            startFrame: null,
            endFrame: null,
            startTime: 0,
            endTime: 0,
            duration: 0,
            confidence: 0,
          },
          tokenSpans: [],
          sentences: [],
        },
      };
    }

    const inputTensor = createInputFeaturesTensor(
      state.ort,
      state.session,
      featuresTxM,
      features.frameCount,
      features.featureSize,
    );
    const maskTensor = createAttentionMaskTensor(state.ort, state.session, features.frameCount);

    const encodeStart = nowMs();
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({
        input_features: inputTensor,
        attention_mask: maskTensor,
      });
    } finally {
      inputTensor.dispose?.();
      maskTensor.dispose?.();
    }
    const encodeMs = nowMs() - encodeStart;
    const encodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.6,
      elapsedMs: roundMetric(encodeElapsedMs),
      remainingMs: estimateRemainingMs(encodeElapsedMs, 0.6),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Encoded logits for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(totalPreprocessMs),
        encodeMs: roundMetric(encodeMs),
      },
    });

    const logitsTensor = findLogitsTensor(outputs);
    const logits = normalizeLogitsData(logitsTensor);
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || (dims[0] ?? 0) !== 1) {
      throw new Error(`Unexpected LASR CTC logits shape: [${dims.join(', ')}].`);
    }

    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    if (outFrames <= 0 || vocabSize <= 0) {
      throw new Error(`LASR CTC logits shape is invalid: [${dims.join(', ')}].`);
    }

    const decodeStart = nowMs();
    const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(logits, outFrames, vocabSize);
    const blankId =
      state.tokenizer.blankId ?? this.config.tokenizer.blankTokenId ?? Math.max(0, vocabSize - 1);
    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(frameIds, selectedLogProbs, blankId);
    const text = state.tokenizer.decode(collapsedIds);
    const secondsPerFrame = estimateSecondsPerOutputFrame({
      audioDurationSec: audio.durationSeconds,
      inputFrames: features.frameCount,
      inputFrameHopSeconds: this.config.featureHopSeconds,
      outFrames,
    });
    const timedTokenSpans = addTimesToTokenSpans(state.tokenizer, tokenSpans, secondsPerFrame);
    const utterance = buildUtteranceTiming(frameIds, selectedLogProbs, blankId, secondsPerFrame);
    const sentenceTimings = buildSentenceTimings(
      text,
      state.tokenizer,
      collapsedIds,
      timedTokenSpans,
    );
    const decodeMs = nowMs() - decodeStart;

    const tokens: LasrCtcNativeToken[] = timedTokenSpans.map((span, index) => ({
      index,
      id: options.returnTokenIds ? span.tokenId : undefined,
      text: span.text,
      startTime: roundTimestampSeconds(span.startTime),
      endTime: roundTimestampSeconds(span.endTime),
      confidence: roundMetric(span.confidence, 4),
      logitIndex: options.returnLogitIndices ? span.startFrame : undefined,
    }));

    const postprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'postprocess',
      progress: 0.9,
      elapsedMs: roundMetric(postprocessElapsedMs),
      remainingMs: estimateRemainingMs(postprocessElapsedMs, 0.9),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Built transcript details for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(totalPreprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
      },
    });

    const totalMs = roundMetric(nowMs() - transcriptionStart);
    const tokenConfidences = timedTokenSpans.map((span) => span.confidence);
    const sentenceConfidences = sentenceTimings.map((sentence) => sentence.confidence);
    const rtf = audio.durationSeconds > 0 ? totalMs / (audio.durationSeconds * 1000) : 0;
    const rtfx = audio.durationSeconds > 0 ? audio.durationSeconds / (totalMs / 1000) : undefined;

    const nativeTranscript: LasrCtcNativeTranscript = {
      utteranceText: text,
      isFinal: true,
      tokens,
      confidence: {
        utterance: utterance.confidence,
        tokenAverage: average(tokenConfidences),
        wordAverage: average(sentenceConfidences),
      },
      metrics: {
        preprocessMs: roundMetric(totalPreprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        postprocessMs: 0,
        totalMs,
        wallMs: totalMs,
        audioDurationSec: roundMetric(audio.durationSeconds, 4),
        rtf: roundMetric(rtf, 4),
        rtfx: rtfx === undefined ? undefined : roundMetric(rtfx, 4),
        preprocessorBackend: 'js',
        encoderFrameCount: outFrames,
        decodeIterations: outFrames,
        emittedTokenCount: tokens.length,
        emittedWordCount: sentenceTimings.length,
      },
      ctc: {
        frameIds: options.returnFrameIds ? frameIds : undefined,
        collapsedIds: options.returnTokenIds ? collapsedIds : undefined,
        secondsPerFrame,
        utterance,
        tokenSpans: timedTokenSpans.map<LasrCtcTokenSpan>((span) => ({
          tokenId: span.tokenId,
          text: span.text,
          startFrame: span.startFrame,
          endFrame: span.endFrame,
          frameCount: span.frameCount,
          startTime: roundTimestampSeconds(span.startTime),
          endTime: roundTimestampSeconds(span.endTime),
          duration: roundMetric(span.duration, 4),
          confidence: roundMetric(span.confidence, 4),
          averageLogProb: roundMetric(span.averageLogProb, 6),
        })),
        sentences: sentenceTimings.map<LasrCtcSentenceTiming>((sentence) => ({
          text: sentence.text,
          startTokenIndex: sentence.startTokenIndex,
          endTokenIndex: sentence.endTokenIndex,
          startFrame: sentence.startFrame,
          endFrame: sentence.endFrame,
          startTime: roundTimestampSeconds(sentence.startTime),
          endTime: roundTimestampSeconds(sentence.endTime),
          duration: roundMetric(sentence.duration, 4),
          confidence: roundMetric(sentence.confidence, 4),
        })),
      },
      warnings,
    };

    emitTranscriptionProgress(options, {
      stage: 'complete',
      progress: clampProgress(1),
      elapsedMs: totalMs,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Finished transcription for ${this.modelId}.`,
      metrics: nativeTranscript.metrics,
    });

    return nativeTranscript;
  }

  dispose(): void {
    for (const handle of this.assetHandles) {
      handle.dispose();
    }
  }
}
