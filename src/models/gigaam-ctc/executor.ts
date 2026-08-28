import { normalizePcmInput } from '../../audio/index.js';
import {
  addTimesToTokenSpans,
  argmaxAndSelectedLogProbs,
  buildSentenceTimings,
  buildUtteranceTiming,
  ctcCollapseWithSpans,
  estimateSecondsPerOutputFrame,
} from '../../ctc/index.js';
import { nowMs, roundMetric, roundTimestampSeconds } from '../../runtime/timing.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import type { AssetProvider, AudioBufferLike, ResolvedAssetHandle, SpeechRuntimeHooks, TranscriptWarning } from '../../types/index.js';
import {
  createOrtSession,
  disposeOrtOutputs,
  initOrt,
  releaseOrtSession,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from '../lasr-ctc/ort.js';
import { GigaAmJsPreprocessor } from './frontend.js';
import { GigaAmTokenizer } from './tokenizer.js';
import type { GigaAmModelConfig, GigaAmModelOptions, GigaAmArtifactSource } from './types.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly session: OrtSessionLike;
  readonly tokenizer: GigaAmTokenizer;
  readonly warnings: readonly TranscriptWarning[];
}

function float32Tensor(ort: OrtModuleLike, data: Float32Array, dims: readonly number[]): OrtTensorLike {
  return new ort.Tensor('float32', data, dims);
}

function int64Tensor(ort: OrtModuleLike, value: number): OrtTensorLike {
  return new ort.Tensor('int64', BigInt64Array.from([BigInt(value)]), [1]);
}

function int64BatchTensor(ort: OrtModuleLike, values: readonly number[]): OrtTensorLike {
  return new ort.Tensor('int64', BigInt64Array.from(values, (value) => BigInt(value)), [values.length]);
}

function parseOrtTensorElementType(ortType: string | undefined, fallback: string): string {
  if (!ortType) return fallback;
  const match = /^tensor\((.+)\)$/.exec(ortType.trim());
  const elementType = (match?.[1] ?? ortType).trim();
  return elementType === 'float' ? 'float32' : elementType;
}

function getInputElementType(session: OrtSessionLike, inputName: string, fallback: string): string {
  const metadata = session.inputMetadata as
    | Record<string, { readonly type?: string; readonly name?: string }>
    | Array<{ readonly name?: string; readonly type?: string }>
    | undefined;
  if (!metadata) return fallback;
  if (Array.isArray(metadata)) {
    const found = metadata.find((entry) => entry.name === inputName);
    return parseOrtTensorElementType(found?.type, fallback);
  }
  return parseOrtTensorElementType(metadata[inputName]?.type, fallback);
}

const FLOAT32_BITS_VIEW = new Float32Array(1);
const UINT32_BITS_VIEW = new Uint32Array(FLOAT32_BITS_VIEW.buffer);

function float32ToFloat16Bits(value: number): number {
  FLOAT32_BITS_VIEW[0] = value;
  const bits = UINT32_BITS_VIEW[0] ?? 0;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x007fffff;
  if (exponent === 0xff) return mantissa !== 0 ? sign | 0x7e00 : sign | 0x7c00;
  if (exponent > 142) return sign | 0x7c00;
  if (exponent < 113) {
    if (exponent < 103) return sign;
    const shifted = (0x00800000 | mantissa) >> (114 - exponent);
    return sign | ((shifted + 1) >> 1);
  }
  return sign | ((exponent - 112) << 10) | ((mantissa + 0x00001000) >> 13);
}

function float32ToFloat16Array(values: Float32Array): Uint16Array {
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = float32ToFloat16Bits(values[index] ?? 0);
  }
  return output;
}

function featuresTensor(
  ort: OrtModuleLike,
  session: OrtSessionLike,
  data: Float32Array,
  dims: readonly number[],
): OrtTensorLike {
  if (getInputElementType(session, 'features', 'float32') === 'float16') {
    return new ort.Tensor('float16', float32ToFloat16Array(data), dims);
  }
  return float32Tensor(ort, data, dims);
}

function findOutput(outputs: Record<string, OrtTensorLike>): OrtTensorLike {
  const output = outputs.log_probs ?? outputs.logprobs ?? outputs.logits ?? Object.values(outputs)[0];
  if (!output) throw new Error('GigaAM CTC graph returned no logits output.');
  return output;
}

function readEncodedLength(outputs: Record<string, OrtTensorLike>, fallback: number): number {
  const tensor = outputs.encoded_lengths ?? outputs.encoded_len;
  const raw = tensor?.data as ArrayLike<number | bigint> | undefined;
  const value = raw && raw.length > 0 ? Number(raw[0]) : Number.NaN;
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
}

function readTensor(tensor: OrtTensorLike): Float32Array {
  if (tensor.type !== 'float16') {
    return tensor.data instanceof Float32Array
      ? new Float32Array(tensor.data)
      : Float32Array.from(tensor.data as unknown as ArrayLike<number>);
  }
  const source = tensor.data as unknown as ArrayLike<number>;
  const result = new Float32Array(source.length);
  for (let index = 0; index < source.length; index += 1) {
    const bits = Number(source[index] ?? 0);
    const sign = (bits & 0x8000) !== 0 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) {
      result[index] = mantissa === 0 ? (sign < 0 ? -0 : 0) : sign * (mantissa / 1024) * 2 ** -14;
    } else if (exponent === 0x1f) {
      result[index] = mantissa === 0 ? (sign < 0 ? -Infinity : Infinity) : NaN;
    } else {
      result[index] = sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
    }
  }
  return result;
}

export class OrtGigaAmCtcExecutor {
  private readonly source?: GigaAmArtifactSource;
  private readonly preprocessor: GigaAmJsPreprocessor;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private readonly loadStatePromise?: Promise<LoadedState>;
  private disposed = false;
  private disposePromise?: Promise<void>;

  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    private readonly config: GigaAmModelConfig,
    options: GigaAmModelOptions | undefined,
    dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks; readonly signal?: import('../../types/index.js').AbortSignalLike | null } = {},
  ) {
    this.source = options?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    this.signal = dependencies.signal;
    this.preprocessor = new GigaAmJsPreprocessor({
      nMels: config.nMels,
      nFft: config.nFft,
      winLength: config.winLength,
      hopLength: config.hopLength,
      center: config.center ?? false,
    });
    if (this.source) this.loadStatePromise = this.initialize();
  }

  private async resolveAsset(source: Extract<GigaAmArtifactSource, { kind: 'huggingface' }>, filename: string): Promise<string> {
    if (!this.assetProvider) {
      const revision = source.revision ?? 'main';
      return `https://huggingface.co/${source.repoId}/resolve/${encodeURIComponent(revision)}/${filename}`;
    }
    const handle = await this.assetProvider.resolve({
      id: `huggingface:${source.repoId}:${source.revision ?? 'main'}:${filename}`,
      provider: 'huggingface', repoId: source.repoId, revision: source.revision ?? 'main', filename,
      cacheKey: `huggingface:${source.repoId}:${source.revision ?? 'main'}:${filename}`,
      onProgress: (event) => this.runtimeHooks?.onProgress?.({
        phase: 'asset:download', modelId: this.modelId, file: filename,
        loaded: event.loaded, total: event.total,
        percent: event.total ? Math.round((event.loaded / event.total) * 100) : event.done && !event.aborted ? 100 : undefined,
        isComplete: Boolean(event.done) && !event.aborted,
        aborted: event.aborted,
        message: event.aborted ? `Cancelled ${filename}.` : event.done ? `Prepared ${filename}.` : `Downloading ${filename}.`,
      }),
    });
    this.assetHandles.push(handle);
    const locator = await handle.getLocator('url');
    if (!locator) throw new Error(`Could not create a URL locator for "${filename}".`);
    return locator;
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw createExperimentalArtifactMissingError('gigaam-ctc', this.modelId);
    let modelUrl: string;
    let tokenizerUrl: string;
    let modelDataUrl: string | undefined;
    let modelDataFilename: string | undefined;
    if (this.source.kind === 'direct') {
      modelUrl = this.source.artifacts.modelUrl;
      tokenizerUrl = this.source.artifacts.tokenizerUrl;
      modelDataUrl = this.source.artifacts.modelDataUrl;
      modelDataFilename = this.source.artifacts.modelDataFilename;
    } else {
      const modelFilename = this.source.modelFilename ?? 'multilingual_ctc.onnx';
      const tokenizerFilename = this.source.tokenizerFilename ?? 'multilingual_vocab.txt';
      modelUrl = await this.resolveAsset(this.source, modelFilename);
      tokenizerUrl = await this.resolveAsset(this.source, tokenizerFilename);
      modelDataFilename = this.source.modelDataFilename;
      if (modelDataFilename) modelDataUrl = await this.resolveAsset(this.source, modelDataFilename);
    }
    const ort = await initOrt(this.backendId, {
      wasmPaths: this.source.wasmPaths,
      cpuThreads: this.source.cpuThreads,
      signal: this.signal,
    });
    const session = await createOrtSession(ort, modelUrl, {
      backendId: this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm',
      enableProfiling: this.source.enableProfiling,
      externalDataUrl: modelDataUrl,
      externalDataPath: modelDataFilename,
      signal: this.signal,
    });
    if (this.disposed) {
      releaseOrtSession(session);
      throw new Error(`GigaAM CTC executor was disposed during load for "${this.modelId}".`);
    }
    const tokenizer = await GigaAmTokenizer.fromUrl(tokenizerUrl, this.signal);
    return { ort, session, tokenizer, warnings: [] };
  }

  async ready(): Promise<void> {
    if (!this.loadStatePromise) throw createExperimentalArtifactMissingError('gigaam-ctc', this.modelId);
    await this.loadStatePromise;
  }

  async transcribe(audioInput: AudioBufferLike, options: LasrCtcTranscriptionOptions = {}): Promise<LasrCtcNativeTranscript> {
    if (this.disposed) throw new Error(`GigaAM CTC executor is disposed for "${this.modelId}".`);
    const state = await this.loadStatePromise;
    if (!state) throw createExperimentalArtifactMissingError('gigaam-ctc', this.modelId);
    const audio = normalizePcmInput(audioInput).toMono();
    const started = nowMs();
    const features = this.preprocessor.process(audio.channels[0] ?? new Float32Array(0));
    if (features.frameCount <= 0) return { utteranceText: '', isFinal: true, warnings: [...state.warnings] };
    const featureTensor = featuresTensor(state.ort, state.session, features.features, [1, features.featureSize, features.frameCount]);
    const lengthTensor = int64Tensor(state.ort, features.frameCount);
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ features: featureTensor, feature_lengths: lengthTensor });
    } finally {
      featureTensor.dispose?.(); lengthTensor.dispose?.();
    }
    try {
      return this.decodeOne(
        state,
        audio,
        features.frameCount,
        findOutput(outputs),
        options,
        started,
        readEncodedLength(outputs, 0),
      );
    } finally {
      disposeOrtOutputs(outputs);
    }
  }

  /** Run mixed-length inputs through one padded GigaAM CTC graph call. */
  async transcribeBatch(
    audioInputs: readonly AudioBufferLike[],
    options: LasrCtcTranscriptionOptions = {},
  ): Promise<readonly LasrCtcNativeTranscript[]> {
    if (this.disposed) throw new Error(`GigaAM CTC executor is disposed for "${this.modelId}".`);
    if (audioInputs.length === 0) return [];
    const state = await this.loadStatePromise;
    if (!state) throw createExperimentalArtifactMissingError('gigaam-ctc', this.modelId);
    const started = nowMs();
    const audios = audioInputs.map((input) => normalizePcmInput(input).toMono());
    const prepared = audios.map((audio) => this.preprocessor.process(audio.channels[0] ?? new Float32Array(0)));
    const maxFrames = Math.max(...prepared.map((item) => item.frameCount));
    if (maxFrames <= 0) return audios.map(() => ({ utteranceText: '', isFinal: true, warnings: [...state.warnings] }));
    const lengths = prepared.map((item) => item.frameCount);
    const batchFeatures = new Float32Array(audios.length * this.config.nMels * maxFrames);
    prepared.forEach((item, batchIndex) => {
      for (let mel = 0; mel < this.config.nMels; mel += 1) {
        const sourceOffset = mel * item.frameCount;
        const targetOffset = batchIndex * this.config.nMels * maxFrames + mel * maxFrames;
        batchFeatures.set(item.features.subarray(sourceOffset, sourceOffset + item.frameCount), targetOffset);
      }
    });
    const features = featuresTensor(state.ort, state.session, batchFeatures, [audios.length, this.config.nMels, maxFrames]);
    const featureLengths = int64BatchTensor(state.ort, lengths);
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ features, feature_lengths: featureLengths });
    } finally {
      features.dispose?.(); featureLengths.dispose?.();
    }
    try {
      const logitsTensor = findOutput(outputs);
      const encodedLengths = outputs.encoded_lengths ?? outputs.encoded_len;
      const encodedLengthData = encodedLengths?.data as ArrayLike<number | bigint> | undefined;
      const encodedLengthValues = Array.from({ length: audios.length }, (_, batchIndex) =>
        encodedLengthData && encodedLengthData.length > batchIndex
          ? Number(encodedLengthData[batchIndex])
          : Number.NaN,
      );
      const dims = [...logitsTensor.dims];
      if (dims.length !== 3 || dims[0] !== audios.length) throw new Error(`Unexpected GigaAM batch logits shape: [${dims.join(', ')}].`);
      const outFrames = dims[1] ?? 0;
      const vocabSize = dims[2] ?? 0;
      const allLogits = readTensor(logitsTensor);
      const perItemMs = nowMs() - started;
      return audios.map((audio, batchIndex) => {
        const encodedLength = encodedLengthValues[batchIndex] ?? Number.NaN;
        const itemOutFrames = Math.min(
          outFrames,
          Number.isFinite(encodedLength) && encodedLength > 0
            ? Math.floor(encodedLength)
            : Math.max(0, Math.floor((lengths[batchIndex]! - 1) / this.config.rawStride) + 1),
        );
        const offset = batchIndex * outFrames * vocabSize;
        const logits = allLogits.subarray(offset, offset + itemOutFrames * vocabSize);
        return this.decodeLogits(state, audio, lengths[batchIndex]!, logits, itemOutFrames, vocabSize, options, started, perItemMs);
      });
    } finally {
      disposeOrtOutputs(outputs);
    }
  }

  private decodeOne(
    state: LoadedState,
    audio: ReturnType<ReturnType<typeof normalizePcmInput>['toMono']>,
    inputFrames: number,
    logitsTensor: OrtTensorLike,
    options: LasrCtcTranscriptionOptions,
    started: number,
    encodedLength = 0,
  ): LasrCtcNativeTranscript {
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) throw new Error(`Unexpected GigaAM logits shape: [${dims.join(', ')}].`);
    const vocabSize = dims[2] ?? 0;
    const rawFrames = dims[1] ?? 0;
    const outFrames = encodedLength > 0 ? Math.min(rawFrames, encodedLength) : rawFrames;
    return this.decodeLogits(state, audio, inputFrames, readTensor(logitsTensor).subarray(0, outFrames * vocabSize), outFrames, vocabSize, options, started, nowMs() - started);
  }

  private decodeLogits(
    state: LoadedState,
    audio: ReturnType<ReturnType<typeof normalizePcmInput>['toMono']>,
    inputFrames: number,
    logits: Float32Array,
    outFrames: number,
    vocabSize: number,
    options: LasrCtcTranscriptionOptions,
    started: number,
    elapsedMs: number,
  ): LasrCtcNativeTranscript {
    const { frameIds, selectedLogProbs } = argmaxAndSelectedLogProbs(logits, outFrames, vocabSize);
    const { collapsedIds, tokenSpans } = ctcCollapseWithSpans(frameIds, selectedLogProbs, state.tokenizer.blankId);
    const text = state.tokenizer.decode(collapsedIds);
    const secondsPerFrame = estimateSecondsPerOutputFrame({ audioDurationSec: audio.durationSeconds, inputFrames, inputFrameHopSeconds: this.config.featureHopSeconds, outFrames });
    const timedSpans = addTimesToTokenSpans(state.tokenizer, tokenSpans, secondsPerFrame);
    const utterance = buildUtteranceTiming(frameIds, selectedLogProbs, state.tokenizer.blankId, secondsPerFrame);
    const sentences = buildSentenceTimings(text, state.tokenizer, collapsedIds, timedSpans);
    const tokens = timedSpans.map((span, index) => ({ index, id: options.returnTokenIds ? span.tokenId : undefined, text: span.text, startTime: roundTimestampSeconds(span.startTime), endTime: roundTimestampSeconds(span.endTime), confidence: roundMetric(span.confidence, 4), logitIndex: options.returnLogitIndices ? span.startFrame : undefined }));
    const totalMs = elapsedMs || nowMs() - started;
    return {
      utteranceText: text, isFinal: true, tokens,
      confidence: { utterance: utterance.confidence, tokenAverage: utterance.confidence, wordAverage: sentences.length ? sentences.reduce((sum, item) => sum + item.confidence, 0) / sentences.length : 0 },
      metrics: { preprocessMs: 0, encodeMs: roundMetric(totalMs), decodeMs: 0, totalMs: roundMetric(totalMs), wallMs: roundMetric(totalMs), audioDurationSec: roundMetric(audio.durationSeconds, 4), rtf: audio.durationSeconds ? roundMetric(totalMs / (audio.durationSeconds * 1000), 4) : 0, rtfx: audio.durationSeconds ? roundMetric(audio.durationSeconds / (totalMs / 1000), 4) : undefined, preprocessorBackend: 'js', encoderFrameCount: outFrames, decodeIterations: outFrames, emittedTokenCount: tokens.length, emittedWordCount: sentences.length },
      ctc: { frameIds: options.returnFrameIds ? frameIds : undefined, collapsedIds: options.returnTokenIds ? collapsedIds : undefined, secondsPerFrame, utterance, tokenSpans: timedSpans, sentences },
      warnings: [...state.warnings],
    };
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = this.flushDispose();
    return this.disposePromise;
  }

  private async flushDispose(): Promise<void> {
    if (this.loadStatePromise) {
      try {
        const loaded = await this.loadStatePromise;
        releaseOrtSession(loaded.session);
      } catch {
        // Keep the original load error; still drop asset handles.
      }
    }
    const handles = this.assetHandles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
