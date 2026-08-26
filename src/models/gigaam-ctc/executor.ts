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
import type { AssetProvider, AudioBufferLike, ResolvedAssetHandle, SpeechRuntimeHooks, TranscriptWarning } from '../../types/index.js';
import {
  createOrtSession,
  initOrt,
  type OrtModuleLike,
  type OrtSessionLike,
  type OrtTensorLike,
} from '../lasr-ctc/ort.js';
import { MedAsrTextTokenizer } from '../lasr-ctc/tokenizer.js';
import { GigaAmJsPreprocessor } from './frontend.js';
import type { GigaAmModelConfig, GigaAmModelOptions, GigaAmArtifactSource } from './types.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';

interface LoadedState {
  readonly ort: OrtModuleLike;
  readonly session: OrtSessionLike;
  readonly tokenizer: MedAsrTextTokenizer;
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

function findOutput(outputs: Record<string, OrtTensorLike>): OrtTensorLike {
  const output = outputs.log_probs ?? outputs.logprobs ?? outputs.logits ?? Object.values(outputs)[0];
  if (!output) throw new Error('GigaAM CTC graph returned no logits output.');
  return output;
}

function readTensor(tensor: OrtTensorLike): Float32Array {
  if (tensor.type !== 'float16') {
    return tensor.data instanceof Float32Array
      ? tensor.data
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
  private readonly preprocessor = new GigaAmJsPreprocessor();
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private readonly loadStatePromise?: Promise<LoadedState>;

  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    private readonly config: GigaAmModelConfig,
    options: GigaAmModelOptions | undefined,
    dependencies: { readonly assetProvider?: AssetProvider; readonly runtimeHooks?: SpeechRuntimeHooks } = {},
  ) {
    this.source = options?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
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
        percent: event.total ? Math.round((event.loaded / event.total) * 100) : event.done ? 100 : undefined,
        isComplete: event.done, message: event.done ? `Prepared ${filename}.` : `Downloading ${filename}.`,
      }),
    });
    this.assetHandles.push(handle);
    const locator = await handle.getLocator('url');
    if (!locator) throw new Error(`Could not create a URL locator for "${filename}".`);
    return locator;
  }

  private async initialize(): Promise<LoadedState> {
    if (!this.source) throw new Error(`No GigaAM artifact source is configured for "${this.modelId}".`);
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
    });
    const session = await createOrtSession(ort, modelUrl, {
      backendId: this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm',
      enableProfiling: this.source.enableProfiling,
      externalDataUrl: modelDataUrl,
      externalDataPath: modelDataFilename,
    });
    const tokenizer = await MedAsrTextTokenizer.fromUrl(tokenizerUrl);
    return { ort, session, tokenizer, warnings: [] };
  }

  async ready(): Promise<void> {
    if (!this.loadStatePromise) throw new Error(`No GigaAM artifact source is configured for "${this.modelId}".`);
    await this.loadStatePromise;
  }

  async transcribe(audioInput: AudioBufferLike, options: LasrCtcTranscriptionOptions = {}): Promise<LasrCtcNativeTranscript> {
    const state = await this.loadStatePromise;
    if (!state) throw new Error(`No GigaAM artifact source is configured for "${this.modelId}".`);
    const audio = normalizePcmInput(audioInput).toMono();
    const started = nowMs();
    const features = this.preprocessor.process(audio.channels[0] ?? new Float32Array(0));
    if (features.frameCount <= 0) return { utteranceText: '', isFinal: true, warnings: [...state.warnings] };
    const featureTensor = float32Tensor(state.ort, features.features, [1, features.featureSize, features.frameCount]);
    const lengthTensor = int64Tensor(state.ort, features.frameCount);
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ features: featureTensor, feature_lengths: lengthTensor });
    } finally {
      featureTensor.dispose?.(); lengthTensor.dispose?.();
    }
    return this.decodeOne(state, audio, features.frameCount, findOutput(outputs), options, started);
  }

  /** Run mixed-length inputs through one padded GigaAM CTC graph call. */
  async transcribeBatch(
    audioInputs: readonly AudioBufferLike[],
    options: LasrCtcTranscriptionOptions = {},
  ): Promise<readonly LasrCtcNativeTranscript[]> {
    const state = await this.loadStatePromise;
    if (!state) throw new Error(`No GigaAM artifact source is configured for "${this.modelId}".`);
    if (audioInputs.length === 0) return [];
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
    const features = float32Tensor(state.ort, batchFeatures, [audios.length, this.config.nMels, maxFrames]);
    const featureLengths = int64BatchTensor(state.ort, lengths);
    let outputs: Record<string, OrtTensorLike>;
    try {
      outputs = await state.session.run({ features, feature_lengths: featureLengths });
    } finally {
      features.dispose?.(); featureLengths.dispose?.();
    }
    const logitsTensor = findOutput(outputs);
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== audios.length) throw new Error(`Unexpected GigaAM batch logits shape: [${dims.join(', ')}].`);
    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
    const allLogits = readTensor(logitsTensor);
    const perItemMs = nowMs() - started;
    return audios.map((audio, batchIndex) => {
      const itemOutFrames = Math.min(outFrames, Math.max(0, Math.floor((lengths[batchIndex]! - 1) / this.config.rawStride) + 1));
      const offset = batchIndex * outFrames * vocabSize;
      const logits = allLogits.subarray(offset, offset + itemOutFrames * vocabSize);
      return this.decodeLogits(state, audio, lengths[batchIndex]!, logits, itemOutFrames, vocabSize, options, started, perItemMs);
    });
  }

  private decodeOne(
    state: LoadedState,
    audio: ReturnType<ReturnType<typeof normalizePcmInput>['toMono']>,
    inputFrames: number,
    logitsTensor: OrtTensorLike,
    options: LasrCtcTranscriptionOptions,
    started: number,
  ): LasrCtcNativeTranscript {
    const dims = [...logitsTensor.dims];
    if (dims.length !== 3 || dims[0] !== 1) throw new Error(`Unexpected GigaAM logits shape: [${dims.join(', ')}].`);
    const outFrames = dims[1] ?? 0;
    const vocabSize = dims[2] ?? 0;
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

  dispose(): void {
    for (const handle of this.assetHandles) void handle.dispose();
  }
}
