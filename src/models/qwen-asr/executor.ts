import type {
  AbortSignalLike,
  AssetProvider,
  AudioBufferLike,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptWarning,
  TranscriptMetrics,
  TranscriptionProgressEvent,
} from '../../types/index.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import { PipelineAbortedError } from '../../pipeline/composition.js';
import { fetchModelFiles } from '../../runtime/huggingface.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import {
  createQwenOrtSession,
  initQwenOrt,
  releaseQwenOrtSession,
  resolveQwen3AsrArtifacts,
  type QwenOrtModuleLike,
  type QwenOrtSessionLike,
  type QwenOrtTensorLike,
  type ResolvedQwen3AsrArtifacts,
} from './ort.js';
import { normalizeQwenLanguage } from './config.js';
import { applyOfficialQwen3AsrGraphDefaults } from './official.js';
import { Qwen3AsrFeatureProcessor, getQwenAudioTokenCount } from './processor.js';
import { Qwen3AsrTokenizer } from './tokenizer.js';
import type {
  Qwen3AsrExecutor,
  Qwen3AsrFeatureResult,
  Qwen3AsrModelConfig,
  Qwen3AsrModelOptions,
  Qwen3AsrNativeToken,
  Qwen3AsrNativeTranscript,
  Qwen3AsrTranscriptionOptions,
  QwenCacheOutputLocation,
} from './types.js';

interface LoadedQwenState {
  readonly ort: QwenOrtModuleLike;
  readonly tokenizer: Qwen3AsrTokenizer;
  readonly processor: Qwen3AsrFeatureProcessor;
  encoderSession?: QwenOrtSessionLike;
  decoderSession?: QwenOrtSessionLike;
  decoderStepSession?: QwenOrtSessionLike;
  readonly resolved: ResolvedQwen3AsrArtifacts;
  readonly warnings: TranscriptWarning[];
  readonly sequentialSessions: boolean;
}

interface PromptTensors {
  readonly inputIds: Int32Array;
  readonly audioEmbeddings: Uint16Array;
  readonly audioMask: Uint16Array;
  readonly attentionMask: Uint16Array;
  readonly positionIds: Int32Array;
  readonly audioTokenPositions: readonly number[];
}

function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: { readonly loaded: number; readonly total?: number; readonly done?: boolean; readonly aborted?: boolean },
): RuntimeProgressEvent {
  const percent = event.total && event.total > 0
    ? Math.min(100, Math.round((event.loaded / event.total) * 100))
    : event.done && !event.aborted
      ? 100
      : undefined;
  return {
    phase: 'asset:download',
    modelId,
    file,
    loaded: event.loaded,
    total: event.total,
    percent,
    loadedMiB: roundMetric(event.loaded / (1024 * 1024), 2),
    totalMiB: event.total === undefined ? undefined : roundMetric(event.total / (1024 * 1024), 2),
    isComplete: Boolean(event.done) && !event.aborted,
    aborted: event.aborted,
    message: event.aborted ? `Cancelled ${file}.` : event.done ? `Prepared ${file}.` : `Downloading ${file}.`,
  };
}

function normalizeRepoPath(path: string): string {
  return String(path || '').replace(/^\.\/+/, '').replace(/\\/g, '/');
}

function hasListedRepoFile(files: readonly string[], filename: string): boolean {
  const target = normalizeRepoPath(filename);
  return files.some((path) => normalizeRepoPath(path) === target || normalizeRepoPath(path).endsWith(`/${target}`));
}

function isMissingAssetError(error: unknown): boolean {
  return error instanceof Error && (/\b404\b/.test(error.message) || /not found/i.test(error.message));
}

function float32ToFloat16Bits(input: Float32Array): Uint16Array {
  const output = new Uint16Array(input.length);
  const view = new DataView(new ArrayBuffer(4));
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index] as number;
    if (Number.isNaN(value)) {
      output[index] = 0x7e00;
      continue;
    }
    if (value === Infinity) {
      output[index] = 0x7c00;
      continue;
    }
    if (value === -Infinity) {
      output[index] = 0xfc00;
      continue;
    }
    view.setFloat32(0, value, false);
    const bits = view.getUint32(0, false);
    const sign = (bits >>> 16) & 0x8000;
    const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
    const fraction = bits & 0x7fffff;
    if (exponent <= 0) {
      if (exponent < -10) {
        output[index] = sign;
      } else {
        const mantissa = (fraction | 0x800000) >>> (1 - exponent);
        output[index] = sign | ((mantissa + 0x1000) >>> 13);
      }
    } else if (exponent >= 31) {
      output[index] = sign | 0x7c00;
    } else {
      output[index] = sign | (exponent << 10) | ((fraction + 0x1000) >>> 13);
    }
  }
  return output;
}

function float16BitsToFloat32(input: Uint16Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const bits = input[index] as number;
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0) {
      output[index] = fraction === 0 ? sign * 0 : sign * (fraction / 1024) * 2 ** -14;
    } else if (exponent === 31) {
      output[index] = fraction === 0 ? sign * Infinity : Number.NaN;
    } else {
      output[index] = sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
    }
  }
  return output;
}

function asFloat32(data: ArrayBufferView, type?: string): Float32Array {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array || type === 'float16') {
    return float16BitsToFloat32(data instanceof Uint16Array ? data : new Uint16Array(data.buffer, data.byteOffset, Math.floor(data.byteLength / 2)));
  }
  if (data instanceof Float64Array) return Float32Array.from(data);
  if (data instanceof Int32Array) return Float32Array.from(data);
  return Float32Array.from(data as unknown as ArrayLike<number>);
}

async function readTensorData(tensor: QwenOrtTensorLike): Promise<ArrayBufferView> {
  if (tensor.getData) return tensor.getData(true);
  return tensor.data;
}

/** Owned float32 copy so Ort logits can be disposed without freeing the argmax view. */
export async function copyQwenLogits(tensor: QwenOrtTensorLike): Promise<Float32Array> {
  return new Float32Array(asFloat32(await readTensorData(tensor), tensor.type));
}

function disposeTensor(tensor: QwenOrtTensorLike | undefined): void {
  tensor?.dispose?.();
}

function disposeTensorRecord(tensors: Record<string, QwenOrtTensorLike>): void {
  for (const tensor of Object.values(tensors)) disposeTensor(tensor);
}

function tokenId(tokenizer: Qwen3AsrTokenizer, token: string, fallback: number): number {
  return tokenizer.getTokenId(token) ?? fallback;
}

function toInt64(ids: ArrayLike<number>): BigInt64Array {
  const output = new BigInt64Array(ids.length);
  for (let index = 0; index < ids.length; index += 1) output[index] = BigInt(ids[index] as number);
  return output;
}

function sessionInputType(session: QwenOrtSessionLike, name: string, fallback: 'float16' | 'float32'): 'float16' | 'float32' {
  const meta = session.inputMetadata?.find((item) => item.name === name);
  const raw = `${meta?.type ?? ''}`.toLowerCase();
  if (raw.includes('float16')) return 'float16';
  if (raw.includes('float')) return 'float32';
  return fallback;
}

function floatFeed(
  ort: QwenOrtModuleLike,
  session: QwenOrtSessionLike,
  name: string,
  data: Float32Array,
  dims: readonly number[],
  fallback: 'float16' | 'float32' = 'float32',
): QwenOrtTensorLike {
  const type = sessionInputType(session, name, fallback);
  if (type === 'float16') return new ort.Tensor('float16', float32ToFloat16Bits(data), dims);
  return new ort.Tensor('float32', data, dims);
}

function isStackedKv(config: Qwen3AsrModelConfig): boolean {
  return config.graph.kvLayout === 'stacked';
}

function buildPrompt(
  tokenizer: Qwen3AsrTokenizer,
  graph: Qwen3AsrModelConfig['graph'],
  audioTokenCount: number,
  language: string | undefined,
  context: string,
): PromptTensors {
  const imStart = tokenId(tokenizer, '<|im_start|>', graph.imStartTokenId);
  const imEnd = tokenId(tokenizer, '<|im_end|>', graph.imEndTokenId);
  const audioStart = tokenId(tokenizer, '<|audio_start|>', graph.audioStartTokenId);
  const audioEnd = tokenId(tokenizer, '<|audio_end|>', graph.audioEndTokenId);
  const audioPad = tokenId(tokenizer, '<|audio_pad|>', graph.audioPadTokenId);
  const systemText = `<|im_start|>system\n${context}<|im_end|>\n<|im_start|>user\n`;
  const ids = [...tokenizer.encode(systemText)];
  if (ids.length === 0 || ids[0] !== imStart) ids.unshift(imStart);
  ids.push(audioStart);
  const audioTokenPositions: number[] = [];
  for (let index = 0; index < audioTokenCount; index += 1) {
    audioTokenPositions.push(ids.length);
    ids.push(audioPad);
  }
  ids.push(audioEnd, imEnd, ...tokenizer.encode('\n<|im_start|>assistant\n'));
  if (language) ids.push(...tokenizer.encode(`language ${language}<asr_text>`));

  const sequenceLength = ids.length;
  const audioEmbeddings = new Uint16Array(sequenceLength * graph.hiddenSize);
  const audioMask = new Uint16Array(sequenceLength);
  const attentionMask = new Uint16Array(sequenceLength * (graph.pastSeedLength + sequenceLength));
  const attentionMaskF32 = new Float32Array(attentionMask.length);
  attentionMaskF32.fill(graph.pastSeedAttentionMask);
  for (let query = 0; query < sequenceLength; query += 1) {
    const rowOffset = query * (graph.pastSeedLength + sequenceLength);
    for (let key = graph.pastSeedLength; key <= graph.pastSeedLength + query; key += 1) {
      attentionMaskF32[rowOffset + key] = 0;
    }
  }
  attentionMask.set(float32ToFloat16Bits(attentionMaskF32));
  for (const position of audioTokenPositions) audioMask[position] = 0x3c00;

  return {
    inputIds: Int32Array.from(ids),
    audioEmbeddings,
    audioMask,
    attentionMask,
    positionIds: Int32Array.from({ length: sequenceLength }, (_, index) => index + graph.pastSeedLength),
    audioTokenPositions,
  };
}

function outputLocationMap(
  config: Qwen3AsrModelConfig,
  location: QwenCacheOutputLocation,
): Record<string, QwenCacheOutputLocation> {
  const map: Record<string, QwenCacheOutputLocation> = { logits: 'cpu' };
  for (let layer = 0; layer < config.graph.numLayers; layer += 1) {
    map[`present.${layer}.key`] = location;
    map[`present.${layer}.value`] = location;
  }
  return map;
}

function parseQwenOutput(
  rawText: string,
  forcedLanguage: string | undefined,
): { readonly language?: string; readonly text: string } {
  const text = rawText.trim();
  if (forcedLanguage) return { language: forcedLanguage, text };
  const match = text.match(/^language\s+([^<\r\n]+)<asr_text>([\s\S]*)$/i);
  if (match) {
    return {
      language: normalizeQwenLanguage(match[1]?.trim()),
      text: match[2]?.trim() ?? '',
    };
  }
  const glued = text.match(/^language\s+([A-Za-z]+)(?=[A-Z“"‘])([\s\S]*)$/);
  if (glued) {
    return {
      language: normalizeQwenLanguage(glued[1]),
      text: glued[2]?.trim() ?? '',
    };
  }
  return { text };
}

function argmaxLastRow(logits: Float32Array, vocabularySize: number): number {
  const rowStart = Math.max(0, logits.length - vocabularySize);
  let bestId = 0;
  let bestValue = -Infinity;
  for (let id = 0; id < vocabularySize && rowStart + id < logits.length; id += 1) {
    const value = logits[rowStart + id] as number;
    if (value > bestValue) {
      bestValue = value;
      bestId = id;
    }
  }
  return bestId;
}

function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('decode');
  }
}

function emitProgress(
  options: Qwen3AsrTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

export class OrtQwen3AsrExecutor implements Qwen3AsrExecutor {
  private readonly sourceOptions: Qwen3AsrModelOptions['source'];
  private readonly loadStatePromise?: Promise<LoadedQwenState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly signal?: import('../../types/index.js').AbortSignalLike | null;
  private readonly dependencies: {
    readonly tokenizer?: Qwen3AsrTokenizer;
    readonly featureProcessor?: Qwen3AsrFeatureProcessor;
    readonly ort?: QwenOrtModuleLike;
    readonly encoderSession?: QwenOrtSessionLike;
    readonly decoderSession?: QwenOrtSessionLike;
    readonly decoderStepSession?: QwenOrtSessionLike;
  };
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private readonly config: Qwen3AsrModelConfig;
  private disposed = false;
  private disposePromise?: Promise<void>;

  constructor(
    private readonly modelId: string,
    config: Qwen3AsrModelConfig,
    private readonly backendId: string,
    loadOptions: Qwen3AsrModelOptions | undefined,
    dependencies: {
      readonly assetProvider?: AssetProvider;
      readonly runtimeHooks?: SpeechRuntimeHooks;
      readonly tokenizer?: Qwen3AsrTokenizer;
      readonly featureProcessor?: Qwen3AsrFeatureProcessor;
      /** Test/reference injection point; production loads ORT sessions from the source. */
      readonly ort?: QwenOrtModuleLike;
      readonly encoderSession?: QwenOrtSessionLike;
      readonly decoderSession?: QwenOrtSessionLike;
      readonly decoderStepSession?: QwenOrtSessionLike;
      readonly signal?: import('../../types/index.js').AbortSignalLike | null;
    } = {},
  ) {
    this.config = applyOfficialQwen3AsrGraphDefaults(config, loadOptions?.source);
    this.sourceOptions = loadOptions?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    this.signal = dependencies.signal;
    this.dependencies = {
      tokenizer: dependencies.tokenizer,
      featureProcessor: dependencies.featureProcessor,
      ort: dependencies.ort,
      encoderSession: dependencies.encoderSession,
      decoderSession: dependencies.decoderSession,
      decoderStepSession: dependencies.decoderStepSession,
    };
    if (this.sourceOptions || (dependencies.ort && dependencies.encoderSession && dependencies.decoderSession)) {
      this.loadStatePromise = this.initialize();
    }
  }

  private async materializeHuggingFaceArtifacts(
    resolved: ResolvedQwen3AsrArtifacts,
  ): Promise<ResolvedQwen3AsrArtifacts['artifacts']> {
    const source = this.sourceOptions;
    if (!this.assetProvider || !source || source.kind !== 'huggingface') return resolved.artifacts;

    const revision = source.revision ?? 'main';
    let filesPromise: Promise<readonly string[]> | undefined;
    const getFiles = (): Promise<readonly string[]> => {
      filesPromise ??= fetchModelFiles(source.repoId, revision);
      return filesPromise;
    };
    const resolveFile = async (filename: string, optional = false): Promise<string | undefined> => {
      if (optional) {
        const files = await getFiles();
        if (files.length > 0 && !hasListedRepoFile(files, filename)) return undefined;
      }
      const cacheKey = `huggingface:${source.repoId}:${revision}:${filename}`;
      const cacheKeyFallbacks = (source.cacheKeyFallbackRevisions ?? [])
        .filter((fallback) => fallback !== revision)
        .map((fallback) => `huggingface:${source.repoId}:${fallback}:${filename}`);
      try {
        const handle = await this.assetProvider!.resolve({
          id: cacheKey,
          provider: 'huggingface',
          repoId: source.repoId,
          revision,
          filename,
          preferBlobUrl: true,
          cacheKey,
          cacheKeyFallbacks,
          onProgress: (event) => this.runtimeHooks?.onProgress?.(
            createAssetProgressEvent(this.modelId, filename, event),
          ),
        });
        this.assetHandles.push(handle);
        const locator = await handle.getLocator('url');
        if (!locator) throw new Error(`Could not create a URL locator for ${filename}.`);
        return locator;
      } catch (error) {
        if (optional && isMissingAssetError(error)) return undefined;
        throw error;
      }
    };

    const encoderPath = source.encoderPath ?? 'onnx/audio_encoder_fp16.onnx';
    const decoderPath = source.decoderPath ?? 'onnx/decoder_with_past_fp16.onnx';
    const tokenizerPath = source.tokenizerPath ?? 'processor/tokenizer.json';
    const encoderDataFile = source.encoderDataPath ?? 'audio_encoder_fp16.onnx_data';
    const decoderDataFile = source.decoderDataPath ?? 'decoder_with_past_fp16.onnx_data';
    const encoderDataRepoPath = encoderDataFile.includes('/') ? encoderDataFile : `onnx/${encoderDataFile}`;
    const decoderDataRepoPath = decoderDataFile.includes('/') ? decoderDataFile : `onnx/${decoderDataFile}`;
    const [encoderUrl, decoderUrl, tokenizerUrl, encoderDataUrl, decoderDataUrl] = await Promise.all([
      resolveFile(encoderPath),
      resolveFile(decoderPath),
      resolveFile(tokenizerPath),
      resolveFile(encoderDataRepoPath),
      resolveFile(decoderDataRepoPath),
    ]);
    return {
      ...resolved.artifacts,
      encoderUrl: encoderUrl ?? resolved.artifacts.encoderUrl,
      decoderUrl: decoderUrl ?? resolved.artifacts.decoderUrl,
      tokenizerUrl: tokenizerUrl ?? resolved.artifacts.tokenizerUrl,
      encoderDataUrl: encoderDataUrl ?? resolved.artifacts.encoderDataUrl,
      decoderDataUrl: decoderDataUrl ?? resolved.artifacts.decoderDataUrl,
    };
  }

  private async initialize(): Promise<LoadedQwenState> {
    if (!this.sourceOptions) throw createExperimentalArtifactMissingError('qwen-asr', this.modelId);
    const resolved = resolveQwen3AsrArtifacts(this.sourceOptions, this.backendId);
    const artifacts = await this.materializeHuggingFaceArtifacts(resolved);
    const ort = this.dependencies.ort ?? await initQwenOrt(resolved.ortBackend, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
      enableProfiling: resolved.enableProfiling,
      signal: this.signal,
    });
    const tokenizer = this.dependencies.tokenizer ?? await Qwen3AsrTokenizer.fromUrl(artifacts.tokenizerUrl, this.signal);
    const processor = this.dependencies.featureProcessor ?? new Qwen3AsrFeatureProcessor(this.config);
    const warnings: TranscriptWarning[] = [];
    const cacheLocation = resolved.decoderBackendForOrt === 'webgpu'
      ? resolved.cacheOutputLocation
      : 'cpu';
    if (cacheLocation !== resolved.cacheOutputLocation) {
      warnings.push({
        code: 'qwen3-asr.gpu-kv-cache-unavailable',
        message: 'The selected decoder backend is not WebGPU; Qwen KV tensors will remain on the CPU.',
        recoverable: true,
      });
    }
    const sequentialSessions = resolved.ortBackend === 'wasm'
      && isStackedKv(this.config)
      && !this.dependencies.encoderSession
      && !this.dependencies.decoderSession;
    const sessionExtras = sequentialSessions ? { lowMemory: true } as const : {};
    const encoderSession = this.dependencies.encoderSession ?? await createQwenOrtSession(ort, artifacts.encoderUrl, {
      backendId: resolved.encoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.encoderDataUrl,
      externalDataPath: artifacts.encoderDataPath,
      preferredOutputLocation: resolved.decoderBackendForOrt === 'webgpu' ? 'cpu' : undefined,
      ...sessionExtras,
    });
    if (this.disposed) {
      releaseQwenOrtSession(encoderSession);
      throw new Error(`Qwen3-ASR executor was disposed during load for "${this.modelId}".`);
    }
    if (sequentialSessions) {
      return {
        ort,
        tokenizer,
        processor,
        encoderSession,
        decoderSession: undefined,
        decoderStepSession: undefined,
        resolved,
        warnings,
        sequentialSessions: true,
      };
    }
    const decoderSession = this.dependencies.decoderSession ?? await createQwenOrtSession(ort, artifacts.decoderUrl, {
      backendId: resolved.decoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      externalDataUrl: artifacts.decoderPrefillDataUrl ?? artifacts.decoderDataUrl,
      externalDataPath: artifacts.decoderPrefillDataPath ?? artifacts.decoderDataPath,
      preferredOutputLocation: this.config.graph.kvLayout === 'stacked'
        ? { logits: 'cpu', present_keys: cacheLocation, present_values: cacheLocation }
        : outputLocationMap(this.config, cacheLocation),
    });
    if (this.disposed) {
      releaseQwenOrtSession(encoderSession);
      releaseQwenOrtSession(decoderSession);
      throw new Error(`Qwen3-ASR executor was disposed during load for "${this.modelId}".`);
    }
    const decoderStepSession = this.dependencies.decoderStepSession
      ?? (artifacts.decoderStepUrl
        ? await createQwenOrtSession(ort, artifacts.decoderStepUrl, {
          backendId: resolved.decoderBackendForOrt,
          enableProfiling: resolved.enableProfiling,
          externalDataUrl: artifacts.decoderStepDataUrl,
          externalDataPath: artifacts.decoderStepDataPath,
          preferredOutputLocation: { logits: 'cpu', present_keys: cacheLocation, present_values: cacheLocation },
        })
        : undefined);
    if (this.disposed) {
      releaseQwenOrtSession(encoderSession);
      releaseQwenOrtSession(decoderSession);
      releaseQwenOrtSession(decoderStepSession);
      throw new Error(`Qwen3-ASR executor was disposed during load for "${this.modelId}".`);
    }
    return {
      ort,
      tokenizer,
      processor,
      encoderSession,
      decoderSession,
      decoderStepSession,
      resolved,
      warnings,
      sequentialSessions: false,
    };
  }

  private async loaded(): Promise<LoadedQwenState> {
    if (this.disposed) throw new Error(`Qwen3-ASR executor is disposed for "${this.modelId}".`);
    if (!this.loadStatePromise) throw createExperimentalArtifactMissingError('qwen-asr', this.modelId);
    return this.loadStatePromise;
  }

  async ready(): Promise<void> {
    await this.loaded();
  }

  private stackedCacheLocation(loaded: LoadedQwenState, options?: Qwen3AsrTranscriptionOptions): QwenCacheOutputLocation {
    return loaded.resolved.decoderBackendForOrt === 'webgpu'
      ? (options?.cacheOutputLocation ?? loaded.resolved.cacheOutputLocation)
      : 'cpu';
  }

  private async createArtifactSession(
    loaded: LoadedQwenState,
    url: string,
    extras: {
      readonly externalDataUrl?: string;
      readonly externalDataPath?: string;
      readonly preferredOutputLocation?: QwenCacheOutputLocation | Record<string, QwenCacheOutputLocation>;
      readonly encoder?: boolean;
    },
  ): Promise<QwenOrtSessionLike> {
    if (this.disposed) {
      throw new Error(`Qwen3-ASR executor is disposed for "${this.modelId}".`);
    }
    const session = await createQwenOrtSession(loaded.ort, url, {
      backendId: extras.encoder ? loaded.resolved.encoderBackendForOrt : loaded.resolved.decoderBackendForOrt,
      enableProfiling: loaded.resolved.enableProfiling,
      externalDataUrl: extras.externalDataUrl,
      externalDataPath: extras.externalDataPath,
      preferredOutputLocation: extras.preferredOutputLocation,
      lowMemory: loaded.sequentialSessions,
    });
    if (this.disposed) {
      releaseQwenOrtSession(session);
      throw new Error(`Qwen3-ASR executor was disposed during load for "${this.modelId}".`);
    }
    return session;
  }

  private async ensureEncoderSession(loaded: LoadedQwenState): Promise<QwenOrtSessionLike> {
    if (loaded.encoderSession) return loaded.encoderSession;
    if (loaded.sequentialSessions) {
      this.releaseLoadedSession(loaded, 'prefill');
      this.releaseLoadedSession(loaded, 'step');
    }
    const artifacts = loaded.resolved.artifacts;
    loaded.encoderSession = await this.createArtifactSession(loaded, artifacts.encoderUrl, {
      encoder: true,
      externalDataUrl: artifacts.encoderDataUrl,
      externalDataPath: artifacts.encoderDataPath,
      preferredOutputLocation: loaded.resolved.decoderBackendForOrt === 'webgpu' ? 'cpu' : undefined,
    });
    return loaded.encoderSession;
  }

  private async ensurePrefillSession(loaded: LoadedQwenState, cacheLocation: QwenCacheOutputLocation): Promise<QwenOrtSessionLike> {
    if (loaded.decoderSession) return loaded.decoderSession;
    const artifacts = loaded.resolved.artifacts;
    loaded.decoderSession = await this.createArtifactSession(loaded, artifacts.decoderUrl, {
      externalDataUrl: artifacts.decoderPrefillDataUrl ?? artifacts.decoderDataUrl,
      externalDataPath: artifacts.decoderPrefillDataPath ?? artifacts.decoderDataPath,
      preferredOutputLocation: this.config.graph.kvLayout === 'stacked'
        ? { logits: 'cpu', present_keys: cacheLocation, present_values: cacheLocation }
        : outputLocationMap(this.config, cacheLocation),
    });
    return loaded.decoderSession;
  }

  private async ensureStepSession(loaded: LoadedQwenState, cacheLocation: QwenCacheOutputLocation): Promise<QwenOrtSessionLike> {
    if (loaded.decoderStepSession) return loaded.decoderStepSession;
    const stepUrl = loaded.resolved.artifacts.decoderStepUrl;
    if (!stepUrl) throw new Error('Official Qwen decoder requires decoder-prefill and decoder-step sessions.');
    loaded.decoderStepSession = await this.createArtifactSession(loaded, stepUrl, {
      externalDataUrl: loaded.resolved.artifacts.decoderStepDataUrl,
      externalDataPath: loaded.resolved.artifacts.decoderStepDataPath,
      preferredOutputLocation: { logits: 'cpu', present_keys: cacheLocation, present_values: cacheLocation },
    });
    return loaded.decoderStepSession;
  }

  private releaseLoadedSession(loaded: LoadedQwenState, which: 'encoder' | 'prefill' | 'step'): void {
    if (which === 'encoder') {
      releaseQwenOrtSession(loaded.encoderSession);
      loaded.encoderSession = undefined;
      return;
    }
    if (which === 'prefill') {
      releaseQwenOrtSession(loaded.decoderSession);
      loaded.decoderSession = undefined;
      return;
    }
    releaseQwenOrtSession(loaded.decoderStepSession);
    loaded.decoderStepSession = undefined;
  }

  private async encodeAudio(
    loaded: LoadedQwenState,
    features: Qwen3AsrFeatureResult,
  ): Promise<{ readonly embeddings: Float32Array; readonly tokenCount: number; readonly encodeMs: number }> {
    const encodeStart = nowMs();
    const encoderSession = await this.ensureEncoderSession(loaded);
    if (isStackedKv(this.config)) {
      const featureTensor = new loaded.ort.Tensor(
        'float32',
        features.features,
        [features.nMels, features.frameCount],
      );
      let outputs: Record<string, QwenOrtTensorLike> | undefined;
      try {
        outputs = await encoderSession.run({ input_features: featureTensor });
        const embeddingsTensor = outputs.audio_embeddings;
        if (!embeddingsTensor) throw new Error('Qwen audio encoder did not return audio_embeddings.');
        const embeddingData = asFloat32(await readTensorData(embeddingsTensor), embeddingsTensor.type);
        const rawCount = embeddingsTensor.dims.length >= 2
          ? embeddingsTensor.dims[0] ?? 0
          : Math.floor(embeddingData.length / this.config.graph.hiddenSize);
        const expected = getQwenAudioTokenCount(features.validFrameCount);
        const tokenCount = expected > 0 && expected <= rawCount ? expected : rawCount;
        const embeddings = new Float32Array(embeddingData.subarray(0, tokenCount * this.config.graph.hiddenSize));
        return {
          embeddings,
          tokenCount,
          encodeMs: roundMetric(nowMs() - encodeStart, 3),
        };
      } finally {
        disposeTensor(featureTensor);
        for (const tensor of Object.values(outputs ?? {})) disposeTensor(tensor);
      }
    }
    const featureTensor = new loaded.ort.Tensor(
      'float16',
      float32ToFloat16Bits(features.features),
      [1, features.nMels, features.frameCount],
    );
    const maskTensor = new loaded.ort.Tensor('int32', features.inputFeaturesMask, [1, features.frameCount]);
    let outputs: Record<string, QwenOrtTensorLike> | undefined;
    try {
      outputs = await encoderSession.run({
        input_features: featureTensor,
        input_features_mask: maskTensor,
      });
      const embeddingsTensor = outputs.audio_embeddings;
      const tokenMaskTensor = outputs.audio_token_mask;
      if (!embeddingsTensor || !tokenMaskTensor) {
        throw new Error('Qwen audio encoder did not return audio_embeddings and audio_token_mask.');
      }
      const embeddingRaw = await readTensorData(embeddingsTensor);
      const embeddingData = asFloat32(embeddingRaw, embeddingsTensor.type);
      const maskRaw = await readTensorData(tokenMaskTensor);
      const maskData = maskRaw instanceof Uint8Array || maskRaw instanceof Int32Array || maskRaw instanceof Float32Array
        ? maskRaw
        : new Uint8Array(maskRaw.buffer, maskRaw.byteOffset, maskRaw.byteLength);
      const availableTokens = embeddingsTensor.dims.length >= 2
        ? embeddingsTensor.dims[embeddingsTensor.dims.length - 2] ?? 0
        : Math.floor(embeddingData.length / this.config.graph.hiddenSize);
      const maskLength = Math.min(availableTokens, maskData.length);
      const selected: number[] = [];
      for (let index = 0; index < maskLength; index += 1) {
        if (Number(maskData[index]) !== 0) selected.push(index);
      }
      const tokenIndices = selected.length > 0
        ? selected
        : Array.from({ length: Math.min(
          availableTokens,
          getQwenAudioTokenCount(features.validFrameCount),
        ) }, (_, index) => index);
      const embeddings = new Float32Array(tokenIndices.length * this.config.graph.hiddenSize);
      for (let outputIndex = 0; outputIndex < tokenIndices.length; outputIndex += 1) {
        const sourceIndex = tokenIndices[outputIndex] as number;
        const sourceOffset = sourceIndex * this.config.graph.hiddenSize;
        const targetOffset = outputIndex * this.config.graph.hiddenSize;
        embeddings.set(
          embeddingData.subarray(sourceOffset, sourceOffset + this.config.graph.hiddenSize),
          targetOffset,
        );
      }
      return {
        embeddings,
        tokenCount: tokenIndices.length,
        encodeMs: roundMetric(nowMs() - encodeStart, 3),
      };
    } finally {
      disposeTensor(featureTensor);
      disposeTensor(maskTensor);
      for (const tensor of Object.values(outputs ?? {})) disposeTensor(tensor);
    }
  }

  private createPastSeed(ort: QwenOrtModuleLike): Record<string, QwenOrtTensorLike> {
    const past: Record<string, QwenOrtTensorLike> = {};
    const size = this.config.graph.pastSeedLength * this.config.graph.numKvHeads * this.config.graph.headDim;
    const data = new Uint16Array(size);
    for (let layer = 0; layer < this.config.graph.numLayers; layer += 1) {
      past[`past.${layer}.key`] = new ort.Tensor(
        'float16',
        data.slice(),
        [1, this.config.graph.numKvHeads, this.config.graph.pastSeedLength, this.config.graph.headDim],
      );
      past[`past.${layer}.value`] = new ort.Tensor(
        'float16',
        data.slice(),
        [1, this.config.graph.numKvHeads, this.config.graph.pastSeedLength, this.config.graph.headDim],
      );
    }
    return past;
  }

  private createPromptEmbeddings(prompt: PromptTensors, audioEmbeddings: Float32Array): Uint16Array {
    const data = prompt.audioEmbeddings;
    for (let index = 0; index < prompt.audioTokenPositions.length; index += 1) {
      const sourceOffset = index * this.config.graph.hiddenSize;
      const targetOffset = (prompt.audioTokenPositions[index] as number) * this.config.graph.hiddenSize;
      data.set(
        float32ToFloat16Bits(audioEmbeddings.subarray(sourceOffset, sourceOffset + this.config.graph.hiddenSize)),
        targetOffset,
      );
    }
    return data;
  }

  private async runOfficialDecoder(
    loaded: LoadedQwenState,
    prompt: PromptTensors,
    audioEmbeddings: Float32Array,
    options: Qwen3AsrTranscriptionOptions,
  ): Promise<{
    readonly tokenIds: readonly number[];
    readonly decoderInitMs: number;
    readonly decoderInitInputMs: number;
    readonly decoderInitRunMs: number;
    readonly decoderInitOutputMs: number;
    readonly decoderStepMs: number;
    readonly decoderStepFeedBuildMs: number;
    readonly decoderStepRunMs: number;
    readonly decoderStepOutputMs: number;
    readonly decoderStepCount: number;
    readonly kvLocation: QwenCacheOutputLocation;
  }> {
    const graph = this.config.graph;
    const cacheLocation = this.stackedCacheLocation(loaded, options);
    throwIfDecodeAborted(options.signal);
    if (loaded.sequentialSessions) this.releaseLoadedSession(loaded, 'encoder');
    const prefillSession = await this.ensurePrefillSession(loaded, cacheLocation);
    const decoderInitStart = nowMs();
    const decoderInitInputStart = nowMs();
    const aligned = new Float32Array(prompt.inputIds.length * graph.hiddenSize);
    for (let index = 0; index < prompt.audioTokenPositions.length; index += 1) {
      const sourceOffset = index * graph.hiddenSize;
      const targetOffset = (prompt.audioTokenPositions[index] as number) * graph.hiddenSize;
      aligned.set(audioEmbeddings.subarray(sourceOffset, sourceOffset + graph.hiddenSize), targetOffset);
    }
    const positionIds = toInt64(Array.from({ length: prompt.inputIds.length }, (_, index) => index));
    const inputs: Record<string, unknown> = {
      input_ids: new loaded.ort.Tensor('int64', toInt64(prompt.inputIds), [1, prompt.inputIds.length]),
      audio_embeddings: floatFeed(
        loaded.ort,
        prefillSession,
        'audio_embeddings',
        aligned,
        [1, prompt.inputIds.length, graph.hiddenSize],
      ),
      position_ids: new loaded.ort.Tensor('int64', positionIds, [1, prompt.inputIds.length]),
    };
    const decoderInitInputMs = nowMs() - decoderInitInputStart;
    let initOutputs: Record<string, QwenOrtTensorLike> | undefined;
    const tokenIds: number[] = [];
    let decoderInitMs = 0;
    let decoderInitRunMs = 0;
    let decoderInitOutputMs = 0;
    let pastKeys: QwenOrtTensorLike | undefined;
    let pastValues: QwenOrtTensorLike | undefined;
    let keepPrefillPresent = false;
    try {
      const initRunStart = nowMs();
      initOutputs = await prefillSession.run(inputs);
      decoderInitRunMs = nowMs() - initRunStart;
      const initOutputStart = nowMs();
      const initLogits = initOutputs.logits;
      if (!initLogits) throw new Error('Qwen official prefill did not return logits.');
      const firstToken = argmaxLastRow(await copyQwenLogits(initLogits), graph.vocabularySize);
      pastKeys = initOutputs.present_keys;
      pastValues = initOutputs.present_values;
      if (!pastKeys || !pastValues) throw new Error('Qwen official prefill did not return present_keys/present_values.');
      decoderInitOutputMs = nowMs() - initOutputStart;
      decoderInitMs = nowMs() - decoderInitStart;
      if (graph.eosTokenIds.includes(firstToken)) {
        return {
          tokenIds,
          decoderInitMs: roundMetric(decoderInitMs, 3),
          decoderInitInputMs: roundMetric(decoderInitInputMs, 3),
          decoderInitRunMs: roundMetric(decoderInitRunMs, 3),
          decoderInitOutputMs: roundMetric(decoderInitOutputMs, 3),
          decoderStepMs: 0,
          decoderStepFeedBuildMs: 0,
          decoderStepRunMs: 0,
          decoderStepOutputMs: 0,
          decoderStepCount: 0,
          kvLocation: cacheLocation,
        };
      }
      tokenIds.push(firstToken);
      keepPrefillPresent = true;
    } finally {
      for (const tensor of Object.values(inputs)) {
        if (this.isTensor(tensor)) disposeTensor(tensor);
      }
      if (initOutputs) {
        for (const [name, tensor] of Object.entries(initOutputs)) {
          if (keepPrefillPresent && (name === 'present_keys' || name === 'present_values')) continue;
          disposeTensor(tensor);
        }
      }
    }

    let decoderStepMs = 0;
    let decoderStepFeedBuildMs = 0;
    let decoderStepRunMs = 0;
    let decoderStepOutputMs = 0;
    let decoderStepCount = 0;
    const maxNewTokens = Math.max(1, Math.floor(options.maxNewTokens ?? 512));
    let seq = prompt.inputIds.length;
    try {
      throwIfDecodeAborted(options.signal);
      if (loaded.sequentialSessions) this.releaseLoadedSession(loaded, 'prefill');
      const stepSession = await this.ensureStepSession(loaded, cacheLocation);
      while (tokenIds.length < maxNewTokens) {
        throwIfDecodeAborted(options.signal);
        const feedBuildStart = nowMs();
        const lastToken = tokenIds[tokenIds.length - 1] as number;
        const stepInputs: Record<string, unknown> = {
          input_ids: new loaded.ort.Tensor('int64', toInt64([lastToken]), [1, 1]),
          position_ids: new loaded.ort.Tensor('int64', toInt64([seq]), [1, 1]),
          past_keys: pastKeys,
          past_values: pastValues,
        };
        let stepOutputs: Record<string, QwenOrtTensorLike> | undefined;
        const stepStart = nowMs();
        decoderStepFeedBuildMs += stepStart - feedBuildStart;
        let transferred = false;
        try {
          const runStart = nowMs();
          stepOutputs = await stepSession.run(stepInputs);
          decoderStepRunMs += nowMs() - runStart;
          const outputStart = nowMs();
          const logits = stepOutputs.logits;
          if (!logits) throw new Error('Qwen official step did not return logits.');
          const nextToken = argmaxLastRow(await copyQwenLogits(logits), graph.vocabularySize);
          decoderStepCount += 1;
          const nextKeys = stepOutputs.present_keys;
          const nextValues = stepOutputs.present_values;
          if (!nextKeys || !nextValues) throw new Error('Qwen official step did not return present_keys/present_values.');
          disposeTensor(pastKeys);
          disposeTensor(pastValues);
          pastKeys = nextKeys;
          pastValues = nextValues;
          transferred = true;
          decoderStepOutputMs += nowMs() - outputStart;
          decoderStepMs += nowMs() - stepStart;
          if (graph.eosTokenIds.includes(nextToken)) break;
          tokenIds.push(nextToken);
          seq += 1;
        } finally {
          disposeTensor(stepInputs.input_ids as QwenOrtTensorLike);
          disposeTensor(stepInputs.position_ids as QwenOrtTensorLike);
          if (stepOutputs) {
            for (const [name, tensor] of Object.entries(stepOutputs)) {
              if (transferred && (name === 'present_keys' || name === 'present_values')) continue;
              disposeTensor(tensor);
            }
          }
        }
      }
    } finally {
      disposeTensor(pastKeys);
      disposeTensor(pastValues);
    }
    return {
      tokenIds,
      decoderInitMs: roundMetric(decoderInitMs, 3),
      decoderInitInputMs: roundMetric(decoderInitInputMs, 3),
      decoderInitRunMs: roundMetric(decoderInitRunMs, 3),
      decoderInitOutputMs: roundMetric(decoderInitOutputMs, 3),
      decoderStepMs: roundMetric(decoderStepMs, 3),
      decoderStepFeedBuildMs: roundMetric(decoderStepFeedBuildMs, 3),
      decoderStepRunMs: roundMetric(decoderStepRunMs, 3),
      decoderStepOutputMs: roundMetric(decoderStepOutputMs, 3),
      decoderStepCount,
      kvLocation: cacheLocation,
    };
  }

  private async runDecoder(
    loaded: LoadedQwenState,
    prompt: PromptTensors,
    audioEmbeddings: Float32Array,
    options: Qwen3AsrTranscriptionOptions,
  ): Promise<{
    readonly tokenIds: readonly number[];
    readonly decoderInitMs: number;
    readonly decoderInitInputMs: number;
    readonly decoderInitRunMs: number;
    readonly decoderInitOutputMs: number;
    readonly decoderStepMs: number;
    readonly decoderStepFeedBuildMs: number;
    readonly decoderStepRunMs: number;
    readonly decoderStepOutputMs: number;
    readonly decoderStepCount: number;
    readonly kvLocation: QwenCacheOutputLocation;
  }> {
    const graph = this.config.graph;
    const requestedCacheLocation = options.cacheOutputLocation ?? loaded.resolved.cacheOutputLocation;
    throwIfDecodeAborted(options.signal);
    const cacheLocation: QwenCacheOutputLocation = loaded.resolved.decoderBackendForOrt === 'webgpu'
      ? requestedCacheLocation
      : 'cpu';
    const decoderInitStart = nowMs();
    const decoderInitInputStart = nowMs();
    const inputs: Record<string, unknown> = {
      input_ids: new loaded.ort.Tensor('int32', prompt.inputIds, [1, prompt.inputIds.length]),
      audio_embeddings: new loaded.ort.Tensor(
        'float16',
        this.createPromptEmbeddings(prompt, audioEmbeddings),
        [1, prompt.inputIds.length, graph.hiddenSize],
      ),
      audio_mask: new loaded.ort.Tensor('float16', prompt.audioMask, [1, prompt.inputIds.length, 1]),
      attention_mask: new loaded.ort.Tensor(
        'float16',
        prompt.attentionMask,
        [1, 1, prompt.inputIds.length, graph.pastSeedLength + prompt.inputIds.length],
      ),
      position_ids: new loaded.ort.Tensor('int32', prompt.positionIds, [1, prompt.inputIds.length]),
      ...this.createPastSeed(loaded.ort),
    };
    const decoderInitInputMs = nowMs() - decoderInitInputStart;
    let initOutputs: Record<string, QwenOrtTensorLike> | undefined;
    let past: Record<string, QwenOrtTensorLike> = {};
    const tokenIds: number[] = [];
    let decoderInitMs = 0;
    let decoderInitRunMs = 0;
    let decoderInitOutputMs = 0;
    let keepPrefillPresent = false;
    try {
      const initRunStart = nowMs();
      initOutputs = await loaded.decoderSession!.run(inputs);
      decoderInitRunMs = nowMs() - initRunStart;
      const initOutputStart = nowMs();
      const initLogits = initOutputs.logits;
      if (!initLogits) throw new Error('Qwen decoder did not return logits during prefill.');
      const firstToken = argmaxLastRow(await copyQwenLogits(initLogits), graph.vocabularySize);
      decoderInitOutputMs = nowMs() - initOutputStart;
      if (graph.eosTokenIds.includes(firstToken)) {
        decoderInitMs = nowMs() - decoderInitStart;
        return {
          tokenIds,
          decoderInitMs: roundMetric(decoderInitMs, 3),
          decoderInitInputMs: roundMetric(decoderInitInputMs, 3),
          decoderInitRunMs: roundMetric(decoderInitRunMs, 3),
          decoderInitOutputMs: roundMetric(decoderInitOutputMs, 3),
          decoderStepMs: 0,
          decoderStepFeedBuildMs: 0,
          decoderStepRunMs: 0,
          decoderStepOutputMs: 0,
          decoderStepCount: 0,
          kvLocation: cacheLocation,
        };
      }
      tokenIds.push(firstToken);
      for (let layer = 0; layer < graph.numLayers; layer += 1) {
        const key = initOutputs[`present.${layer}.key`];
        const value = initOutputs[`present.${layer}.value`];
        if (!key || !value) throw new Error(`Qwen decoder is missing present.${layer}.key/value outputs.`);
        past[`past.${layer}.key`] = key;
        past[`past.${layer}.value`] = value;
      }
      keepPrefillPresent = true;
      decoderInitMs = nowMs() - decoderInitStart;
    } finally {
      for (const tensor of Object.values(inputs)) {
        if (this.isTensor(tensor)) disposeTensor(tensor);
      }
      if (initOutputs) {
        for (const [name, tensor] of Object.entries(initOutputs)) {
          if (keepPrefillPresent && name.startsWith('present.')) continue;
          disposeTensor(tensor);
        }
      }
    }

    let decoderStepMs = 0;
    let decoderStepFeedBuildMs = 0;
    let decoderStepRunMs = 0;
    let decoderStepOutputMs = 0;
    let decoderStepCount = 0;
    const maxNewTokens = Math.max(1, Math.floor(options.maxNewTokens ?? 512));
    try {
      throwIfDecodeAborted(options.signal);
      while (tokenIds.length < maxNewTokens) {
        throwIfDecodeAborted(options.signal);
        const feedBuildStart = nowMs();
        const representativePast = past['past.0.key'];
        const pastSequenceLength = representativePast?.dims[2] ?? graph.pastSeedLength + prompt.inputIds.length;
        const attention = new Float32Array(pastSequenceLength + 1);
        attention.fill(graph.pastSeedAttentionMask);
        attention.fill(0, graph.pastSeedLength);
        const stepInputs: Record<string, unknown> = {
          input_ids: new loaded.ort.Tensor('int32', Int32Array.of(tokenIds[tokenIds.length - 1] as number), [1, 1]),
          audio_embeddings: new loaded.ort.Tensor('float16', new Uint16Array(graph.hiddenSize), [1, 1, graph.hiddenSize]),
          audio_mask: new loaded.ort.Tensor('float16', new Uint16Array(1), [1, 1, 1]),
          attention_mask: new loaded.ort.Tensor('float16', float32ToFloat16Bits(attention), [1, 1, 1, pastSequenceLength + 1]),
          position_ids: new loaded.ort.Tensor('int32', Int32Array.of(pastSequenceLength), [1, 1]),
          ...past,
        };
        const stepStart = nowMs();
        decoderStepFeedBuildMs += stepStart - feedBuildStart;
        let stepOutputs: Record<string, QwenOrtTensorLike> | undefined;
        let transferred = false;
        try {
          const runStart = nowMs();
          stepOutputs = await loaded.decoderSession!.run(stepInputs);
          decoderStepRunMs += nowMs() - runStart;
          const outputStart = nowMs();
          const logits = stepOutputs.logits;
          if (!logits) throw new Error('Qwen decoder did not return logits during autoregressive decoding.');
          const nextToken = argmaxLastRow(await copyQwenLogits(logits), graph.vocabularySize);
          decoderStepCount += 1;
          const nextPast: Record<string, QwenOrtTensorLike> = {};
          for (let layer = 0; layer < graph.numLayers; layer += 1) {
            const key = stepOutputs[`present.${layer}.key`];
            const value = stepOutputs[`present.${layer}.value`];
            if (!key || !value) throw new Error(`Qwen decoder step is missing present.${layer}.key/value outputs.`);
            nextPast[`past.${layer}.key`] = key;
            nextPast[`past.${layer}.value`] = value;
          }
          disposeTensorRecord(past);
          past = nextPast;
          transferred = true;
          decoderStepOutputMs += nowMs() - outputStart;
          decoderStepMs += nowMs() - stepStart;
          if (graph.eosTokenIds.includes(nextToken)) break;
          tokenIds.push(nextToken);
        } finally {
          for (const [name, tensor] of Object.entries(stepInputs)) {
            if (!name.startsWith('past.')) {
              if (this.isTensor(tensor)) disposeTensor(tensor);
            }
          }
          if (stepOutputs) {
            for (const [name, tensor] of Object.entries(stepOutputs)) {
              if (transferred && name.startsWith('present.')) continue;
              disposeTensor(tensor);
            }
          }
        }
      }
    } finally {
      disposeTensorRecord(past);
    }
    return {
      tokenIds,
      decoderInitMs: roundMetric(decoderInitMs, 3),
      decoderInitInputMs: roundMetric(decoderInitInputMs, 3),
      decoderInitRunMs: roundMetric(decoderInitRunMs, 3),
      decoderInitOutputMs: roundMetric(decoderInitOutputMs, 3),
      decoderStepMs: roundMetric(decoderStepMs, 3),
      decoderStepFeedBuildMs: roundMetric(decoderStepFeedBuildMs, 3),
      decoderStepRunMs: roundMetric(decoderStepRunMs, 3),
      decoderStepOutputMs: roundMetric(decoderStepOutputMs, 3),
      decoderStepCount,
      kvLocation: cacheLocation,
    };
  }

  private isTensor(value: unknown): value is QwenOrtTensorLike {
    return typeof value === 'object' && value !== null && 'dims' in value && 'dispose' in value;
  }

  async transcribe(
    audio: AudioBufferLike,
    options: Qwen3AsrTranscriptionOptions,
    context: { readonly modelId: string; readonly classification: ModelClassification; readonly config: Qwen3AsrModelConfig },
  ): Promise<Qwen3AsrNativeTranscript> {
    const loaded = await this.loaded();
    const start = nowMs();
    emitProgress(options, { stage: 'start', progress: 0, modelId: context.modelId, backendId: this.backendId });
    const forcedLanguage = normalizeQwenLanguage(options.language);
    if (forcedLanguage && !context.config.languages.some((language) => language.toLowerCase() === forcedLanguage.toLowerCase())) {
      throw new RangeError(`Unsupported Qwen3-ASR language "${options.language}".`);
    }
    if (audio.durationSeconds > context.config.maxInputDurationSec && !options.unsafeAllowOverMaxWindow) {
      throw new RangeError(`Qwen3-ASR browser graph supports at most ${context.config.maxInputDurationSec} seconds per request.`);
    }
    const preprocessStart = nowMs();
    const features = loaded.processor.process(audio);
    const preprocessMs = roundMetric(nowMs() - preprocessStart, 3);
    emitProgress(options, {
      stage: 'preprocess',
      progress: 0.25,
      elapsedMs: roundMetric(nowMs() - start, 3),
      modelId: context.modelId,
      backendId: this.backendId,
      metrics: { preprocessMs, encoderFrameCount: features.frameCount },
    });
    const encoded = await this.encodeAudio(loaded, features);
    emitProgress(options, {
      stage: 'encode',
      progress: 0.55,
      elapsedMs: roundMetric(nowMs() - start, 3),
      modelId: context.modelId,
      backendId: this.backendId,
      metrics: { preprocessMs, encodeMs: encoded.encodeMs, encoderFrameCount: features.frameCount },
    });
    const prompt = buildPrompt(
      loaded.tokenizer,
      context.config.graph,
      encoded.tokenCount,
      forcedLanguage,
      options.context ?? '',
    );
    const decoded = isStackedKv(context.config)
      ? await this.runOfficialDecoder(loaded, prompt, encoded.embeddings, options)
      : await this.runDecoder(loaded, prompt, encoded.embeddings, options);
    const rawText = loaded.tokenizer.decode(decoded.tokenIds, { skipSpecialTokens: true });
    const parsed = parseQwenOutput(rawText, forcedLanguage);
    const tokens: Qwen3AsrNativeToken[] = decoded.tokenIds.map((id, index) => ({
      index,
      id,
      text: loaded.tokenizer.decode([id]),
      special: loaded.tokenizer.isSpecialTokenId(id),
    }));
    const warnings: TranscriptWarning[] = [...loaded.warnings];
    if (options.returnTimestamps) {
      warnings.push({
        code: 'qwen3-asr.timestamps-unavailable',
        message: 'Qwen3-ASR timestamps require the separate Qwen3-ForcedAligner artifact; this graph returns an utterance segment only.',
        recoverable: true,
      });
    }
    const totalMs = roundMetric(nowMs() - start, 3);
    const metrics: TranscriptMetrics = {
      preprocessMs,
      encodeMs: encoded.encodeMs,
      decodeMs: roundMetric(decoded.decoderInitMs + decoded.decoderStepMs, 3),
      decoderInitMs: decoded.decoderInitMs,
      decoderInitInputMs: decoded.decoderInitInputMs,
      decoderInitRunMs: decoded.decoderInitRunMs,
      decoderInitOutputMs: decoded.decoderInitOutputMs,
      decoderStepMs: decoded.decoderStepMs,
      decoderStepFeedBuildMs: decoded.decoderStepFeedBuildMs,
      decoderStepRunMs: decoded.decoderStepRunMs,
      decoderStepOutputMs: decoded.decoderStepOutputMs,
      decoderStepAvgMs: decoded.decoderStepCount > 0
        ? roundMetric(decoded.decoderStepMs / decoded.decoderStepCount, 3)
        : undefined,
      decoderStepCount: decoded.decoderStepCount,
      decoderKvCacheLocation: decoded.kvLocation,
      totalMs,
      wallMs: totalMs,
      audioDurationSec: features.durationSeconds,
      rtf: features.durationSeconds > 0 ? totalMs / 1000 / features.durationSeconds : undefined,
      rtfx: features.durationSeconds > 0 ? features.durationSeconds / (totalMs / 1000) : undefined,
      encoderFrameCount: features.frameCount,
      emittedTokenCount: decoded.tokenIds.length,
      decodeIterations: decoded.decoderStepCount,
    };
    emitProgress(options, {
      stage: 'complete',
      progress: 1,
      elapsedMs: totalMs,
      modelId: context.modelId,
      backendId: this.backendId,
      metrics,
    });
    return {
      utteranceText: parsed.text,
      rawText,
      language: parsed.language,
      isFinal: true,
      tokens: options.returnTokens || options.detail === 'detailed' || options.returnSpecialTokens
        ? (options.returnSpecialTokens ? tokens : tokens.filter((token) => !token.special))
        : undefined,
      segments: [{ index: 0, text: parsed.text, startTime: 0, endTime: features.durationSeconds }],
      metrics,
      warnings,
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
        releaseQwenOrtSession(loaded.encoderSession);
        releaseQwenOrtSession(loaded.decoderSession);
        releaseQwenOrtSession(loaded.decoderStepSession);
        loaded.encoderSession = undefined;
        loaded.decoderSession = undefined;
        loaded.decoderStepSession = undefined;
      } catch {
        // Keep the original load error; still drop asset handles.
      }
    }
    const handles = this.assetHandles.splice(0);
    await Promise.all(handles.map((handle) => Promise.resolve(handle.dispose())));
  }
}
