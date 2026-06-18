import type {
  AudioBufferLike,
  AssetProvider,
  ModelClassification,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
  SpeechRuntimeHooks,
  TranscriptMetrics,
  TranscriptWarning,
  TranscriptionProgressEvent,
} from '../../types/index.js';
import { argmax, confidenceFromLogits } from '../../inference/index.js';
import { fetchModelFiles } from '../../runtime/huggingface.js';
import { nowMs, roundMetric } from '../../runtime/timing.js';
import { WhisperMelProcessor } from '../../audio/whisper-mel.js';
import { planWhisperChunks } from '../../pipeline/whisper-chunking.js';
import {
  createInitialWhisperBeam,
  rankWhisperBeamCandidates,
  selectBestWhisperBeam,
  type WhisperBeamState,
} from './beam-search.js';
import { mergeWhisperChunkTranscripts } from './chunking.js';
import {
  createWhisperOrtSession,
  initWhisperOrt,
  resolveWhisperArtifacts,
  type OrtModuleLike,
  type OrtPreferredOutputLocation,
  type OrtSessionLike,
  type OrtTensorLike,
  type ResolvedWhisperArtifacts,
} from './ort.js';
import { WhisperTokenizer, fetchText } from './tokenizer.js';
import { WhisperTimestampLogitProcessor } from './processors.js';
import { buildWhisperWordTimestampsFromTokenDetails } from './word-timestamps.js';
import { computeWhisperDtwTokenTimestamps } from './attention-alignment.js';
import { whisperDecode, type WhisperCoreSession } from './core.js';
import {
  parseWhisperGenerationConfig,
  parseWhisperModelConfig,
  type WhisperGenerationConfig,
  type WhisperModelConfig,
} from './generation-config.js';
import type {
  WhisperArtifactSource,
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperNativeWord,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: WhisperTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession?: OrtSessionLike;
  readonly generationConfig: WhisperGenerationConfig;
  readonly modelConfig: WhisperModelConfig;
  readonly warnings: readonly TranscriptWarning[];
  readonly isSplitGraph: boolean;
  readonly decoderInitSession?: OrtSessionLike;
  readonly decoderStepSession?: OrtSessionLike;
  readonly decoderAlignSession?: OrtSessionLike;
  readonly decoderBackendForOrt?: string;
  readonly experimentalGpuKvCache?: boolean;
  readonly sessionCreateMs?: number;
}

interface DecoderStepResult {
  readonly lastLogits: Float32Array;
  readonly vocabSize: number;
  readonly pastKeyValues: Record<string, OrtTensorLike<Float32Array>>;
  readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
}

function extractCrossAttentions(
  outputs: Record<string, unknown>,
): OrtTensorLike<Float32Array>[] {
  const entries: { layer: number; tensor: OrtTensorLike<Float32Array> }[] = [];
  for (const [key, value] of Object.entries(outputs)) {
    const match = key.match(/^cross_attentions\.(\d+)$/);
    if (match) {
      entries.push({
        layer: parseInt(match[1]!, 10),
        tensor: value as OrtTensorLike<Float32Array>,
      });
    }
  }
  entries.sort((a, b) => a.layer - b.layer);
  return entries.map((e) => e.tensor);
}

interface BeamPayload {
  readonly tokenDetails: readonly WhisperNativeToken[];
  readonly pastKeyValues: Record<string, OrtTensorLike<Float32Array>>;
}

function roundMiB(bytes: number | undefined): number | undefined {
  if (!Number.isFinite(bytes)) return undefined;
  return roundMetric((bytes as number) / (1024 * 1024), 2);
}

function clampProgress(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function estimateRemainingMs(elapsedMs: number, progress: number): number | undefined {
  if (progress <= 0 || progress >= 1) {
    return undefined;
  }

  return roundMetric((elapsedMs / progress) * (1 - progress), 2);
}

function emitTranscriptionProgress(
  options: WhisperSeq2SeqTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

/**
 * Convert a Float32Array to a Uint16Array of fp16 bits.
 * Uses round-to-nearest-even.
 */
function float32ToFloat16Bits(src: Float32Array): Uint16Array {
  const dst = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const x = src[i]!;
    const b = new DataView(new ArrayBuffer(4));
    b.setFloat32(0, x, true);
    const uint32 = b.getUint32(0, true);
    const sign = (uint32 >>> 31) & 0x1;
    let exp = (uint32 >>> 23) & 0xff;
    let mant = uint32 & 0x7fffff;
    let fp16: number;
    if (exp === 0xff) {
      // Inf/NaN
      fp16 = (sign << 15) | 0x7c00 | (mant ? 0x200 : 0);
    } else if (exp > 142) {
      // Overflow to infinity
      fp16 = (sign << 15) | 0x7c00;
    } else if (exp < 113) {
      // Subnormal or zero
      if (exp < 103) {
        fp16 = sign << 15;
      } else {
        mant |= 0x800000;
        const shift = 113 - exp - 1;
        fp16 = (sign << 15) | (mant >>> shift);
      }
    } else {
      exp -= 112;
      mant >>= 13;
      fp16 = (sign << 15) | (exp << 10) | mant;
    }
    dst[i] = fp16;
  }
  return dst;
}

/**
 * If the decoder expects float16 encoder hidden states, cast the fp32 encoder output.
 * When the Cast-injected decoder_init model is used (accepts fp32 natively), this
 * is a no-op and the GPU tensor flows through with zero CPU touch.
 */
async function maybeCastEncoderHiddenStates(
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  decoderInitSession: OrtSessionLike,
  ort: OrtModuleLike,
): Promise<OrtTensorLike<Float32Array>> {
  const metadata = (decoderInitSession as unknown as { inputMetadata?: Array<{ name?: string; type?: string; shape?: number[] }> }).inputMetadata;
  const encMeta = metadata?.find((m) => m.name === 'encoder_hidden_states');
  // If decoder_init already accepts fp32 (Cast-injected model), skip the CPU cast.
  // The encoder output flows directly from GPU to decoder_init with zero CPU touch.
  if (!encMeta || encMeta.type !== 'float16') {
    return encoderHiddenStates;
  }
  // If encoder already outputs fp16 (stripped encoder) and decoder accepts fp16,
  // no cast needed — GPU tensor passes through directly.
  if (encoderHiddenStates.type === 'float16') {
    return encoderHiddenStates;
  }
  // Decoder expects float16 but encoder outputs fp32 — CPU cast needed.
  // If tensor is on GPU, download first (this path is only hit with the original
  // fp32-output encoder + original fp16-input decoder_init combination).
  const f32Data = isGpuBufferTensor(encoderHiddenStates) && encoderHiddenStates.getData
    ? (await encoderHiddenStates.getData(true)) as Float32Array
    : encoderHiddenStates.data as Float32Array;
  const dims = encoderHiddenStates.dims as number[];
  const size = dims.reduce((a, b) => a * b, 1);
  const f16Bits = float32ToFloat16Bits(
    (f32Data as Float32Array).length === size
      ? (f32Data as Float32Array)
      : (f32Data as Float32Array).subarray(0, size),
  );
  return new ort.Tensor('float16', f16Bits, dims) as unknown as OrtTensorLike<Float32Array>;
}

function createWhisperGpuKvOutputLocation(
  config: WhisperModelConfig,
  role: 'init' | 'step',
): OrtPreferredOutputLocation {
  const outputLocation: Record<string, 'cpu' | 'gpu-buffer'> = {
    logits: 'cpu',
    // GPU ArgMax output: always keep on CPU (INT32 scalar, 4 bytes)
    next_token_id: 'cpu',
  };
  for (let layer = 0; layer < config.decoderLayers; layer++) {
    outputLocation[`present.${layer}.decoder.key`] = 'gpu-buffer';
    outputLocation[`present.${layer}.decoder.value`] = 'gpu-buffer';
    if (role === 'init') {
      outputLocation[`present.${layer}.encoder.key`] = 'gpu-buffer';
      outputLocation[`present.${layer}.encoder.value`] = 'gpu-buffer';
    }
  }
  return outputLocation;
}

function createAssetProgressEvent(
  modelId: string,
  file: string,
  event: { readonly loaded: number; readonly total?: number; readonly done?: boolean },
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

function isAssetMissingError(error: unknown): boolean {
  if (error instanceof Error) {
    return /\b404\b/.test(error.message) || /\bnot found\b/i.test(error.message);
  }
  return false;
}

function normalizeRepoPath(path: string): string {
  return String(path || '').replace(/^\.\/+/, '').replace(/\\/g, '/');
}

function hasListedRepoFile(files: readonly string[], filename: string): boolean {
  const target = normalizeRepoPath(filename);
  return files.some(
    (p) => normalizeRepoPath(p) === target || normalizeRepoPath(p).endsWith(`/${target}`),
  );
}

export function computeEmptyPastKeyValueShapes(
  config: WhisperModelConfig,
  encoderSeqLen: number,
): Record<string, readonly number[]> {
  const shapes: Record<string, readonly number[]> = {};
  const { decoderLayers, decoderAttentionHeads, headDim } = config;
  for (let i = 0; i < decoderLayers; i++) {
    shapes[`past_key_values.${i}.decoder.key`] = [1, decoderAttentionHeads, 0, headDim];
    shapes[`past_key_values.${i}.decoder.value`] = [1, decoderAttentionHeads, 0, headDim];
    shapes[`past_key_values.${i}.encoder.key`] = [1, decoderAttentionHeads, encoderSeqLen, headDim];
    shapes[`past_key_values.${i}.encoder.value`] = [1, decoderAttentionHeads, encoderSeqLen, headDim];
  }
  return shapes;
}

export interface SplitGraphDecodeCallbacks {
  runInit(
    promptTokens: readonly number[],
    encoderHiddenStates: Float32Array,
    encoderDims: readonly number[],
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: Record<string, Float32Array> }>;
  runStep(
    tokenId: number,
    pastKv: Record<string, Float32Array>,
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: Record<string, Float32Array> }>;
}

export interface SplitGraphDecodeResult {
  readonly tokens: readonly number[];
}

interface DecoderSessionTiming {
  readonly inputMs: number;
  readonly runMs: number;
  readonly outputMs: number;
  readonly gpuInputCount: number;
  readonly cpuInputCount: number;
  readonly gpuOutputCount: number;
  readonly cpuOutputCount: number;
  readonly gpuDownloadCount: number;
  // Profiling sub-buckets (added by runDecoderInit / runDecoderStepSplit)
  readonly tensorCreateMs?: number;
  readonly logitReadMs?: number;
  readonly kvExtractMs?: number;
}

interface TensorLocationCounts {
  readonly gpu: number;
  readonly cpu: number;
}

function isOrtTensorLike(value: unknown): value is OrtTensorLike {
  return Boolean(value) && typeof value === 'object' && Array.isArray((value as { dims?: unknown }).dims);
}

function isGpuBufferTensor(tensor: OrtTensorLike): boolean {
  return tensor.location === 'gpu-buffer';
}

function countTensorLocations(values: Iterable<unknown>): TensorLocationCounts {
  let gpu = 0;
  let cpu = 0;
  for (const value of values) {
    if (!isOrtTensorLike(value)) continue;
    if (isGpuBufferTensor(value)) {
      gpu++;
    } else {
      cpu++;
    }
  }
  return { gpu, cpu };
}

async function readOrtTensorData<TData extends ArrayBufferView>(
  tensor: OrtTensorLike<TData>,
  options: { readonly releaseGpu?: boolean } = {},
): Promise<{ readonly data: TData; readonly downloaded: boolean }> {
  if (isGpuBufferTensor(tensor)) {
    if (!tensor.getData) {
      throw new Error('ORT GPU tensor does not expose getData().');
    }
    return {
      data: await tensor.getData(options.releaseGpu ?? false),
      downloaded: true,
    };
  }
  return { data: tensor.data, downloaded: false };
}

function disposeGpuTensor(tensor: OrtTensorLike | undefined): void {
  if (tensor && isGpuBufferTensor(tensor)) {
    tensor.dispose?.();
  }
}

function disposeReplacedGpuKv(
  previous: Record<string, OrtTensorLike>,
  next: Record<string, OrtTensorLike>,
): void {
  for (const [key, tensor] of Object.entries(previous)) {
    if (next[key] !== tensor) {
      disposeGpuTensor(tensor);
    }
  }
}

function disposeGpuKv(kv: Record<string, OrtTensorLike>): void {
  for (const tensor of Object.values(kv)) {
    disposeGpuTensor(tensor);
  }
}

function mapPresentKvToPastKv(
  presentKv: Record<string, OrtTensorLike<Float32Array>>,
): Record<string, OrtTensorLike<Float32Array>> {
  const mapped: Record<string, OrtTensorLike<Float32Array>> = {};
  for (const [name, tensor] of Object.entries(presentKv)) {
    mapped[name.replace(/^present\./, 'past_key_values.')] = tensor;
  }
  return mapped;
}

export async function splitGraphDecodeLoop(params: {
  promptTokens: readonly number[];
  encoderHiddenStates: Float32Array;
  eosTokenId: number;
  maxNewTokens: number;
  modelConfig: WhisperModelConfig;
  runInit: SplitGraphDecodeCallbacks['runInit'];
  runStep: SplitGraphDecodeCallbacks['runStep'];
  processLogits?: (logits: Float32Array, generatedTokens: readonly number[], beginIndex: number) => void;
  onTokenLogits?: (chosenTokenId: number, processedLogits: Float32Array, ctx: { readonly tokens: readonly number[]; readonly beginIndex: number }) => void;
  /** Beam search: number of beams (default: 1 = greedy) */
  numBeams?: number;
  /** Length penalty for beam search (default: 0.0) */
  lengthPenalty?: number;
  /** Beam search patience for early stopping. */
  patience?: number;
  /** Greedy decoding temperature. 0 = argmax. */
  temperature?: number;
  /** Number of independent decodings, pick best by score (WhisperX: best_of) */
  bestOf?: number;
}): Promise<SplitGraphDecodeResult> {
  const {
    promptTokens,
    encoderHiddenStates,
    eosTokenId,
    maxNewTokens,
    modelConfig,
    runInit,
    runStep,
    processLogits,
    onTokenLogits,
    numBeams,
    lengthPenalty,
    patience,
    temperature,
    bestOf,
  } = params;

  const encoderDims: readonly number[] = [1, encoderHiddenStates.length / modelConfig.dModel, modelConfig.dModel];
  const session: WhisperCoreSession = {
    runInit: async (pt, enc, dims) => runInit(pt, enc, dims),
    runStep: async (tid, kv) => runStep(tid, kv),
  };
  const result = await whisperDecode(session, {
    promptTokens, encoderOutput: encoderHiddenStates, encoderDims, eosTokenId, maxNewTokens, processLogits, onTokenLogits,
    strategy: (numBeams ?? 1) > 1 ? 'beam' : 'greedy',
    beamSize: numBeams ?? 1,
    lengthPenalty: lengthPenalty ?? 0,
    patience: patience ?? 1,
    temperature: temperature ?? 0,
    bestOf: bestOf ?? 1,
  });
  return { tokens: result.tokens };
}

function percentile(values: readonly number[], p: number): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
}

export interface SplitGraphAlignmentOptions {
  readonly alignmentData: Float32Array;
  readonly totalTokens: number;
  readonly promptLen: number;
  readonly textTokenCount: number;
  readonly frameCount: number;
  readonly medianFilterWidth?: number;
  readonly timePrecisionSeconds?: number;
}

export function processSplitGraphAlignment(
  options: SplitGraphAlignmentOptions,
): readonly number[] {
  const {
    alignmentData, totalTokens: _totalTokens, promptLen, textTokenCount, frameCount,
    medianFilterWidth, timePrecisionSeconds,
  } = options;

  if (textTokenCount === 0) return [0];
  if (frameCount === 0) return Array.from({ length: textTokenCount + 1 }, () => 0);

  // Slice off prompt rows: alignment[T_all, S] → extract text-only rows [promptLen:totalTokens, :]
  const textValues = new Float32Array(textTokenCount * frameCount);
  const srcOffset = promptLen * frameCount;
  textValues.set(alignmentData.subarray(srcOffset, srcOffset + textTokenCount * frameCount));

  const headMatrix = {
    values: textValues,
    tokenCount: textTokenCount,
    frameCount,
  };

  return computeWhisperDtwTokenTimestamps({
    attentionHeads: [headMatrix],
    tokenCount: textTokenCount,
    frameCount,
    medianFilterWidth,
    timePrecisionSeconds,
  });
}

export class WhisperOnnxExecutor {
  private readonly sourceOptions: WhisperArtifactSource | undefined;
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];

  constructor(
    private readonly modelId: string,
    _classification: ModelClassification,
    private readonly config: WhisperSeq2SeqModelConfig,
    private readonly backendId: string,
    loadOptions: { readonly source?: WhisperArtifactSource } | undefined,
    dependencies: {
      readonly assetProvider?: AssetProvider;
      readonly runtimeHooks?: SpeechRuntimeHooks;
    } = {},
  ) {
    this.sourceOptions = loadOptions?.source;
    this.assetProvider = dependencies.assetProvider;
    this.runtimeHooks = dependencies.runtimeHooks;
    if (this.sourceOptions) {
      this.loadStatePromise = this.initialize();
    }
  }

  private async materializeResolvedArtifacts(
    resolved: ResolvedWhisperArtifacts,
  ): Promise<ResolvedWhisperArtifacts> {
    const source = this.sourceOptions;
    if (!this.assetProvider || !source) {
      return resolved;
    }

    const resolveRemoteUrl = async (
      url: string | undefined,
      fileLabel: string,
    ): Promise<string | undefined> => {
      if (!url || !/^https?:\/\//i.test(url)) {
        return url;
      }

      // Local browser demos serve very large ONNX external-data files from the
      // same origin. Hand those URLs directly to ORT instead of routing them
      // through IndexedDB/blob URLs, which is useful for remote assets but a
      // fragile extra hop for multi-GB local files.
      if (
        typeof window !== 'undefined' &&
        url.startsWith(`${window.location.origin}/models/`)
      ) {
        return url;
      }

      const handle = await this.assetProvider!.resolve({
        id: `url:${url}`,
        provider: 'url',
        url,
        preferBlobUrl: true,
        cacheKey: `url:${url}`,
        onProgress: (event) => {
          this.runtimeHooks?.onProgress?.(createAssetProgressEvent(this.modelId, fileLabel, event));
        },
      });
      this.assetHandles.push(handle);
      const locator = await handle.getLocator('url');
      if (!locator) {
        throw new Error(`Could not create a URL locator for "${fileLabel}".`);
      }
      return locator;
    };

    if (source.kind === 'splitgraph') {
      const materializedExternalData: Record<string, { dataUrl: string; path: string }[]> = {};
      for (const [graphName, entries] of Object.entries(resolved.externalData ?? {})) {
        const nextEntries: { dataUrl: string; path: string }[] = [];
        for (const entry of entries ?? []) {
          nextEntries.push({
            path: entry.path,
            dataUrl:
              (await resolveRemoteUrl(
                entry.dataUrl,
                `${graphName}/${entry.path.replace(/^\.\//, '')}`,
              )) ?? entry.dataUrl,
          });
        }
        if (nextEntries.length > 0) {
          materializedExternalData[graphName] = nextEntries;
        }
      }

      const encoderUrl =
        (await resolveRemoteUrl(resolved.artifacts.encoderUrl, 'encoder_model.onnx')) ??
        resolved.artifacts.encoderUrl;
      const decoderInitUrl =
        (await resolveRemoteUrl(resolved.decoderInitUrl, 'decoder_init.onnx')) ??
        resolved.decoderInitUrl;
      const decoderStepUrl =
        (await resolveRemoteUrl(resolved.decoderStepUrl, 'decoder_step.onnx')) ??
        resolved.decoderStepUrl;
      const decoderAlignUrl =
        (await resolveRemoteUrl(resolved.decoderAlignUrl, 'decoder_align.onnx')) ??
        resolved.decoderAlignUrl;

      return {
        ...resolved,
        artifacts: {
          ...resolved.artifacts,
          encoderUrl,
          decoderUrl: decoderInitUrl ?? resolved.artifacts.decoderUrl,
          // Keep tokenizerUrl remote/file based so config.json and
          // generation_config.json remain derivable by sibling path replacement.
          tokenizerUrl: resolved.artifacts.tokenizerUrl,
        },
        decoderInitUrl,
        decoderStepUrl,
        decoderAlignUrl,
        externalData:
          Object.keys(materializedExternalData).length > 0
            ? materializedExternalData
            : undefined,
      };
    }

    if (source.kind !== 'huggingface') {
      return resolved;
    }

    const revision = source.revision ?? 'main';
    let repoFilesPromise: Promise<readonly string[]> | null = null;
    const getRepoFiles = (): Promise<readonly string[]> => {
      repoFilesPromise ??= fetchModelFiles(source.repoId, revision);
      return repoFilesPromise;
    };

    const resolveFile = async (filename: string | undefined): Promise<string | undefined> => {
      if (!filename) return undefined;
      const cacheKey = `huggingface:${source.repoId}:${revision}:${filename}`;
      const cacheKeyFallbacks = (source.cacheKeyFallbackRevisions ?? [])
        .filter((fb) => fb !== revision)
        .map((fb) => `huggingface:${source.repoId}:${fb}:${filename}`);

      const handle = await this.assetProvider!.resolve({
        id: `huggingface:${source.repoId}:${revision}:${filename}`,
        provider: 'huggingface',
        repoId: source.repoId,
        revision,
        filename,
        preferBlobUrl: true,
        cacheKey,
        cacheKeyFallbacks,
        onProgress: (event) => {
          this.runtimeHooks?.onProgress?.(createAssetProgressEvent(this.modelId, filename, event));
        },
      });
      this.assetHandles.push(handle);
      const locator = await handle.getLocator('url');
      if (!locator) {
        throw new Error(`Could not create a URL locator for "${filename}".`);
      }
      return locator;
    };

    const resolveOptionalFile = async (filename: string | undefined): Promise<string | undefined> => {
      if (!filename) return undefined;
      const repoFiles = await getRepoFiles();
      if (repoFiles.length > 0 && !hasListedRepoFile(repoFiles, filename)) {
        return undefined;
      }
      try {
        return await resolveFile(filename);
      } catch (error) {
        if (isAssetMissingError(error)) return undefined;
        throw error;
      }
    };

    void resolveOptionalFile;

    return {
      ...resolved,
      artifacts: {
        ...resolved.artifacts,
        encoderUrl:
          (await resolveFile(resolved.artifacts.encoderUrl.split('/').pop())) ??
          resolved.artifacts.encoderUrl,
        decoderUrl:
          (await resolveFile(resolved.artifacts.decoderUrl.split('/').pop())) ??
          resolved.artifacts.decoderUrl,
        // Keep tokenizerUrl remote/file based so config.json and generation_config.json
        // remain derivable by sibling path replacement.
        tokenizerUrl: resolved.artifacts.tokenizerUrl,
      },
    };
  }

  private async initialize(): Promise<LoadedExecutorState> {
    if (!this.sourceOptions) {
      throw new Error(`No artifact source is configured for "${this.modelId}".`);
    }

    const resolved = await this.materializeResolvedArtifacts(
      resolveWhisperArtifacts(this.sourceOptions, this.backendId),
    );
    const artifacts = resolved.artifacts;

    const ort = await initWhisperOrt(resolved.ortBackend, {
      wasmPaths: resolved.wasmPaths,
      cpuThreads: resolved.cpuThreads,
      enableProfiling: resolved.enableProfiling,
    });

    const tokenizer = await WhisperTokenizer.fromUrl(artifacts.tokenizerUrl);
    const warnings = [...resolved.warnings];

    // Time session creation
    const sessionStart = nowMs();
    // Only create encoder session for now (decoder sessions created below if splitgraph)
    const encoderSession = await createWhisperOrtSession(ort, artifacts.encoderUrl, {
      backendId: resolved.encoderBackendForOrt,
      enableProfiling: resolved.enableProfiling,
      enableGraphCapture: resolved.experimentalWebGpuEncoderGraphCapture,
      // GPU encoder bridge: keep encoder output on GPU to avoid the CPU
      // f32→f16 cast round-trip when using the Cast-injected decoder_init.
      // DIAGNOSTIC (Track A2): encoderOutputCpu forces CPU output to measure
      // cross-session GPU tensor handoff penalty.
      ...((resolved.experimentalGpuKvCache && resolved.encoderBackendForOrt === 'webgpu' && !resolved.encoderOutputCpu)
        ? { preferredOutputLocation: 'gpu-buffer' as const }
        : {}),
      ...(resolved.externalData?.encoder?.[0]
        ? { externalDataUrl: resolved.externalData.encoder[0].dataUrl, externalDataPath: resolved.externalData.encoder[0].path }
        : {}),
    });

    // Merged decoder session — only needed for non-splitgraph path
    let decoderSession: OrtSessionLike | undefined;
    const isSplitGraph = resolved.isSplitGraph;
    if (!isSplitGraph) {
      decoderSession = await createWhisperOrtSession(ort, artifacts.decoderUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        ...(resolved.externalData?.decoder_init?.[0]
          ? { externalDataUrl: resolved.externalData.decoder_init[0].dataUrl, externalDataPath: resolved.externalData.decoder_init[0].path }
          : {}),
      });
    }

    const genConfig = await this.loadGenerationConfig(artifacts);
    const modelConfig = await this.loadModelConfig(artifacts);

    let decoderInitSession: OrtSessionLike | undefined;
    let decoderStepSession: OrtSessionLike | undefined;
    let decoderAlignSession: OrtSessionLike | undefined;
    const decoderInitPreferredOutputLocation =
      resolved.experimentalGpuKvCache && resolved.decoderBackendForOrt === 'webgpu'
        ? createWhisperGpuKvOutputLocation(modelConfig, 'init')
        : undefined;
    const decoderStepPreferredOutputLocation =
      resolved.experimentalGpuKvCache && resolved.decoderBackendForOrt === 'webgpu'
        ? createWhisperGpuKvOutputLocation(modelConfig, 'step')
        : undefined;

    if (isSplitGraph && resolved.decoderInitUrl && resolved.decoderStepUrl) {
      decoderInitSession = await createWhisperOrtSession(ort, resolved.decoderInitUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        preferredOutputLocation: decoderInitPreferredOutputLocation,
        ...(resolved.externalData?.decoder_init?.[0]
          ? { externalDataUrl: resolved.externalData.decoder_init[0].dataUrl, externalDataPath: resolved.externalData.decoder_init[0].path }
          : {}),
      });
      decoderStepSession = await createWhisperOrtSession(ort, resolved.decoderStepUrl, {
        backendId: resolved.decoderBackendForOrt,
        enableProfiling: resolved.enableProfiling,
        preferredOutputLocation: decoderStepPreferredOutputLocation,
        // DIAGNOSTIC (B2-C): graph capture for decoder_step
        ...(resolved.decoderGraphCapture ? { enableGraphCapture: true } : {}),
        // DIAGNOSTIC (B2-B): freeDimensionOverrides for decoder_step
        ...(resolved.decoderFreeDimensionOverrides ? { freeDimensionOverrides: resolved.decoderFreeDimensionOverrides } : {}),
        ...(resolved.externalData?.decoder_step?.[0]
          ? { externalDataUrl: resolved.externalData.decoder_step[0].dataUrl, externalDataPath: resolved.externalData.decoder_step[0].path }
          : {}),
      });
      // Defer decoder_align — only load when needed for alignment (saves VRAM)
    }

    return {
      ort, tokenizer, encoderSession, decoderSession,
      generationConfig: genConfig, modelConfig, warnings,
      isSplitGraph, decoderInitSession, decoderStepSession, decoderAlignSession,
      decoderBackendForOrt: resolved.decoderBackendForOrt,
      experimentalGpuKvCache: resolved.experimentalGpuKvCache,
      sessionCreateMs: nowMs() - sessionStart,
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

  private async runDecoderStep(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    generatedTokens: readonly number[],
    pastKeyValues: Record<string, OrtTensorLike<Float32Array>>,
    isFirstStep: boolean,
  ): Promise<DecoderStepResult> {
    const inputIds = new BigInt64Array(generatedTokens.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, generatedTokens.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const decoderInputNames = loaded.decoderSession!.inputNames ?? [];
    if (decoderInputNames.includes('use_cache_branch')) {
      feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([isFirstStep ? 1 : 0]), [1]);
    }

    if (!isFirstStep) {
      for (const [name, tensor] of Object.entries(pastKeyValues)) {
        feeds[name] = tensor;
      }
    } else {
      const numLayers = loaded.modelConfig.decoderLayers;
      const numHeads = loaded.modelConfig.decoderAttentionHeads;
      const headDim = loaded.modelConfig.headDim;
      const encoderSeqLen = encoderHiddenStates.dims[1] as number;
      for (let i = 0; i < numLayers; i++) {
        feeds[`past_key_values.${i}.decoder.key`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(0),
          [1, numHeads, 0, headDim],
        );
        feeds[`past_key_values.${i}.decoder.value`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(0),
          [1, numHeads, 0, headDim],
        );
        const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
        feeds[`past_key_values.${i}.encoder.key`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(encoderCacheSize),
          [1, numHeads, encoderSeqLen, headDim],
        );
        feeds[`past_key_values.${i}.encoder.value`] = new loaded.ort.Tensor(
          'float32',
          new Float32Array(encoderCacheSize),
          [1, numHeads, encoderSeqLen, headDim],
        );
      }
    }

    const outputs = await loaded.decoderSession!.run(feeds);
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logits = logitsTensor.data;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
    const lastLogitsOffset = logits.length - vocabSize;

    const nextPastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        const pastName = key.replace(/^present/, 'past_key_values');
        nextPastKeyValues[pastName] = value as OrtTensorLike<Float32Array>;
      }
    }

    return {
      lastLogits: logits.subarray(lastLogitsOffset),
      vocabSize,
      pastKeyValues: nextPastKeyValues,
      crossAttentions: extractCrossAttentions(outputs),
    };
  }

  private async runForcedAlignment(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    language: string,
    textTokenIds: number[],
  ): Promise<{
    readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
    readonly logitsForText: Float32Array;
  }> {
    const tokenizer = loaded.tokenizer;
    const sotId = tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
    const langToken = language === 'auto' ? '<|en|>' : `<|${language}|>`;
    const langId = tokenizer.getTokenId(langToken) ?? 50268;
    const taskId = tokenizer.getTokenId('<|transcribe|>') ?? 50359;
    const noTsId = tokenizer.getTokenId('<|notimestamps|>') ?? 50363;
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const forcedIds = [sotId, langId, taskId, noTsId, ...textTokenIds, eosId];
    const inputIds = new BigInt64Array(forcedIds.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, forcedIds.length]);

    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const decoderInputNames = loaded.decoderSession!.inputNames ?? [];
    if (decoderInputNames.includes('use_cache_branch')) {
      feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([1]), [1]);
    }

    // First step: provide empty past_key_values
    // Use config-driven layer/head counts
    const numLayers = loaded.modelConfig.decoderLayers;
    const numHeads = loaded.modelConfig.decoderAttentionHeads;
    const headDim = loaded.modelConfig.headDim;
    const encoderSeqLen = encoderHiddenStates.dims[1] as number;
    for (let i = 0; i < numLayers; i++) {
      feeds[`past_key_values.${i}.decoder.key`] = new loaded.ort.Tensor(
        'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
      feeds[`past_key_values.${i}.decoder.value`] = new loaded.ort.Tensor(
        'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
      const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
      feeds[`past_key_values.${i}.encoder.key`] = new loaded.ort.Tensor(
        'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
      feeds[`past_key_values.${i}.encoder.value`] = new loaded.ort.Tensor(
        'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
    }

    const outputs = await loaded.decoderSession!.run(feeds);
    const crossAttentions = extractCrossAttentions(outputs);

    // Extract logits for the text tokens (skip prompt + EOS)
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const totalVocab = (logitsTensor.dims[logitsTensor.dims.length - 1] as number) ?? 51865;

    // forcedIds: [SOT, lang, task, notimestamps, ...text, EOS]
    const promptLen = 4; // SOT + lang + task + notimestamps
    const textStart = promptLen;
    const textCount = textTokenIds.length;
    const logitsForText = new Float32Array(textCount * totalVocab);
    const srcOffset = textStart * totalVocab;
    logitsForText.set(logitsTensor.data.subarray(srcOffset, srcOffset + textCount * totalVocab));

    return { crossAttentions, logitsForText };
  }

  private async runForcedAlignmentSplitGraph(
    loaded: Required<Pick<LoadedExecutorState, 'decoderAlignSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    allTokenIds: readonly number[],
    _promptLen: number,
  ): Promise<Float32Array> {
    const inputIds = new BigInt64Array(allTokenIds.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, allTokenIds.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const outputs = await loaded.decoderAlignSession.run(feeds);
    const alignKey = Object.keys(outputs)[0]!;
    const alignTensor = outputs[alignKey] as OrtTensorLike<Float32Array>;

    return alignTensor.data;
  }

  private async computeAttentionWordTimestampsSplitGraph(
    loaded: Required<Pick<LoadedExecutorState, 'decoderAlignSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: WhisperTokenizer,
    segments: WhisperNativeSegment[],
    allTokens: readonly number[],
    promptLen: number,
    _options: WhisperSeq2SeqTranscriptionOptions,
  ): Promise<WhisperNativeTranscript['words']> {
    // Collect text token IDs from segments
    const textTokenIds: number[] = [];
    for (const seg of segments) {
      const ids = tokenizer.encode(seg.text);
      for (const id of ids) {
        if (!tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id)) {
          textTokenIds.push(id);
        }
      }
    }
    if (textTokenIds.length === 0) return [];

    try {
      const alignmentData = await this.runForcedAlignmentSplitGraph(
        loaded, encoderHiddenStates, allTokens, promptLen,
      );

      const encoderFrameCount = (encoderHiddenStates.dims[1] as number) ?? 0;
      const frameCount = encoderFrameCount; // decoder_align outputs full encoder seq
      const totalTokens = allTokens.length;

      const dtwTimestamps = processSplitGraphAlignment({
        alignmentData,
        totalTokens,
        promptLen,
        textTokenCount: textTokenIds.length,
        frameCount,
        medianFilterWidth: loaded.modelConfig.medianFilterWidth,
        timePrecisionSeconds: 0.02,
      });

      return this.buildWordsFromDtwTimestamps(
        tokenizer, textTokenIds, dtwTimestamps,
      );
    } catch {
      // Alignment failed — fall back to timestamp-token interpolation
      return buildWhisperWordTimestampsFromTokenDetails(
        [], // No token details with timestamps to build from
        {
          timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
          timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
          language: '',
        },
      );
    }
  }

  private async computeAttentionWordTimestamps(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: WhisperTokenizer,
    tokenDetails: WhisperNativeToken[],
    segments: WhisperNativeSegment[],
    language: string,
    _options: WhisperSeq2SeqTranscriptionOptions,
  ): Promise<WhisperNativeTranscript['words']> {
    const alignmentHeads = loaded.generationConfig.alignmentHeads;
    if (alignmentHeads.length === 0) {
      // No alignment heads configured — fall back to timestamp-token interpolation
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }

    // Collect text token IDs from segments (exclude special/timestamp tokens)
    const textTokenIds: number[] = [];
    for (const seg of segments) {
      const ids = tokenizer.encode(seg.text);
      for (const id of ids) {
        if (!tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id)) {
          textTokenIds.push(id);
        }
      }
    }
    if (textTokenIds.length === 0) return [];

    try {
      const alignment = await this.runForcedAlignment(
        loaded, encoderHiddenStates, language, textTokenIds,
      );
      const crossAttentions = alignment.crossAttentions;
      if (crossAttentions.length === 0) {
        // Decoder had no cross-attention outputs — fall back
        return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
          timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
          timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
          language,
        });
      }

      // Build attention head matrices for DTW — select only alignment_heads
      const encoderFrameCount = (encoderHiddenStates.dims[1] as number) ?? 0;
      const croppedFrames = Math.floor(encoderFrameCount / 2); // Whisper encoder downsamples by 2

      const attentionHeads = alignmentHeads.map(({ layer, head }) => {
        const layerTensor = crossAttentions[layer];
        if (!layerTensor) {
          throw new Error(`Cross-attention layer ${layer} missing from decoder outputs.`);
        }
        const numLayerHeads = (layerTensor.dims[1] as number) ?? 6;
        if (head >= numLayerHeads) {
          throw new Error(`Alignment head ${head} exceeds layer ${layer} head count ${numLayerHeads}.`);
        }
        const totalTokens = (layerTensor.dims[2] as number) ?? 0;
        const totalFramesPerHead = (layerTensor.dims[3] as number) ?? 0;
        // Extract single head: tensor has shape [batch=1, heads, tokens, frames]
        const headSize = totalTokens * totalFramesPerHead;
        const headOffset = head * headSize;
        const headValues = layerTensor.data.subarray(headOffset, headOffset + headSize);
        return {
          values: new Float32Array(headValues),
          tokenCount: totalTokens,
          frameCount: croppedFrames,
        };
      });

      const dtwTimestamps = computeWhisperDtwTokenTimestamps({
        attentionHeads,
        tokenCount: textTokenIds.length,
        frameCount: croppedFrames,
        timePrecisionSeconds: 0.02,
      });

      // Compute token logprobs from forced alignment logits
      const logits = alignment.logitsForText;
      const totalVocab = (logits?.length ?? 0) / textTokenIds.length;
      let tokenLogprobs: Float32Array | undefined;
      if (logits && totalVocab > 0) {
        tokenLogprobs = new Float32Array(textTokenIds.length);
        for (let t = 0; t < textTokenIds.length; t++) {
          const tokenId = textTokenIds[t] ?? 0;
          // logprob = log(softmax(logits[t]))[tokenId]
          // = logits[t][tokenId] - logsumexp(logits[t])
          let maxVal = -Infinity;
          const tStart = t * totalVocab;
          for (let v = 0; v < totalVocab; v++) {
            const val = logits[tStart + v] ?? -Infinity;
            if (val > maxVal) maxVal = val;
          }
          let sumExp = 0;
          for (let v = 0; v < totalVocab; v++) {
            sumExp += Math.exp((logits[tStart + v] ?? -Infinity) - maxVal);
          }
          const logProb = (logits[tStart + tokenId] ?? -Infinity) - maxVal - Math.log(sumExp);
          tokenLogprobs[t] = logProb;
        }
      }

      // Build word timestamps from token timestamps using tokenizer
      return this.buildWordsFromDtwTimestamps(
        tokenizer, textTokenIds, dtwTimestamps, tokenLogprobs,
      );
    } catch {
      // Forced alignment failed — fall back to timestamp-token interpolation
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }
  }

  private buildWordsFromDtwTimestamps(
    tokenizer: WhisperTokenizer,
    textTokenIds: number[],
    dtwTimestamps: readonly number[],
    tokenLogprobs?: Float32Array,
  ): WhisperNativeTranscript['words'] {
    // DTW gives tokenCount+1 timestamps — start of each text token + end of last
    const words: WhisperNativeWord[] = [];
    const text = tokenizer.decode(textTokenIds, { skipSpecialTokens: true });
    const wordTexts = text.split(/\s+/).filter((w) => w.length > 0);
    if (wordTexts.length === 0) return undefined;

    // Approximate word boundaries from token-to-word mapping using tokenizer encode
    const allWordTokenIds: number[][] = [];
    for (const w of wordTexts) {
      const ids = tokenizer.encode(w);
      if (ids.length > 0) allWordTokenIds.push(ids);
    }

    let tokenOffset = 0;
    for (let wi = 0; wi < allWordTokenIds.length; wi++) {
      const wIds = allWordTokenIds[wi]!;
      if (wIds.length === 0) continue;
      const wordStartIdx = tokenOffset;
      const wordEndIdx = tokenOffset + wIds.length - 1;
      const startTime = dtwTimestamps[wordStartIdx] ?? 0;
      const endTime = dtwTimestamps[wordEndIdx + 1] ?? dtwTimestamps[wordStartIdx] ?? 0;

      // Compute word probability as mean of token probabilities
      let confidence = -1;
      if (tokenLogprobs && wIds.length > 0) {
        let probSum = 0;
        for (let t = wordStartIdx; t <= wordEndIdx; t++) {
          probSum += Math.exp(tokenLogprobs[t] ?? -Infinity);
        }
        confidence = probSum / wIds.length;
      }

      words.push({
        index: wi,
        text: wordTexts[wi]!,
        startTime,
        endTime,
        confidence,
        tokenIds: wIds,
      });
      tokenOffset += wIds.length;
    }
    return words.length > 0 ? words : undefined;
  }

  private async loadGenerationConfig(
    artifacts: { readonly tokenizerUrl: string },
  ): Promise<WhisperGenerationConfig> {
    try {
      const genConfigUrl = artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'generation_config.json');
      const text = await fetchText(genConfigUrl);
      const json = JSON.parse(text) as Record<string, unknown>;
      return parseWhisperGenerationConfig(json);
    } catch {
      return parseWhisperGenerationConfig({});
    }
  }

  private async loadModelConfig(
    artifacts: { readonly tokenizerUrl: string },
  ): Promise<WhisperModelConfig> {
    try {
      const configUrl = artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'config.json');
      const text = await fetchText(configUrl);
      const json = JSON.parse(text) as Record<string, unknown>;
      const config = parseWhisperModelConfig(json);

      // Also try generation_config.json for num_mel_bins (not in HF config.json)
      if (!config.numMelBins) {
        try {
          const genUrl = artifacts.tokenizerUrl.replace(/tokenizer\.json$/, 'generation_config.json');
          const genText = await fetchText(genUrl);
          const genJson = JSON.parse(genText) as Record<string, unknown>;
          if (typeof genJson.num_mel_bins === 'number') {
            return { ...config, numMelBins: genJson.num_mel_bins };
          }
        } catch { /* ignore */ }
      }
      return config;
    } catch {
      return parseWhisperModelConfig({});
    }
  }

  private async runDecoderInit(
    loaded: Required<Pick<LoadedExecutorState, 'decoderInitSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    promptTokens: readonly number[],
  ): Promise<{
    logits: Float32Array;
    vocabSize: number;
    presentKv: Record<string, OrtTensorLike<Float32Array>>;
    timings: DecoderSessionTiming;
  }> {
    const inputStart = nowMs();
    const inputIds = new BigInt64Array(promptTokens.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, promptTokens.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };
    const inputLocations = countTensorLocations(Object.values(feeds));

    const runStart = nowMs();
    const outputs = await loaded.decoderInitSession.run(feeds);
    const outputStart = nowMs();
    const outputLocations = countTensorLocations(Object.values(outputs));
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logitsReadStart = nowMs();
    const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
    const logitReadMs = nowMs() - logitsReadStart;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;

    const kvStart = nowMs();
    const presentKv: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        presentKv[key] = value as OrtTensorLike<Float32Array>;
      }
    }
    const kvExtractMs = nowMs() - kvStart;

    const outputEnd = nowMs();
    return {
      logits: logitsData.data,
      vocabSize,
      presentKv,
      timings: {
        inputMs: runStart - inputStart,
        runMs: outputStart - runStart,
        outputMs: outputEnd - outputStart,
        tensorCreateMs: runStart - inputStart,
        logitReadMs,
        kvExtractMs,
        gpuInputCount: inputLocations.gpu,
        cpuInputCount: inputLocations.cpu,
        gpuOutputCount: outputLocations.gpu,
        cpuOutputCount: outputLocations.cpu,
        gpuDownloadCount: logitsData.downloaded ? 1 : 0,
      },
    };
  }

  private async runDecoderStepSplit(
    loaded: Required<Pick<LoadedExecutorState, 'decoderStepSession' | 'ort'>> & LoadedExecutorState,
    tokenId: number,
    pastKv: Record<string, OrtTensorLike<Float32Array>>,
  ): Promise<{
    logits: Float32Array;
    vocabSize: number;
    /** GPU ArgMax: pre-computed next token ID from model output (undefined if not exported). */
    nextTokenId?: number;
    presentKv: Record<string, OrtTensorLike<Float32Array>>;
    timings: DecoderSessionTiming;
  }> {
    const inputStart = nowMs();
    const inputIdsTensor = new loaded.ort.Tensor('int64', new BigInt64Array([BigInt(tokenId)]), [1, 1]);
    const feeds: Record<string, unknown> = { input_ids: inputIdsTensor };

    // Add all past_key_values (decoder + encoder KV). Step model expects both.
    // CRITICAL: Clone tensor data for cross-session safety. ORT WASM cannot
    // reuse tensor objects from one session as inputs to another.
    for (const [name, tensor] of Object.entries(pastKv)) {
      if (isGpuBufferTensor(tensor)) {
        feeds[name] = tensor;
      } else {
        const isFloat16 = tensor.type === 'float16' || tensor.data.constructor.name === 'Float16Array';
        const TypedArrayCtor = tensor.data.constructor as { new(buffer: ArrayBufferLike, byteOffset: number, length: number): ArrayBufferView };
        const rawData = new TypedArrayCtor(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);
        feeds[name] = new loaded.ort.Tensor(isFloat16 ? 'float16' : 'float32', rawData, tensor.dims);
      }
    }
    const inputLocations = countTensorLocations(Object.values(feeds));

    const runStart = nowMs();
    const outputs = await loaded.decoderStepSession.run(feeds);
    const outputStart = nowMs();
    const outputLocations = countTensorLocations(Object.values(outputs));
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const logitReadStart = nowMs();
    const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
    const logitReadMs = nowMs() - logitReadStart;
    const logitsDims = logitsTensor.dims;
    const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;

    // decoder_step outputs only self-attention present KV. Merge with encoder KV from input.
    const kvStart = nowMs();
    const presentKv: Record<string, OrtTensorLike<Float32Array>> = {};
    for (const [key, value] of Object.entries(outputs)) {
      if (key.startsWith('present')) {
        const pastName = key.replace(/^present/, 'past_key_values');
        presentKv[pastName] = value as OrtTensorLike<Float32Array>;
      }
    }
    // Preserve encoder KV from input (they don't change and step model doesn't output them)
    for (const [key, value] of Object.entries(pastKv)) {
      if (key.includes('encoder') && !presentKv[key]) {
        presentKv[key] = value;
      }
    }

    // GPU ArgMax: if the model exports next_token_id, read it directly.
    let nextTokenId: number | undefined;
    const nextTokenTensor = outputs['next_token_id'] as OrtTensorLike<Int32Array> | undefined;
    if (nextTokenTensor && nextTokenTensor.data) {
      // CPU tensor — read directly (4 bytes, not a download)
      nextTokenId = nextTokenTensor.data[0];
    }

    const kvEnd = nowMs();
    const outputEnd = nowMs();
    return {
      logits: logitsData.data,
      vocabSize,
      nextTokenId,
      presentKv,
      timings: {
        inputMs: runStart - inputStart,
        runMs: outputStart - runStart,
        outputMs: outputEnd - outputStart,
        tensorCreateMs: runStart - inputStart,
        logitReadMs,
        kvExtractMs: kvEnd - kvStart,
        gpuInputCount: inputLocations.gpu,
        cpuInputCount: inputLocations.cpu,
        gpuOutputCount: outputLocations.gpu,
        cpuOutputCount: outputLocations.cpu,
        gpuDownloadCount: logitsData.downloaded ? 1 : 0,
      },
    };
  }

  async transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];

    if (loaded.isSplitGraph && loaded.decoderInitSession && loaded.decoderStepSession) {
      return this.transcribeWithSplitGraph(audio, options, _context);
    }

    if (this.shouldChunkAudio(audio, options)) {
      return this.transcribeLongAudio(audio, options, _context);
    }

    // 1. Preprocess audio to mel spectrogram
    const melBins = loaded.modelConfig.numMelBins ?? this.config.melBins;
    const melProcessor = new WhisperMelProcessor({ nMels: melBins });
    // Audio is already normalized to mono by the session before calling executor
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const melResult = melProcessor.process(pcmData);
    // Whisper conv layers downsample by 2x: input 3000 frames → output 1500 time positions.
    const encoderOutputPositions = this.config.maxSourcePositions;
    const melInputFrames = encoderOutputPositions <= 1500 ? encoderOutputPositions * 2 : encoderOutputPositions;
    const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, melInputFrames);

    // Reshape to [1, n_mels, melInputFrames] channels-first
    const featureTensor = new loaded.ort.Tensor(
      'float32',
      paddedFeatures,
      [1, melBins, melInputFrames],
    );

    // 2. Run encoder
    const encoderOutputs = await loaded.encoderSession.run({
      input_features: featureTensor,
    });
    const encoderHiddenStates = await maybeCastEncoderHiddenStates(
      encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>,
      loaded.decoderInitSession ?? loaded.decoderSession!,
      loaded.ort,
    );

    // 3. Build initial decoder input IDs
    const tokenizer = loaded.tokenizer;
    const language = options.language ?? this.config.languages[0] ?? 'auto';
    const langToken = language === 'auto' ? '<|en|>' : `<|${language}|>`;
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      tokenizer.getTokenId(langToken) ?? 50268,
      tokenizer.getTokenId(taskToken) ?? 50359,
    ];
    if (noTimestampsToken) {
      const ntId = tokenizer.getTokenId(noTimestampsToken);
      if (ntId !== undefined) {
        promptTokens.push(ntId);
      }
    }

    // 4. Decode loop (greedy by default, beam search when numBeams > 1)
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const maxNewTokens = options.maxNewTokens ?? this.config.maxTargetPositions ?? 448;
    const numBeams = Math.max(1, Math.floor(options.numBeams ?? 1));
    const lengthPenalty = options.lengthPenalty ?? 0;
    const beamCandidateWidth = Math.max(
      numBeams,
      Math.ceil(numBeams * Math.max(1, options.patience ?? 1)),
    );
    let tokenDetails: WhisperNativeToken[] = [];

    // Build timestamp logit processor
    const timestampBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const timestampProcessor = new WhisperTimestampLogitProcessor({
      eosTokenId: eosId,
      noTimestampsTokenId: loaded.generationConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
      timestampBegin,
      suppressTokens: loaded.generationConfig.suppressTokens ?? [],
      beginSuppressTokens: loaded.generationConfig.beginSuppressTokens ?? [],
    });

    if (numBeams === 1) {
      const generatedTokens: number[] = [...promptTokens];
      let pastKeyValues: Record<string, OrtTensorLike<Float32Array>> = {};

      for (let step = 0; step < maxNewTokens; step++) {
        const result = await this.runDecoderStep(
          loaded,
          encoderHiddenStates,
          generatedTokens,
          pastKeyValues,
          step === 0,
        );
        pastKeyValues = result.pastKeyValues;
        timestampProcessor.process(result.lastLogits, generatedTokens, promptTokens.length);
        const nextTokenId = argmax(result.lastLogits);
        generatedTokens.push(nextTokenId);

        const { confidence } = confidenceFromLogits(
          new Float32Array(result.lastLogits),
          nextTokenId,
          result.vocabSize,
        );

        const tokenText = this.formatTokenText(tokenizer, nextTokenId);
        tokenDetails.push({
          index: step,
          id: nextTokenId,
          text: tokenText,
          confidence,
          special: tokenizer.isSpecialTokenId(nextTokenId),
        });

        if (nextTokenId === eosId) break;
      }
    } else {
      let beams: WhisperBeamState<BeamPayload>[] = [
        createInitialWhisperBeam(promptTokens, 0, { tokenDetails: [], pastKeyValues: {} }),
      ];

      for (let step = 0; step < maxNewTokens; step++) {
        const activeBeams = beams.filter((beam) => !beam.completed);
        if (activeBeams.length === 0) break;

        const logitsByBeam: Float32Array[] = [];
        const nextPastByBeam = new Map<WhisperBeamState<BeamPayload>, Record<string, OrtTensorLike<Float32Array>>>();
        let vocabSize = 0;

        for (const beam of beams) {
          if (beam.completed) {
            logitsByBeam.push(new Float32Array(0));
            continue;
          }
          const result = await this.runDecoderStep(
            loaded,
            encoderHiddenStates,
            beam.tokens,
            beam.payload?.pastKeyValues ?? {},
            step === 0,
          );
          timestampProcessor.process(result.lastLogits, beam.tokens, promptTokens.length);
          logitsByBeam.push(result.lastLogits);
          nextPastByBeam.set(beam, result.pastKeyValues);
          vocabSize = result.vocabSize;
        }

        beams = rankWhisperBeamCandidates({
          beams,
          logitsByBeam,
          beamWidth: beamCandidateWidth,
          eosTokenId: eosId,
          lengthPenalty,
          expandPayload: (beam, tokenId) => {
            const { confidence } = confidenceFromLogits(
              logitsByBeam[beams.indexOf(beam)] ?? new Float32Array(0),
              tokenId,
              vocabSize,
            );
            const tokenText = this.formatTokenText(tokenizer, tokenId);
            return {
              tokenDetails: [
                ...(beam.payload?.tokenDetails ?? []),
                {
                  index: step,
                  id: tokenId,
                  text: tokenText,
                  confidence,
                  special: tokenizer.isSpecialTokenId(tokenId),
                },
              ],
              pastKeyValues: nextPastByBeam.get(beam) ?? {},
            };
          },
        });
      }

      tokenDetails = [...(selectBestWhisperBeam(beams, lengthPenalty)?.payload?.tokenDetails ?? [])];
    }

    // 5. Build segments from decoded tokens
    const segments = this.buildSegments(tokenDetails, tokenizer, options.noTimestamps);
    const words = this.shouldReturnWordTimestamps(options)
      ? await this.computeAttentionWordTimestamps(
          loaded,
          encoderHiddenStates,
          tokenizer,
          tokenDetails,
          segments,
          language,
          options,
        )
      : [];
    const utteranceText = segments.map((s) => s.text).join(' ').trim();

    return {
      utteranceText,
      isFinal: true,
      language,
      segments,
      ...(words && words.length > 0 ? { words } : {}),
      tokens: options.returnSpecialTokens
        ? tokenDetails
        : tokenDetails.filter((t) => !t.special),
      warnings,
    };
  }

  private buildSegments(
    tokens: WhisperNativeToken[],
    tokenizer: WhisperTokenizer,
    noTimestamps?: boolean,
  ): WhisperNativeSegment[] {
    if (noTimestamps) {
      // No timestamps: single segment with all text
      const text = tokenizer.decode(
        tokens.map((t) => t.id ?? 0),
        { skipSpecialTokens: true },
      );
      if (!text.trim()) return [];
      return [
        {
          index: 0,
          text,
          startTime: 0,
          endTime: 30,
          confidence: tokens.length > 0 ? (tokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / tokens.length) : 0,
        },
      ];
    }

    // With timestamps: split on timestamp tokens
    const segments: WhisperNativeSegment[] = [];
    let currentTokens: WhisperNativeToken[] = [];
    let segmentStart = 0;

    for (const token of tokens) {
      if (tokenizer.isTimestampTokenId(token.id ?? 0)) {
        const ts = tokenizer.timestampTokenIdToSeconds(token.id ?? 0);
        if (ts !== undefined) {
          if (currentTokens.length > 0) {
            const text = tokenizer.decode(
              currentTokens.map((t) => t.id ?? 0),
              { skipSpecialTokens: true },
            );
            if (text.trim()) {
              const avgConf =
                currentTokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / currentTokens.length;
              segments.push({
                index: segments.length,
                text,
                startTime: segmentStart,
                endTime: ts,
                confidence: avgConf,
              });
            }
            currentTokens = [];
          }
          segmentStart = ts;
        }
      } else {
        currentTokens.push(token);
      }
    }

    // Remaining tokens
    if (currentTokens.length > 0) {
      const text = tokenizer.decode(
        currentTokens.map((t) => t.id ?? 0),
        { skipSpecialTokens: true },
      );
      if (text.trim()) {
        const avgConf =
          currentTokens.reduce((s, t) => s + (t.confidence ?? 0), 0) / currentTokens.length;
        segments.push({
          index: segments.length,
          text,
          startTime: segmentStart,
          endTime: 30,
          confidence: avgConf,
        });
      }
    }

    return segments;
  }

  private shouldReturnWordTimestamps(options: WhisperSeq2SeqTranscriptionOptions): boolean {
    return options.returnWords === true || options.returnTimestamps === 'word' || options.detail === 'words' || options.detail === 'detailed';
  }

  private shouldChunkAudio(audio: AudioBufferLike, options: WhisperSeq2SeqTranscriptionOptions): boolean {
    if (options.windowing === 'disabled' || options.unsafeAllowOverMaxWindow) return false;
    const maxDuration = options.chunkLengthSeconds ?? options.maxInputDurationSeconds ?? 30;
    return audio.durationSeconds > maxDuration;
  }

  /**
   * Detect language from encoder output using decoder_init with single start token.
   * Returns language code (e.g. 'en', 'tr') or 'auto' if detection fails.
   */
  private async detectLanguageFromEncoder(
    loaded: Required<LoadedExecutorState>,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
  ): Promise<string> {
    try {
      const sotId = loaded.tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
      const inputIds = new BigInt64Array([BigInt(sotId)]);
      const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, 1]);
      const feeds: Record<string, unknown> = {
        input_ids: inputIdsTensor,
        encoder_hidden_states: encoderHiddenStates,
      };
      const outputs = await loaded.decoderInitSession.run(feeds);
      try {
        const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
        const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
        const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
        const vocabSize = logitsTensor.dims[logitsTensor.dims.length - 1] ?? 0;

        // Language tokens span 50259-50357 in Whisper vocabulary
        const logits = logitsData.data;
        let maxLogit = -Infinity;
        let maxLangToken = -1;
        for (let i = 50259; i <= 50357 && i < vocabSize; i++) {
          if (logits[i]! > maxLogit) {
            maxLogit = logits[i]!;
            maxLangToken = i;
          }
        }

        // Decode the language token to get the code
        if (maxLangToken > 0) {
          const langToken = loaded.tokenizer.idsToTokens?.([maxLangToken])?.[0] ?? '';
          const match = langToken.match(/<\|(\w+)\|>/);
          if (match) return match[1]!;
        }
      } finally {
        for (const output of Object.values(outputs)) {
          disposeGpuTensor(output);
        }
      }

      return 'auto';
    } catch {
      return 'auto';
    }
  }

  private async runGreedyGpuKvDecode(params: {
    readonly loaded: Required<Pick<LoadedExecutorState, 'decoderInitSession' | 'decoderStepSession' | 'ort'>> & LoadedExecutorState;
    readonly encoderHiddenStates: OrtTensorLike<Float32Array>;
    readonly promptTokens: readonly number[];
    readonly eosTokenId: number;
    readonly maxNewTokens: number;
    readonly processLogits?: (logits: Float32Array, generatedTokens: readonly number[], beginIndex: number) => void;
    readonly onTokenLogits?: WhisperSeq2SeqTranscriptionOptions['onTokenLogits'];
    readonly onInitTiming?: (timings: DecoderSessionTiming, elapsedMs: number) => void;
    readonly onStepTiming?: (timings: DecoderSessionTiming, elapsedMs: number) => void;
  }): Promise<SplitGraphDecodeResult> {
    const {
      loaded,
      encoderHiddenStates,
      promptTokens,
      eosTokenId,
      maxNewTokens,
      processLogits,
      onTokenLogits,
      onInitTiming,
      onStepTiming,
    } = params;

    const initStart = nowMs();
    const init = await this.runDecoderInit(loaded, encoderHiddenStates, promptTokens);
    onInitTiming?.(init.timings, nowMs() - initStart);

    const vocabSize = init.vocabSize;
    const firstLogits = init.logits.subarray(init.logits.length - vocabSize);
    processLogits?.(firstLogits, promptTokens, promptTokens.length);

    const firstTokenId = argmax(firstLogits);
    const tokens: number[] = [firstTokenId];
    onTokenLogits?.(firstTokenId, firstLogits, { tokens, beginIndex: promptTokens.length });

    let pastKv = mapPresentKvToPastKv(init.presentKv);
    try {
      if (firstTokenId === eosTokenId) {
        return { tokens };
      }

      for (let stepIndex = 1; stepIndex < maxNewTokens; stepIndex++) {
        const stepStart = nowMs();
        const step = await this.runDecoderStepSplit(loaded, tokens[tokens.length - 1]!, pastKv);
        onStepTiming?.(step.timings, nowMs() - stepStart);

        const previousKv = pastKv;
        pastKv = step.presentKv;
        disposeReplacedGpuKv(previousKv, pastKv);

        processLogits?.(step.logits, [...promptTokens, ...tokens], promptTokens.length);
        // GPU ArgMax: use model-computed next_token_id when available, fall back to JS argmax
        const nextTokenId = step.nextTokenId ?? argmax(step.logits);
        tokens.push(nextTokenId);
        onTokenLogits?.(nextTokenId, step.logits, { tokens, beginIndex: promptTokens.length });

        if (nextTokenId === eosTokenId) {
          break;
        }
      }

      return { tokens };
    } finally {
      disposeGpuKv(pastKv);
    }
  }

  private async transcribeWithSplitGraph(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    _context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const transcriptionStart = nowMs();
    const loaded = await this.getLoadedState();
    const warnings = [...loaded.warnings];
    const splitLoaded = loaded as Required<LoadedExecutorState>;

    emitTranscriptionProgress(options, {
      stage: 'start',
      progress: 0,
      elapsedMs: 0,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Starting transcription for ${this.modelId}.`,
    });

    // 1. Preprocess audio to mel spectrogram
    const preprocessStart = nowMs();
    const melBins = loaded.modelConfig.numMelBins ?? this.config.melBins;
    const melProcessor = new WhisperMelProcessor({ nMels: melBins });
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const melResult = melProcessor.process(pcmData);
    // Whisper conv layers downsample by 2x: input 3000 frames → output 1500 time positions.
    // config.maxSourcePositions is encoder output positions (1500); mel input needs 2x.
    const encoderOutputPositions = this.config.maxSourcePositions;
    const melInputFrames = encoderOutputPositions <= 1500 ? encoderOutputPositions * 2 : encoderOutputPositions;
    const paddedFeatures = WhisperMelProcessor.padToFrames(melResult, melInputFrames);
    const preprocessMs = nowMs() - preprocessStart;
    const preprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'preprocess',
      progress: 0.2,
      elapsedMs: roundMetric(preprocessElapsedMs),
      remainingMs: estimateRemainingMs(preprocessElapsedMs, 0.2),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Prepared Whisper mel features for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        audioDurationSec: roundMetric(audio.durationSeconds, 4),
      },
    });

    const featureTensor = new loaded.ort.Tensor(
      'float32', paddedFeatures,
      [1, melBins, melInputFrames],
    );

    // 2. Run encoder
    const encodeStart = nowMs();
    const encoderRunStart = nowMs();
    const encoderOutputs = await loaded.encoderSession.run({ input_features: featureTensor });
    const encoderRunEnd = nowMs();
    const encoderHiddenStates = await maybeCastEncoderHiddenStates(
      encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>,
      loaded.decoderInitSession ?? loaded.decoderSession!,
      loaded.ort,
    );
    const encoderOutputEnd = nowMs();
    const encodeMs = nowMs() - encodeStart;
    // DIAGNOSTIC: sub-timing for encoder run vs output processing
    const encoderRunMs = encoderRunEnd - encoderRunStart;
    const encoderOutputMs = encoderOutputEnd - encoderRunEnd;
    const encoderOutputLocation = (encoderHiddenStates as OrtTensorLike<Float32Array>).location ?? 'cpu';
    const encoderOutputDtype = (encoderHiddenStates as OrtTensorLike<Float32Array>).type ?? 'float32';
    const encoderFrameCount = encoderHiddenStates.dims[1] ?? encoderOutputPositions;
    const encodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'encode',
      progress: 0.4,
      elapsedMs: roundMetric(encodeElapsedMs),
      remainingMs: estimateRemainingMs(encodeElapsedMs, 0.4),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Encoded Whisper frames for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        encoderFrameCount,
      },
    });

    // 3. Detect language if auto
    const tokenizer = loaded.tokenizer;
    let language = options.language ?? 'auto';
    let languageDetectionMs = 0;
    if (language === 'auto' && loaded.isSplitGraph && loaded.decoderInitSession) {
      const languageDetectionStart = nowMs();
      language = await this.detectLanguageFromEncoder(loaded as Required<LoadedExecutorState>, encoderHiddenStates);
      languageDetectionMs = nowMs() - languageDetectionStart;
    }
    if (language === 'auto') {
      language = this.config.languages[0] ?? 'en';
    }
    const langToken = `<|${language}|>`;
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      tokenizer.getTokenId(langToken) ?? 50268,
      tokenizer.getTokenId(taskToken) ?? 50359,
    ];
    if (noTimestampsToken) {
      const ntId = tokenizer.getTokenId(noTimestampsToken);
      if (ntId !== undefined) promptTokens.push(ntId);
    }
    // Append extra prompt tokens for condition_on_previous_text
    if (options.extraPromptTokens && options.extraPromptTokens.length > 0) {
      promptTokens.push(...options.extraPromptTokens);
    }

    // 4. Run 4-graph decode loop
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const maxNewTokens = options.maxNewTokens ?? this.config.maxTargetPositions ?? 448;

    // Greedy or beam search supported via splitgraph
    const timestampBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const splitTimestampProcessor = new WhisperTimestampLogitProcessor({
      eosTokenId: eosId,
      noTimestampsTokenId: loaded.generationConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
      timestampBegin,
      suppressTokens: loaded.generationConfig.suppressTokens ?? [],
      beginSuppressTokens: loaded.generationConfig.beginSuppressTokens ?? [],
    });

    // Tensor dimension storage for KV cache bridge (init→step tensor reconstruction)
    let kvDims: Record<string, readonly number[]> = {};
    let kvDtype: 'float32' | 'float16' = 'float32';
    let decoderInitMs = 0;
    let decoderInitInputMs = 0;
    let decoderInitRunMs = 0;
    let decoderInitOutputMs = 0;
    let decoderStepMs = 0;
    let decoderStepFeedBuildMs = 0;
    let decoderStepTensorCloneMs = 0;
    let decoderStepRunMs = 0;
    let decoderStepOutputMs = 0;
    let decoderLogitProcessMs = 0;
    let decoderStepCount = 0;
    let decoderGpuTensorInputs = 0;
    let decoderCpuTensorInputs = 0;
    let decoderGpuTensorOutputs = 0;
    let decoderCpuTensorOutputs = 0;
    let decoderGpuTensorDownloads = 0;
    // ── Profiling: fine-grained timing buckets ──
    let encoderTensorCreateMs = 0;
    let encoderOutputReadMs = 0;
    let decoderInitTensorCreateMs = 0;
    let decoderInitLogitReadMs = 0;
    let decoderInitKvExtractMs = 0;
    let decoderStepTensorCreateMs = 0;
    let decoderStepLogitReadMs = 0;
    let decoderStepKvMergeMs = 0;
    let decoderStepKvDisposeMs = 0;
    let sessionCreateMs = 0;
    const decoderStepTimings: number[] = [];
    const requestedNumBeams = Math.max(1, Math.floor(options.numBeams ?? 1));
    const requestedBestOf = Math.max(1, Math.floor(options.bestOf ?? 1));
    const requestedTemperature = options.temperature ?? 0;
    const useExperimentalGpuKvCache = Boolean(
      splitLoaded.experimentalGpuKvCache &&
      splitLoaded.decoderBackendForOrt === 'webgpu',
    );
    const requestedDecoderKvCacheLocation = useExperimentalGpuKvCache ? 'gpu-buffer' : 'cpu';
    const recordDecoderTiming = (timings: DecoderSessionTiming): void => {
      decoderGpuTensorInputs += timings.gpuInputCount;
      decoderCpuTensorInputs += timings.cpuInputCount;
      decoderGpuTensorOutputs += timings.gpuOutputCount;
      decoderCpuTensorOutputs += timings.cpuOutputCount;
      decoderGpuTensorDownloads += timings.gpuDownloadCount;
    };

    if (
      useExperimentalGpuKvCache &&
      (requestedNumBeams > 1 || requestedBestOf > 1 || requestedTemperature > 0)
    ) {
      throw new Error(
        'experimentalGpuKvCache currently supports only greedy argmax decoding; disable it for beam search, best_of, or temperature sampling.',
      );
    }

    const decoderStart = nowMs();
    const processSplitGraphLogits = (logits: Float32Array, genTokens: readonly number[], beginIdx: number): void => {
      const logitsStart = nowMs();
      splitTimestampProcessor.process(logits, genTokens, beginIdx);
      decoderLogitProcessMs += nowMs() - logitsStart;
    };
    const result = useExperimentalGpuKvCache
      ? await this.runGreedyGpuKvDecode({
          loaded: splitLoaded,
          encoderHiddenStates,
          promptTokens,
          eosTokenId: eosId,
          maxNewTokens,
          processLogits: processSplitGraphLogits,
          onTokenLogits: options.onTokenLogits,
          onInitTiming: (timings, elapsedMs) => {
            decoderInitMs += elapsedMs;
            decoderInitInputMs += timings.inputMs;
            decoderInitRunMs += timings.runMs;
            decoderInitOutputMs += timings.outputMs;
            decoderInitTensorCreateMs += timings.tensorCreateMs ?? 0;
            decoderInitLogitReadMs += timings.logitReadMs ?? 0;
            decoderInitKvExtractMs += timings.kvExtractMs ?? 0;
            recordDecoderTiming(timings);
          },
          onStepTiming: (timings, elapsedMs) => {
            decoderStepMs += elapsedMs;
            decoderStepTensorCloneMs += timings.inputMs;
            decoderStepRunMs += timings.runMs;
            decoderStepOutputMs += timings.outputMs;
            decoderStepTensorCreateMs += timings.tensorCreateMs ?? 0;
            decoderStepLogitReadMs += timings.logitReadMs ?? 0;
            decoderStepKvMergeMs += timings.kvExtractMs ?? 0;
            decoderStepCount += 1;
            decoderStepTimings.push(elapsedMs);
            recordDecoderTiming(timings);
          },
        })
      : await splitGraphDecodeLoop({
          promptTokens,
          encoderHiddenStates: encoderHiddenStates.data,
          eosTokenId: eosId,
          maxNewTokens,
          modelConfig: loaded.modelConfig,
          processLogits: processSplitGraphLogits,
          onTokenLogits: options.onTokenLogits,
          numBeams: requestedNumBeams,
          lengthPenalty: options.lengthPenalty ?? 0,
          patience: options.patience ?? 1,
          temperature: requestedTemperature,
          bestOf: requestedBestOf,
          runInit: async (prompt, _encHs, _dims) => {
            const decoderInitStart = nowMs();
            const init = await this.runDecoderInit(splitLoaded, encoderHiddenStates, prompt);
            const decoderInitTotal = nowMs() - decoderInitStart;
            decoderInitMs += decoderInitTotal;
            decoderInitInputMs += init.timings.inputMs;
            decoderInitRunMs += init.timings.runMs;
            decoderInitOutputMs += init.timings.outputMs;
            recordDecoderTiming(init.timings);
            // Store tensor dims for runStep reconstruction (init→step tensor bridging).
            // Store both present.* (init output) and past_key_values.* (step output) formats.
            kvDims = {};
            for (const [k, v] of Object.entries(init.presentKv)) {
              kvDims[k] = v.dims;                                         // present.0.decoder.key
              kvDims[k.replace(/^present\./, 'past_key_values.')] = v.dims; // past_key_values.0.decoder.key
            }
            // Detect KV dtype from init output (fp16 models may expose Float16Array or Uint16Array data)
            const firstKv = Object.values(init.presentKv)[0];
            if (firstKv) {
              kvDtype = firstKv.type === 'float16' || firstKv.data.constructor.name === 'Float16Array' ? 'float16' : 'float32';
            }
            return {
              logits: init.logits,
              vocabSize: init.vocabSize,
              presentKv: Object.fromEntries(
                Object.entries(init.presentKv).map(([k, v]) => [k, v.data]),
              ),
            };
          },
          runStep: async (tokenId, pastKv) => {
            const decoderStepStart = nowMs();
            // Reconstruct tensors from raw data + stored dims.
            // Init outputs present.* prefix; step model expects past_key_values.* prefix.
            // Convert prefix and clone tensor data for cross-session safety.
            const feedBuildStart = nowMs();
            const feeds: Record<string, OrtTensorLike<Float32Array>> = {};
            for (const [name, data] of Object.entries(pastKv)) {
              const stepName = name.replace(/^present\./, 'past_key_values.');
              // Try multiple key formats for dims lookup (init uses present.*, step uses past_key_values.*)
              const dims = kvDims[name] ?? kvDims[stepName] ?? kvDims[name.replace(/^past_key_values\./, 'present.')];
              if (dims) {
                feeds[stepName] = new splitLoaded.ort.Tensor(kvDtype, data, dims) as unknown as OrtTensorLike<Float32Array>;
              } else {
                const numHeads = splitLoaded.modelConfig.decoderAttentionHeads;
                const headDim = splitLoaded.modelConfig.headDim;
                const seqLen = Math.round(data.length / (numHeads * headDim));
                feeds[stepName] = new splitLoaded.ort.Tensor(kvDtype, data, [1, numHeads, seqLen, headDim]) as unknown as OrtTensorLike<Float32Array>;
              }
            }
            decoderStepFeedBuildMs += nowMs() - feedBuildStart;
            const step = await this.runDecoderStepSplit(splitLoaded, tokenId, feeds);
            const decoderStepTotal = nowMs() - decoderStepStart;
            decoderStepMs += decoderStepTotal;
            decoderStepTensorCloneMs += step.timings.inputMs;
            decoderStepRunMs += step.timings.runMs;
            decoderStepOutputMs += step.timings.outputMs;
            decoderStepCount += 1;
            decoderStepTimings.push(decoderStepTotal);
            recordDecoderTiming(step.timings);
            // Update stored dims from step output
            for (const [k, v] of Object.entries(step.presentKv)) {
              kvDims[k] = v.dims;
            }
            return {
              logits: step.logits,
              vocabSize: step.vocabSize,
              presentKv: Object.fromEntries(
                Object.entries(step.presentKv).map(([k, v]) => [k, v.data]),
              ),
            };
          },
        });
    const decodeMs = nowMs() - decoderStart;
    const decoderStepAvgMs = decoderStepCount > 0 ? decoderStepMs / decoderStepCount : undefined;
    const decoderStepP50Ms = percentile(decoderStepTimings, 50);
    const decoderStepP95Ms = percentile(decoderStepTimings, 95);
    const decoderStepMaxMs = decoderStepTimings.length > 0 ? Math.max(...decoderStepTimings) : undefined;
    const decoderKvCacheLocation =
      useExperimentalGpuKvCache && decoderGpuTensorOutputs > 0
        ? 'gpu-buffer'
        : requestedDecoderKvCacheLocation;
    const decodeElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'decode',
      progress: clampProgress(0.85),
      elapsedMs: roundMetric(decodeElapsedMs),
      remainingMs: estimateRemainingMs(decodeElapsedMs, 0.85),
      completedUnits: result.tokens.length,
      totalUnits: maxNewTokens,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Decoded ${result.tokens.length} Whisper tokens for ${this.modelId}.`,
      metrics: {
        preprocessMs: roundMetric(preprocessMs),
        encodeMs: roundMetric(encodeMs),
        decodeMs: roundMetric(decodeMs),
        decoderInitMs: roundMetric(decoderInitMs),
        decoderInitInputMs: roundMetric(decoderInitInputMs),
        decoderInitRunMs: roundMetric(decoderInitRunMs),
        decoderInitOutputMs: roundMetric(decoderInitOutputMs),
        decoderStepMs: roundMetric(decoderStepMs),
        decoderStepFeedBuildMs: roundMetric(decoderStepFeedBuildMs),
        decoderStepTensorCloneMs: roundMetric(decoderStepTensorCloneMs),
        decoderStepRunMs: roundMetric(decoderStepRunMs),
        decoderStepOutputMs: roundMetric(decoderStepOutputMs),
        decoderStepAvgMs: decoderStepAvgMs !== undefined ? roundMetric(decoderStepAvgMs) : undefined,
        decoderStepP50Ms: decoderStepP50Ms !== undefined ? roundMetric(decoderStepP50Ms) : undefined,
        decoderStepP95Ms: decoderStepP95Ms !== undefined ? roundMetric(decoderStepP95Ms) : undefined,
        decoderStepMaxMs: decoderStepMaxMs !== undefined ? roundMetric(decoderStepMaxMs) : undefined,
        decoderLogitProcessMs: roundMetric(decoderLogitProcessMs),
        decoderStepCount,
        decoderGpuTensorInputs,
        decoderCpuTensorInputs,
        decoderGpuTensorOutputs,
        decoderCpuTensorOutputs,
        decoderGpuTensorDownloads,
        decoderKvCacheLocation,
        encoderFrameCount,
        decodeIterations: result.tokens.length,
      },
    });

    // 5. Build token details
    const tokenizeStart = nowMs();
    const generatedTokens = [...promptTokens, ...result.tokens];
    const tokenDetails: WhisperNativeToken[] = [];
    for (let i = promptTokens.length; i < generatedTokens.length; i++) {
      const tokenId = generatedTokens[i]!;
      tokenDetails.push({
        index: i - promptTokens.length,
        id: tokenId,
        text: this.formatTokenText(tokenizer, tokenId),
        special: tokenizer.isSpecialTokenId(tokenId),
      });
    }

    // 6. Build segments
    const segments = this.buildSegments(tokenDetails, tokenizer, options.noTimestamps);

    // 7. Word timestamps via splitgraph alignment
    const words = this.shouldReturnWordTimestamps(options)
      ? loaded.decoderAlignSession
        ? await this.computeAttentionWordTimestampsSplitGraph(
            loaded as Required<LoadedExecutorState>,
            encoderHiddenStates, tokenizer, segments,
            generatedTokens, promptTokens.length, options,
          )
        : []
      : [];

    const utteranceText = segments.map((s) => s.text).join(' ').trim();
    const tokenizeMs = nowMs() - tokenizeStart;
    const totalMs = roundMetric(nowMs() - transcriptionStart);
    const rtf = audio.durationSeconds > 0 ? totalMs / (audio.durationSeconds * 1000) : 0;
    const rtfx = audio.durationSeconds > 0 ? audio.durationSeconds / (totalMs / 1000) : undefined;
    const metrics: TranscriptMetrics = {
      preprocessMs: roundMetric(preprocessMs),
      encodeMs: roundMetric(encodeMs),
      decodeMs: roundMetric(decodeMs),
      tokenizeMs: roundMetric(tokenizeMs),
      postprocessMs: roundMetric(tokenizeMs),
      languageDetectionMs: roundMetric(languageDetectionMs),
      decoderInitMs: roundMetric(decoderInitMs),
      decoderInitInputMs: roundMetric(decoderInitInputMs),
      decoderInitRunMs: roundMetric(decoderInitRunMs),
      decoderInitOutputMs: roundMetric(decoderInitOutputMs),
      decoderStepMs: roundMetric(decoderStepMs),
      decoderStepFeedBuildMs: roundMetric(decoderStepFeedBuildMs),
      decoderStepTensorCloneMs: roundMetric(decoderStepTensorCloneMs),
      decoderStepRunMs: roundMetric(decoderStepRunMs),
      decoderStepOutputMs: roundMetric(decoderStepOutputMs),
      decoderStepAvgMs: decoderStepAvgMs !== undefined ? roundMetric(decoderStepAvgMs) : undefined,
      decoderStepP50Ms: decoderStepP50Ms !== undefined ? roundMetric(decoderStepP50Ms) : undefined,
      decoderStepP95Ms: decoderStepP95Ms !== undefined ? roundMetric(decoderStepP95Ms) : undefined,
      decoderStepMaxMs: decoderStepMaxMs !== undefined ? roundMetric(decoderStepMaxMs) : undefined,
      decoderLogitProcessMs: roundMetric(decoderLogitProcessMs),
      decoderStepCount,
      decoderGpuTensorInputs,
      decoderCpuTensorInputs,
      decoderGpuTensorOutputs,
      decoderCpuTensorOutputs,
      decoderGpuTensorDownloads,
      decoderKvCacheLocation,
      // Profiling sub-buckets (init)
      decoderInitTensorCreateMs: roundMetric(decoderInitTensorCreateMs),
      decoderInitLogitReadMs: roundMetric(decoderInitLogitReadMs),
      decoderInitKvExtractMs: roundMetric(decoderInitKvExtractMs),
      // Profiling sub-buckets (step, totals across all steps)
      decoderStepTensorCreateMs: roundMetric(decoderStepTensorCreateMs),
      decoderStepLogitReadMs: roundMetric(decoderStepLogitReadMs),
      decoderStepKvMergeMs: roundMetric(decoderStepKvMergeMs),
      sessionCreateMs: roundMetric(loaded.sessionCreateMs ?? 0),
      // DIAGNOSTIC: encoder sub-timing (Track A)
      encoderRunMs: roundMetric(encoderRunMs),
      encoderOutputMs: roundMetric(encoderOutputMs),
      encoderOutputLocation,
      encoderOutputDtype,
      totalMs,
      wallMs: totalMs,
      audioDurationSec: roundMetric(audio.durationSeconds, 4),
      rtf: roundMetric(rtf, 4),
      rtfx: rtfx !== undefined ? roundMetric(rtfx, 4) : undefined,
      preprocessorBackend: 'js-whisper-mel',
      requestedPreprocessorBackend: 'js-whisper-mel',
      encoderFrameCount,
      decodeIterations: result.tokens.length,
      emittedTokenCount: tokenDetails.filter((token) => !token.special).length,
      emittedWordCount: words?.length,
    };
    const postprocessElapsedMs = nowMs() - transcriptionStart;
    emitTranscriptionProgress(options, {
      stage: 'postprocess',
      progress: 0.95,
      elapsedMs: roundMetric(postprocessElapsedMs),
      remainingMs: estimateRemainingMs(postprocessElapsedMs, 0.95),
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Built Whisper transcript details for ${this.modelId}.`,
      metrics,
    });
    emitTranscriptionProgress(options, {
      stage: 'complete',
      progress: 1,
      elapsedMs: metrics.totalMs,
      modelId: this.modelId,
      backendId: this.backendId,
      message: `Finished transcription for ${this.modelId}.`,
      metrics,
    });

    return {
      utteranceText, isFinal: true, language, segments,
      ...(words && words.length > 0 ? { words } : {}),
      tokens: options.returnSpecialTokens
        ? tokenDetails
        : tokenDetails.filter((t) => !t.special),
      metrics,
      warnings,
    };
  }

  private async transcribeLongAudio(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    context: { readonly modelId: string; readonly config: WhisperSeq2SeqModelConfig },
  ): Promise<WhisperNativeTranscript> {
    const pcmData = audio.channels?.[0] ?? new Float32Array(0);
    const chunkLengthSeconds = options.chunkLengthSeconds ?? 30;
    const chunks = planWhisperChunks(
      pcmData.length,
      audio.sampleRate,
      chunkLengthSeconds,
      options.strideLengthSeconds,
    );

    const chunkTranscripts = [];
    for (const chunk of chunks) {
      const samples = pcmData.slice(chunk.startSample, chunk.endSample);
      const chunkAudio: AudioBufferLike = {
        sampleRate: audio.sampleRate,
        durationSeconds: samples.length / audio.sampleRate,
        channels: [samples],
        numberOfChannels: 1,
        numberOfFrames: samples.length,
      };
      const transcript = await this.transcribe(
        chunkAudio,
        { ...options, unsafeAllowOverMaxWindow: true },
        context,
      );
      chunkTranscripts.push({ chunkStartTime: chunk.startTime, transcript });
    }

    return mergeWhisperChunkTranscripts(chunkTranscripts);
  }

  private formatTokenText(tokenizer: WhisperTokenizer, tokenId: number): string {
    if (tokenizer.isTimestampTokenId(tokenId) || tokenizer.isSpecialTokenId(tokenId)) {
      return tokenizer.idsToTokens([tokenId])[0] ?? '';
    }
    return tokenizer.decode([tokenId]);
  }

  async dispose(): Promise<void> {
    await Promise.all(
      this.assetHandles.map(async (handle) => {
        await handle.dispose();
      }),
    );
    this.assetHandles.length = 0;
  }
}
