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
import { argmax, tokenQualityFromLogits } from '../../inference/index.js';

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
import {
  alignmentWindowForWhisperWords,
  buildWhisperWordTimestampsFromDtwTokens,
  buildWhisperWordTimestampsFromTokenDetails,
  clipShortWhisperWordDurations,
  coalesceWhisperWordTimestamps,
  forcedAlignmentLooksAnchored,
  refineWhisperWordsWithForcedAlignment,
  splitWhisperWordsByPause,
} from './word-timestamps.js';
import { computeWhisperDtwTokenTimestamps } from './attention-alignment.js';
import {
  whisperDecode,
  type WhisperCoreSession,
  type WhisperKvCache,
  type WhisperKvCacheValue,
} from './core.js';
import { selectWhisperLanguageFromLogits } from './language-detection.js';
import {
  parseWhisperGenerationConfig,
  parseWhisperModelConfig,
  type WhisperGenerationConfig,
  type WhisperModelConfig,
} from './generation-config.js';
import { parseWhisperManifest } from './manifest.js';
import type {
  WhisperArtifactSource,
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperNativeWord,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

// GPUBuffer is a browser WebGPU global, not available in Node/TS compilation.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type GPUBuffer = any;

export function assertExperimentalGpuKvCacheIsGreedyOnly(input: {
  readonly enabled: boolean;
  readonly numBeams: number;
  readonly bestOf: number;
  readonly temperature: number;
}): void {
  if (
    input.enabled &&
    (input.numBeams > 1 || input.bestOf > 1 || input.temperature > 0)
  ) {
    throw new Error(
      'experimentalGpuKvCache currently supports only greedy argmax decoding; disable it for beam search, best_of, or temperature sampling.',
    );
  }
}

type WhisperOrtSessionOptions = Parameters<typeof createWhisperOrtSession>[2];

/**
 * Create an ORT session with an opt-in graph-capture request.
 *
 * Graph capture is deliberately diagnostic: ORT requires every graph node to
 * be partitioned to the requested execution provider, which is not true for
 * the current Whisper exports on all WebGPU bundles. Retrying without capture
 * keeps that experiment from turning into a model-load failure while allowing
 * the caller to surface the fallback as a recoverable warning.
 */
export async function createWhisperOrtSessionWithGraphCaptureFallback(
  ort: OrtModuleLike,
  url: string,
  options: WhisperOrtSessionOptions,
  onFallback?: (error: unknown) => void,
): Promise<OrtSessionLike> {
  try {
    return await createWhisperOrtSession(ort, url, options);
  } catch (error) {
    if (
      !options.enableGraphCapture ||
      !options.backendId.startsWith('webgpu') ||
      !isWhisperGraphCaptureUnavailableError(error)
    ) {
      throw error;
    }
    onFallback?.(error);
    return createWhisperOrtSession(ort, url, {
      ...options,
      enableGraphCapture: false,
    });
  }
}

function isWhisperGraphCaptureUnavailableError(error: unknown): boolean {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  return message.includes('graph capture') || message.includes('enablegraphcapture');
}

function formatWhisperSessionError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.length > 240 ? `${message.slice(0, 237)}...` : message;
}

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
  readonly decoderAlignUrl?: string;
  /** True only when manifest metadata proves decoder_align is causal. */
  readonly decoderAlignCausalSelfAttention?: boolean;
  readonly enableProfiling?: boolean;
  readonly decoderAlignExternalData?: { readonly dataUrl: string; readonly path: string };
  readonly decoderBackendForOrt?: string;
  readonly experimentalGpuKvCache?: boolean;
  readonly sessionCreateMs?: number;
  /** DIAGNOSTIC (Edge A): Re-wrap encoder GPU buffer as fresh tensor. */
  readonly encoderBufferRewrap?: boolean;
  /** DIAGNOSTIC (Edge B2): Force GPU flush before decoder_init. */
  readonly encoderGpuFlush?: boolean;
  /** PROFILING (encoderGpuDrain): Force GPU drain + re-wrap after encoder. */
  readonly encoderGpuDrain?: boolean;
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
  readonly tokenTraces?: readonly { readonly tokenId: number; readonly logProb: number; readonly entropy: number }[];
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

export function resolveWhisperLanguageCode(
  requestedLanguage: string | undefined,
  configuredLanguages: readonly string[] | undefined,
): string {
  if (requestedLanguage && requestedLanguage !== 'auto') return requestedLanguage;
  return configuredLanguages?.find((language) => language !== 'auto') ?? 'en';
}

export function resolveWhisperLanguageTokenId(
  tokenizer: Pick<WhisperTokenizer, 'getTokenId'>,
  language: string,
): number {
  return tokenizer.getTokenId(`<|${language}|>`)
    ?? tokenizer.getTokenId('<|en|>')
    ?? 50259;
}

/**
 * Build the teacher-forced sequence used by Whisper's reference word
 * alignment. This matches faster-whisper's `tokenizer.sot_sequence` contract:
 * SOT + language + task, followed by text and EOT. Generated timestamp and
 * no-timestamps control tokens are not part of the cross-attention prompt.
 */
export function buildWhisperForcedAlignmentTokenIds(
  tokenizer: Pick<WhisperTokenizer, 'getTokenId'>,
  language: string,
  textTokenIds: readonly number[],
  task: 'transcribe' | 'translate' = 'transcribe',
): number[] {
  const taskToken = task === 'translate' ? '<|translate|>' : '<|transcribe|>';
  const fallbackTaskId = task === 'translate' ? 50359 : 50360;
  return [
    tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
    resolveWhisperLanguageTokenId(tokenizer, language),
    tokenizer.getTokenId(taskToken) ?? fallbackTaskId,
    ...textTokenIds,
    tokenizer.getTokenId('<|endoftext|>') ?? 50257,
  ];
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

function float16BitsToFloat32(src: Uint16Array): Float32Array {
  const dst = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const bits = src[i]!;
    const sign = (bits >>> 15) & 0x1;
    const exp = (bits >>> 10) & 0x1f;
    const mant = bits & 0x3ff;
    let value: number;
    if (exp === 0) {
      value = mant === 0 ? 0 : (mant / 1024) * 2 ** -14;
    } else if (exp === 0x1f) {
      value = mant === 0 ? Infinity : Number.NaN;
    } else {
      value = (1 + mant / 1024) * 2 ** (exp - 15);
    }
    dst[i] = sign ? -value : value;
  }
  return dst;
}

function tensorDataAsFloat32(data: ArrayBufferView): Float32Array {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) return float16BitsToFloat32(data);
  return Float32Array.from(data as unknown as ArrayLike<number>);
}

/**
 * Match the encoder hidden-state tensor to the graph input contract.
 *
 * Split-graph exports are allowed to use different precision at the graph
 * boundary.  The optimized decoder_init usually accepts fp32 (and can keep a
 * GPU tensor), while decoder_align may be exported from the original fp32
 * Transformers module even when the runtime encoder emits fp16.  ORT-WebGPU
 * does not insert this boundary cast for us, so an otherwise valid alignment
 * graph can fail and silently fall back to generated timestamp tokens.
 */
async function maybeCastEncoderHiddenStates(
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  decoderSession: OrtSessionLike,
  ort: OrtModuleLike,
): Promise<OrtTensorLike<Float32Array>> {
  const metadata = (decoderSession as unknown as { inputMetadata?: Array<{ name?: string; type?: string; shape?: number[] }> }).inputMetadata;
  const encMeta = metadata?.find((m) => m.name === 'encoder_hidden_states');
  const expectedType = encMeta?.type;
  if (expectedType !== 'float16' && expectedType !== 'float32') {
    return encoderHiddenStates;
  }
  if (encoderHiddenStates.type === expectedType) {
    return encoderHiddenStates;
  }
  // A precision cast necessarily materializes the tensor on CPU. This is
  // limited to graph boundaries whose declared element types differ.
  const rawData = isGpuBufferTensor(encoderHiddenStates) && encoderHiddenStates.getData
    ? await encoderHiddenStates.getData(true)
    : encoderHiddenStates.data;
  const dims = encoderHiddenStates.dims as number[];
  const size = dims.reduce((a, b) => a * b, 1);
  const f32Data = tensorDataAsFloat32(rawData as ArrayBufferView);
  const boundedF32Data = f32Data.length === size ? f32Data : f32Data.subarray(0, size);
  if (expectedType === 'float16') {
    return new ort.Tensor('float16', float32ToFloat16Bits(boundedF32Data), dims) as unknown as OrtTensorLike<Float32Array>;
  }
  return new ort.Tensor('float32', boundedF32Data, dims) as unknown as OrtTensorLike<Float32Array>;
}

/**
 * Match Whisper mel features to the encoder input contract.
 *
 * Export-time fp16 graphs may require float16 mel input, while the historical
 * fp16-IO graph accepts float32. ORT-WebGPU does not insert this boundary cast
 * for us, so detect the declared input type instead of making the preprocessor
 * choose one dtype for every artifact.
 */
export async function maybeCastWhisperFeatureTensor(
  featureTensor: OrtTensorLike<Float32Array>,
  encoderSession: OrtSessionLike,
  ort: OrtModuleLike,
): Promise<OrtTensorLike<Float32Array>> {
  const metadata = (encoderSession as unknown as {
    inputMetadata?: Array<{ name?: string; type?: string; shape?: number[] }>;
  }).inputMetadata;
  const featureMeta = metadata?.find((entry) => entry.name === 'input_features');
  const expectedType = featureMeta?.type;
  if (expectedType !== 'float16' && expectedType !== 'float32') {
    return featureTensor;
  }
  if (featureTensor.type === expectedType) {
    return featureTensor;
  }

  const rawData = isGpuBufferTensor(featureTensor) && featureTensor.getData
    ? await featureTensor.getData(true)
    : featureTensor.data;
  const dims = featureTensor.dims as number[];
  const size = dims.reduce((a, b) => a * b, 1);
  const f32Data = tensorDataAsFloat32(rawData as ArrayBufferView);
  const boundedF32Data = f32Data.length === size ? f32Data : f32Data.subarray(0, size);
  if (expectedType === 'float16') {
    return new ort.Tensor('float16', float32ToFloat16Bits(boundedF32Data), dims) as unknown as OrtTensorLike<Float32Array>;
  }
  return new ort.Tensor('float32', boundedF32Data, dims) as unknown as OrtTensorLike<Float32Array>;
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
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: WhisperKvCache }>;
  runStep(
    tokenId: number,
    pastKv: WhisperKvCache,
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: WhisperKvCache }>;
  runStepBatch?(
    tokenIds: readonly number[],
    pastKvs: readonly WhisperKvCache[],
  ): Promise<readonly { logits: Float32Array; vocabSize: number; presentKv: WhisperKvCache }[] | undefined>;
}

export interface SplitGraphDecodeResult {
  readonly tokens: readonly number[];
  readonly tokenTraces?: readonly { readonly tokenId: number; readonly logProb: number; readonly entropy: number }[];
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

type TensorDataView = ArrayBufferView & { readonly length: number };
type MutableTensorDataView = TensorDataView & {
  [index: number]: number;
  set?: (source: ArrayLike<number>, offset?: number) => void;
};
type TensorDataConstructor = {
  new(length: number): TensorDataView;
  new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): TensorDataView;
  readonly name?: string;
};

function cloneTensorDataView(data: TensorDataView, ctor: TensorDataConstructor): TensorDataView {
  const buffer = (data.buffer as ArrayBuffer).slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new ctor(buffer, 0, data.length);
}

function sliceTensorDataView(data: TensorDataView, elementOffset: number, length: number): TensorDataView {
  const bytesPerElement = data.byteLength / Math.max(1, data.length);
  const startByte = data.byteOffset + elementOffset * bytesPerElement;
  const endByte = startByte + length * bytesPerElement;
  const buffer = (data.buffer as ArrayBuffer).slice(startByte, endByte);
  return new (data.constructor as TensorDataConstructor)(buffer, 0, length);
}

export function cloneDecoderKvDataForInput(
  data: ArrayBufferView,
  tensorType?: string,
): { readonly type: 'float32' | 'float16'; readonly data: TensorDataView } {
  const view = data as TensorDataView;
  const dataCtor = view.constructor as TensorDataConstructor;
  const isFloat16 = tensorType === 'float16' || dataCtor.name === 'Float16Array';
  if (isFloat16) {
    const globalFloat16Ctor = (globalThis as unknown as { readonly Float16Array?: TensorDataConstructor }).Float16Array;
    return {
      type: 'float16',
      data: cloneTensorDataView(view, globalFloat16Ctor ?? dataCtor),
    };
  }

  return {
    type: 'float32',
    data: cloneTensorDataView(view, dataCtor),
  };
}

function concatTensorDataViews(parts: readonly TensorDataView[], ctor: TensorDataConstructor): TensorDataView {
  const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
  const merged = new ctor(totalLength) as MutableTensorDataView;
  let offset = 0;
  for (const part of parts) {
    if (merged.set) {
      merged.set(part as unknown as ArrayLike<number>, offset);
    } else {
      const source = part as unknown as ArrayLike<number>;
      for (let i = 0; i < part.length; i++) {
        merged[offset + i] = source[i] ?? 0;
      }
    }
    offset += part.length;
  }
  return merged;
}

export function concatDecoderKvDataForBatch(
  values: readonly { readonly data: ArrayBufferView; readonly type?: string }[],
  fallbackType: 'float32' | 'float16',
): { readonly type: 'float32' | 'float16'; readonly data: TensorDataView } {
  const cloned = values.map((value) => cloneDecoderKvDataForInput(value.data, value.type ?? fallbackType));
  const first = cloned[0];
  if (!first) {
    throw new Error('Cannot batch empty Whisper KV tensor data.');
  }
  for (const value of cloned) {
    if (value.type !== first.type) {
      throw new Error(`Cannot batch mixed Whisper KV tensor dtypes: ${first.type} and ${value.type}.`);
    }
  }
  return {
    type: first.type,
    data: concatTensorDataViews(
      cloned.map((value) => value.data),
      first.data.constructor as TensorDataConstructor,
    ),
  };
}

function cloneDecoderKvTensorDataForInput(
  tensor: OrtTensorLike<Float32Array>,
): { readonly type: 'float32' | 'float16'; readonly data: TensorDataView } {
  const data = tensor.data as unknown as TensorDataView;
  const dataCtor = data.constructor as TensorDataConstructor;
  return cloneDecoderKvDataForInput(data, tensor.type ?? dataCtor.name);
}

function normalizeWhisperKvCacheValue(
  value: WhisperKvCacheValue,
): { readonly data: ArrayBufferView; readonly dims?: readonly number[]; readonly type?: string } {
  if (ArrayBuffer.isView(value)) {
    return { data: value };
  }
  return value;
}

export async function splitGraphDecodeLoop(params: {
  promptTokens: readonly number[];
  encoderHiddenStates: Float32Array;
  eosTokenId: number;
  maxNewTokens: number;
  modelConfig: WhisperModelConfig;
  runInit: SplitGraphDecodeCallbacks['runInit'];
  runStep: SplitGraphDecodeCallbacks['runStep'];
  runStepBatch?: SplitGraphDecodeCallbacks['runStepBatch'];
  processLogits?: (logits: Float32Array, generatedTokens: readonly number[], beginIndex: number) => void;
  onTokenLogits?: (chosenTokenId: number, processedLogits: Float32Array, ctx: { readonly tokens: readonly number[]; readonly beginIndex: number }) => void;
  onDecoderInitLogits?: (
    rawLogits: Float32Array,
    ctx: {
      readonly tokens: readonly number[];
      readonly beginIndex: number;
      readonly vocabSize: number;
      readonly noSpeechTokenId?: number;
    },
  ) => void;
  noSpeechTokenId?: number;
  /** Beam search: number of beams (default: 1 = greedy) */
  numBeams?: number;
  /** Final ranking penalty. Undefined uses length normalization; 0 uses raw score. */
  lengthPenalty?: number;
  /** Beam search patience for early stopping. */
  patience?: number;
  /** Greedy decoding temperature. 0 = argmax. */
  temperature?: number;
  /** Number of independent decodings, pick best by score (WhisperX: best_of) */
  bestOf?: number;
  /** Experimental: batch active beam decoder steps into one ORT call when supported. */
  experimentalBatchedBeam?: boolean;
  /** Collect selected-sequence scalar quality traces. */
  trackQuality?: boolean;
}): Promise<SplitGraphDecodeResult> {
  const {
    promptTokens,
    encoderHiddenStates,
    eosTokenId,
    maxNewTokens,
    modelConfig,
    runInit,
    runStep,
    runStepBatch,
    processLogits,
    onTokenLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    numBeams,
    lengthPenalty,
    patience,
    temperature,
    bestOf,
    experimentalBatchedBeam,
    trackQuality,
  } = params;

  const encoderDims: readonly number[] = [1, encoderHiddenStates.length / modelConfig.dModel, modelConfig.dModel];
  const session: WhisperCoreSession = {
    runInit: async (pt, enc, dims) => runInit(pt, enc, dims),
    runStep: async (tid, kv) => runStep(tid, kv),
    ...(runStepBatch ? { runStepBatch: async (tids, kvs) => runStepBatch(tids, kvs) } : {}),
  };
  const result = await whisperDecode(session, {
    promptTokens,
    encoderOutput: encoderHiddenStates,
    encoderDims,
    eosTokenId,
    maxNewTokens,
    processLogits,
    onTokenLogits,
    onDecoderInitLogits,
    noSpeechTokenId,
    strategy: (numBeams ?? 1) > 1 ? 'beam' : 'greedy',
    beamSize: numBeams ?? 1,
    lengthPenalty,
    patience: patience ?? 1,
    temperature: temperature ?? 0,
    bestOf: bestOf ?? 1,
    experimentalBatchedBeam: experimentalBatchedBeam ?? false,
    trackQuality: trackQuality === true,
  });
  return {
    tokens: result.tokens,
    ...(result.tokenTraces ? { tokenTraces: result.tokenTraces } : {}),
  };
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
  /** Explicit alignment-matrix rows for text tokens. Use this when timestamp
   *  or other special tokens sit between prompt and text tokens. */
  readonly textTokenRowIndices?: readonly number[];
  /** Crop DTW to this many encoder frames (e.g. actual audio duration / 0.02).
   *  The alignment matrix may still be padded to the 30s Whisper window. */
  readonly cropFrameCount?: number;
}

export function collectSplitGraphTextTokenRows(
  tokenIds: readonly number[],
  promptLen: number,
  isTextToken: (tokenId: number) => boolean,
): { readonly tokenIds: number[]; readonly rowIndices: number[] } {
  const textTokenIds: number[] = [];
  const rowIndices: number[] = [];
  for (let index = promptLen; index < tokenIds.length; index++) {
    const tokenId = tokenIds[index]!;
    if (!isTextToken(tokenId)) continue;
    textTokenIds.push(tokenId);
    rowIndices.push(index);
  }
  return { tokenIds: textTokenIds, rowIndices };
}

export function extractSplitGraphAlignmentRows(
  alignmentData: Float32Array,
  rowIndices: readonly number[],
  frameCount: number,
): Float32Array {
  const textValues = new Float32Array(rowIndices.length * frameCount);
  for (let i = 0; i < rowIndices.length; i++) {
    const srcOffset = rowIndices[i]! * frameCount;
    textValues.set(alignmentData.subarray(srcOffset, srcOffset + frameCount), i * frameCount);
  }
  return textValues;
}

export function processSplitGraphAlignment(
  options: SplitGraphAlignmentOptions,
): readonly number[] {
  const {
    alignmentData, promptLen, textTokenCount, frameCount,
    medianFilterWidth, timePrecisionSeconds, textTokenRowIndices, cropFrameCount,
  } = options;

  if (textTokenCount === 0 && (textTokenRowIndices?.length ?? 0) === 0) return [0];
  if (frameCount === 0) {
    const count = textTokenRowIndices?.length ?? textTokenCount;
    return Array.from({ length: count + 1 }, () => 0);
  }

  const textValues = textTokenRowIndices && textTokenRowIndices.length > 0
    ? extractSplitGraphAlignmentRows(alignmentData, textTokenRowIndices, frameCount)
    : (() => {
        const extracted = new Float32Array(textTokenCount * frameCount);
        const srcOffset = promptLen * frameCount;
        extracted.set(alignmentData.subarray(srcOffset, srcOffset + textTokenCount * frameCount));
        return extracted;
      })();
  const resolvedTextTokenCount = textTokenRowIndices?.length ?? textTokenCount;
  const cropFrames = Math.max(1, Math.min(frameCount, cropFrameCount ?? frameCount));

  const headMatrix = {
    values: textValues,
    tokenCount: resolvedTextTokenCount,
    frameCount,
  };

  return computeWhisperDtwTokenTimestamps({
    attentionHeads: [headMatrix],
    tokenCount: resolvedTextTokenCount,
    frameCount: cropFrames,
    medianFilterWidth,
    timePrecisionSeconds,
  });
}

export function extractSplitGraphAlignmentWindow(
  alignmentData: Float32Array,
  rowIndices: readonly number[],
  fullFrameCount: number,
  startFrame: number,
  windowFrames: number,
): Float32Array {
  const output = new Float32Array(rowIndices.length * windowFrames);
  for (let i = 0; i < rowIndices.length; i++) {
    const srcOffset = rowIndices[i]! * fullFrameCount + startFrame;
    output.set(alignmentData.subarray(srcOffset, srcOffset + windowFrames), i * windowFrames);
  }
  return output;
}

export interface SplitGraphTimestampSpanAlignmentOptions {
  readonly alignmentData: Float32Array;
  readonly tokenIds: readonly number[];
  readonly promptLen: number;
  readonly frameCount: number;
  readonly medianFilterWidth?: number;
  readonly timePrecisionSeconds?: number;
  readonly cropFrameCount?: number;
  readonly isTextToken: (tokenId: number) => boolean;
  readonly isTimestampToken: (tokenId: number) => boolean;
  readonly timestampTokenToSeconds: (tokenId: number) => number;
}

/**
 * Whisper-style word alignment: DTW each timestamp-token segment against only
 * the encoder frames in that segment, then concatenate jump times.
 * Returns undefined when the sequence has no timestamp tokens.
 */
export function processSplitGraphAlignmentByTimestampSpans(
  options: SplitGraphTimestampSpanAlignmentOptions,
): readonly number[] | undefined {
  const hop = options.timePrecisionSeconds ?? 0.02;
  const cropFrames = Math.max(1, Math.min(options.frameCount, options.cropFrameCount ?? options.frameCount));
  const audioEnd = cropFrames * hop;

  const spans: { start: number; end: number; rowIndices: number[] }[] = [];
  let spanStart: number | null = null;
  let spanRows: number[] = [];
  for (let index = options.promptLen; index < options.tokenIds.length; index++) {
    const tokenId = options.tokenIds[index]!;
    if (options.isTimestampToken(tokenId)) {
      const time = options.timestampTokenToSeconds(tokenId);
      if (spanStart !== null && spanRows.length > 0) {
        spans.push({ start: spanStart, end: Math.max(time, spanStart + hop), rowIndices: spanRows });
      }
      spanStart = time;
      spanRows = [];
      continue;
    }
    if (spanStart !== null && options.isTextToken(tokenId)) {
      spanRows.push(index);
    }
  }
  if (spanStart !== null && spanRows.length > 0) {
    spans.push({ start: spanStart, end: Math.max(audioEnd, spanStart + hop), rowIndices: spanRows });
  }
  if (spans.length === 0) return undefined;

  const jumpTimes: number[] = [];
  for (const span of spans) {
    const startFrame = Math.max(0, Math.min(cropFrames - 1, Math.round(span.start / hop)));
    const endFrame = Math.max(startFrame + 1, Math.min(cropFrames, Math.round(span.end / hop)));
    const windowFrames = endFrame - startFrame;
    const window = extractSplitGraphAlignmentWindow(
      options.alignmentData,
      span.rowIndices,
      options.frameCount,
      startFrame,
      windowFrames,
    );
    const relative = computeWhisperDtwTokenTimestamps({
      attentionHeads: [{
        values: window,
        tokenCount: span.rowIndices.length,
        frameCount: windowFrames,
      }],
      tokenCount: span.rowIndices.length,
      frameCount: windowFrames,
      medianFilterWidth: options.medianFilterWidth,
      timePrecisionSeconds: hop,
    });
    const spanEnd = Math.min(span.end, audioEnd);
    for (let i = 0; i < span.rowIndices.length; i++) {
      const time = Math.min(spanEnd, Math.max(span.start, span.start + (relative[i] ?? 0)));
      jumpTimes.push(jumpTimes.length === 0 ? time : Math.max(jumpTimes[jumpTimes.length - 1]!, time));
    }
  }

  const lastSpan = spans[spans.length - 1]!;
  jumpTimes.push(Math.max(jumpTimes[jumpTimes.length - 1] ?? 0, Math.min(lastSpan.end, audioEnd)));
  return jumpTimes;
}

function sliceAudioBufferLike(
  audio: AudioBufferLike,
  startSeconds: number,
  endSeconds: number,
): AudioBufferLike {
  const startFrame = Math.max(0, Math.floor(startSeconds * audio.sampleRate));
  const endFrame = Math.min(
    audio.numberOfFrames,
    Math.max(startFrame + 1, Math.ceil(endSeconds * audio.sampleRate)),
  );
  const frameCount = Math.max(1, endFrame - startFrame);
  const channels = audio.channels?.map((channel) => channel.subarray(startFrame, startFrame + frameCount));
  const data = channels?.[0] ?? (audio.data instanceof Float32Array
    ? audio.data.subarray(startFrame, startFrame + frameCount)
    : undefined);
  return {
    sampleRate: audio.sampleRate,
    numberOfChannels: audio.numberOfChannels,
    numberOfFrames: frameCount,
    durationSeconds: frameCount / audio.sampleRate,
    ...(channels ? { channels } : {}),
    ...(data ? { data, format: audio.format ?? 'f32-planar' } : {}),
  };
}

export class WhisperOnnxExecutor {
  private readonly sourceOptions: WhisperArtifactSource | undefined;
  private readonly loadStatePromise?: Promise<LoadedExecutorState>;
  private readonly assetProvider?: AssetProvider;
  private readonly runtimeHooks?: SpeechRuntimeHooks;
  private readonly assetHandles: ResolvedAssetHandle[] = [];
  private decoderAlignSession?: OrtSessionLike;
  private decoderAlignLoadPromise?: Promise<OrtSessionLike | undefined>;

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
      const manifestUrl =
        (await resolveRemoteUrl(resolved.manifestUrl, 'manifest.json')) ??
        resolved.manifestUrl;

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
        manifestUrl,
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
    const warnings: TranscriptWarning[] = [...resolved.warnings];

    // Time session creation
    const sessionStart = nowMs();
    // Only create encoder session for now (decoder sessions created below if splitgraph)
    const encoderSessionOptions: WhisperOrtSessionOptions = {
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
    };
    const encoderSession = await createWhisperOrtSessionWithGraphCaptureFallback(
      ort,
      artifacts.encoderUrl,
      encoderSessionOptions,
      resolved.experimentalWebGpuEncoderGraphCapture && resolved.encoderBackendForOrt === 'webgpu'
        ? (error) => warnings.push({
            code: 'whisper.encoder-graph-capture-fallback',
            message: `WebGPU encoder graph capture was unavailable; using the regular encoder session (${formatWhisperSessionError(error)}).`,
            recoverable: true,
          })
        : undefined,
    );

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
    const decoderAlignCausalSelfAttention = await this.loadDecoderAlignCausalSelfAttention(
      resolved.manifestUrl,
    );
    if (resolved.decoderAlignUrl && decoderAlignCausalSelfAttention === false) {
      warnings.push({
        code: 'whisper.decoder-align-legacy',
        message:
          'decoder_align is missing the causal-self-attention export marker; using generated timestamp interpolation until the artifact is re-exported.',
        recoverable: true,
      });
    }

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
      const decoderStepSessionOptions: WhisperOrtSessionOptions = {
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
      };
      decoderStepSession = await createWhisperOrtSessionWithGraphCaptureFallback(
        ort,
        resolved.decoderStepUrl,
        decoderStepSessionOptions,
        resolved.decoderGraphCapture && resolved.decoderBackendForOrt === 'webgpu'
          ? (error) => warnings.push({
              code: 'whisper.decoder-step-graph-capture-fallback',
              message: `WebGPU decoder_step graph capture was unavailable; using the regular decoder_step session (${formatWhisperSessionError(error)}).`,
              recoverable: true,
            })
          : undefined,
      );
      // Defer decoder_align — only load when needed for alignment (saves VRAM)
    }

    return {
      ort, tokenizer, encoderSession, decoderSession,
      generationConfig: genConfig, modelConfig, warnings,
      isSplitGraph, decoderInitSession, decoderStepSession, decoderAlignSession,
      decoderAlignUrl: resolved.decoderAlignUrl,
      decoderAlignCausalSelfAttention,
      enableProfiling: resolved.enableProfiling,
      decoderAlignExternalData: resolved.externalData?.decoder_align?.[0]
        ? {
            dataUrl: resolved.externalData.decoder_align[0].dataUrl,
            path: resolved.externalData.decoder_align[0].path,
          }
        : undefined,
      decoderBackendForOrt: resolved.decoderBackendForOrt,
      experimentalGpuKvCache: resolved.experimentalGpuKvCache,
      encoderBufferRewrap: resolved.encoderBufferRewrap,
      encoderGpuFlush: resolved.encoderGpuFlush,
      encoderGpuDrain: resolved.encoderGpuDrain,
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

  private async ensureDecoderAlignSession(
    loaded: LoadedExecutorState,
  ): Promise<OrtSessionLike | undefined> {
    if (this.decoderAlignSession) return this.decoderAlignSession;
    if (loaded.decoderAlignSession) {
      this.decoderAlignSession = loaded.decoderAlignSession;
      return this.decoderAlignSession;
    }
    if (!loaded.decoderAlignUrl) return undefined;

    this.decoderAlignLoadPromise ??= (async () => {
      try {
        const session = await createWhisperOrtSession(loaded.ort, loaded.decoderAlignUrl!, {
          backendId: loaded.decoderBackendForOrt ?? this.backendId,
          enableProfiling: loaded.enableProfiling,
          ...(loaded.decoderAlignExternalData
            ? {
                externalDataUrl: loaded.decoderAlignExternalData.dataUrl,
                externalDataPath: loaded.decoderAlignExternalData.path,
              }
            : {}),
        });
        this.decoderAlignSession = session;
        return session;
      } catch (error) {
        this.decoderAlignLoadPromise = undefined;
        throw error;
      }
    })();
    return this.decoderAlignLoadPromise;
  }

  private async materializeCpuEncoderHiddenStates(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    alignSession: OrtSessionLike,
  ): Promise<OrtTensorLike<Float32Array>> {
    let tensor = encoderHiddenStates;
    if (isGpuBufferTensor(tensor)) {
      const { data } = await readOrtTensorData(tensor);
      const type = tensor.type === 'float16' ? 'float16' : 'float32';
      tensor = new loaded.ort.Tensor(
        type,
        data,
        [...tensor.dims],
      ) as OrtTensorLike<Float32Array>;
    }
    return maybeCastEncoderHiddenStates(tensor, alignSession, loaded.ort);
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
    task: 'transcribe' | 'translate' = 'transcribe',
  ): Promise<{
    readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
    readonly logitsForText: Float32Array;
  }> {
    const tokenizer = loaded.tokenizer;
    const resolvedLanguage = resolveWhisperLanguageCode(
      language,
      this.config.languages,
    );
    const forcedIds = buildWhisperForcedAlignmentTokenIds(
      tokenizer,
      resolvedLanguage,
      textTokenIds,
      task,
    );
    const alignmentPromptLen = forcedIds.length - textTokenIds.length - 1;
    const inputIds = new BigInt64Array(forcedIds.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, forcedIds.length]);
    const decoderSession = loaded.decoderSession!;
    const encoderForAlignment = await maybeCastEncoderHiddenStates(
      encoderHiddenStates,
      decoderSession,
      loaded.ort,
    );

    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderForAlignment,
    };

    const decoderInputNames = decoderSession.inputNames ?? [];
    const shouldFeed = (name: string): boolean =>
      decoderInputNames.length === 0 || decoderInputNames.includes(name);
    if (decoderInputNames.includes('use_cache_branch')) {
      feeds.use_cache_branch = new loaded.ort.Tensor('bool', new Uint8Array([1]), [1]);
    }

    // First step: provide empty past_key_values
    // Use config-driven layer/head counts
    const numLayers = loaded.modelConfig.decoderLayers;
    const numHeads = loaded.modelConfig.decoderAttentionHeads;
    const headDim = loaded.modelConfig.headDim;
    const encoderSeqLen = encoderForAlignment.dims[1] as number;
    for (let i = 0; i < numLayers; i++) {
      const decoderKey = `past_key_values.${i}.decoder.key`;
      const decoderValue = `past_key_values.${i}.decoder.value`;
      const encoderKey = `past_key_values.${i}.encoder.key`;
      const encoderValue = `past_key_values.${i}.encoder.value`;
      if (shouldFeed(decoderKey)) {
        feeds[decoderKey] = new loaded.ort.Tensor(
          'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
      }
      if (shouldFeed(decoderValue)) {
        feeds[decoderValue] = new loaded.ort.Tensor(
          'float32', new Float32Array(0), [1, numHeads, 0, headDim]);
      }
      const encoderCacheSize = 1 * numHeads * encoderSeqLen * headDim;
      if (shouldFeed(encoderKey)) {
        feeds[encoderKey] = new loaded.ort.Tensor(
          'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
      }
      if (shouldFeed(encoderValue)) {
        feeds[encoderValue] = new loaded.ort.Tensor(
          'float32', new Float32Array(encoderCacheSize), [1, numHeads, encoderSeqLen, headDim]);
      }
    }

    const outputs = await decoderSession.run(feeds);
    const crossAttentions = await Promise.all(
      extractCrossAttentions(outputs).map(async (tensor) => {
        const { data } = await readOrtTensorData(tensor, { releaseGpu: true });
        return {
          data: tensorDataAsFloat32(data as ArrayBufferView),
          dims: tensor.dims,
          type: 'float32',
        } satisfies OrtTensorLike<Float32Array>;
      }),
    );

    // Decoder logits at row i predict the token after input_ids[i]. The first
    // forced text token is therefore predicted at the final prompt row, not
    // at the row containing that text token.
    const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
    const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
    const totalVocab = (logitsTensor.dims[logitsTensor.dims.length - 1] as number) ?? 51865;
    const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
    const logits = tensorDataAsFloat32(logitsData.data as ArrayBufferView);
    const logitTimeSteps = Number(logitsTensor.dims[logitsTensor.dims.length - 2] ?? forcedIds.length);

    // forcedIds: [SOT, lang, task, ...text, EOS]
    const textLogitStart = Math.max(0, alignmentPromptLen - 1);
    const textCount = textTokenIds.length;
    const logitsForText = new Float32Array(textCount * totalVocab);
    const availableRows = Math.max(0, Math.min(textCount, logitTimeSteps - textLogitStart));
    const srcOffset = textLogitStart * totalVocab;
    logitsForText.set(logits.subarray(srcOffset, srcOffset + availableRows * totalVocab));

    return { crossAttentions, logitsForText };
  }

  private async runForcedAlignmentSplitGraph(
    loaded: Required<Pick<LoadedExecutorState, 'decoderAlignSession' | 'ort'>> & LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    allTokenIds: readonly number[],
  ): Promise<{ readonly data: Float32Array; readonly dims: readonly number[] }> {
    const inputIds = new BigInt64Array(allTokenIds.map((id) => BigInt(id)));
    const inputIdsTensor = new loaded.ort.Tensor('int64', inputIds, [1, allTokenIds.length]);
    const feeds: Record<string, unknown> = {
      input_ids: inputIdsTensor,
      encoder_hidden_states: encoderHiddenStates,
    };

    const outputs = await loaded.decoderAlignSession.run(feeds);
    const alignKey = Object.keys(outputs)[0]!;
    const alignTensor = outputs[alignKey] as OrtTensorLike<Float32Array>;
    const rawData = isGpuBufferTensor(alignTensor)
      ? (await readOrtTensorData(alignTensor, { releaseGpu: true })).data
      : alignTensor.data;
    return {
      data: tensorDataAsFloat32(rawData as ArrayBufferView),
      dims: alignTensor.dims,
    };
  }

  private async computeAttentionWordTimestampsSplitGraph(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: WhisperTokenizer,
    tokenDetails: WhisperNativeToken[],
    allTokens: readonly number[],
    promptLen: number,
    language: string,
    options: WhisperSeq2SeqTranscriptionOptions,
    audioDurationSeconds?: number,
    warnings?: TranscriptWarning[],
  ): Promise<WhisperNativeTranscript['words']> {
    const { tokenIds: textTokenIds, rowIndices } = collectSplitGraphTextTokenRows(
      allTokens,
      promptLen,
      (id) => !tokenizer.isSpecialTokenId(id) && !tokenizer.isTimestampTokenId(id),
    );
    if (textTokenIds.length === 0) return [];

    // Older published 4-graph artifacts used an unmasked teacher-forced
    // decoder_align graph. Running it in WebGPU is both unsafe (future-token
    // leakage) and numerically unreliable for the old fp16 export, so keep
    // generated timestamp semantics until the artifact is re-exported.
    if (loaded.decoderAlignCausalSelfAttention === false) {
      warnings?.push({
        code: 'whisper.decoder-align-legacy-fallback',
        message:
          'Legacy decoder_align cannot provide verified WebGPU word alignment; using generated timestamp interpolation until the artifact is re-exported.',
        recoverable: true,
      });
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }

    try {
      const decoderAlignSession = await this.ensureDecoderAlignSession(loaded);
      if (!decoderAlignSession) {
        warnings?.push({
          code: 'whisper.decoder-align-unavailable',
          message: 'decoder_align is unavailable; using generated timestamp interpolation.',
          recoverable: true,
        });
        return [];
      }
      const encoderForAlign = await this.materializeCpuEncoderHiddenStates(
        loaded,
        encoderHiddenStates,
        decoderAlignSession,
      );
      // Match Whisper/faster-whisper's find_alignment contract: run the
      // decoder-align graph once with a no-timestamps teacher-forced prompt.
      // Generated timestamp tokens remain useful for segment boundaries, but
      // using them here constrains a leading pause to the first timestamp
      // span and anchors the first word at zero.
      const alignmentTokenIds = buildWhisperForcedAlignmentTokenIds(
        tokenizer,
        language,
        textTokenIds,
        options.task ?? 'transcribe',
      );
      // The prefix is SOT + language + task, matching faster-whisper's
      // tokenizer.sot_sequence. Keep this derived from the actual sequence so
      // row extraction cannot drift if the prompt contract changes.
      const alignmentPromptLen = alignmentTokenIds.length - textTokenIds.length - 1;
      const alignmentTextRowIndices = textTokenIds.map(
        (_tokenId, index) => alignmentPromptLen + index,
      );
      const { data: alignmentData, dims } = await this.runForcedAlignmentSplitGraph(
        { ...loaded, decoderAlignSession, ort: loaded.ort },
        encoderForAlign,
        alignmentTokenIds,
      );
      const frameCount = dims.length > 0 ? Number(dims[dims.length - 1]) : 0;
      const cropFrameCount = audioDurationSeconds && audioDurationSeconds > 0
        ? Math.max(1, Math.round(audioDurationSeconds / 0.02))
        : undefined;
      const dtwTimestamps = processSplitGraphAlignment({
        alignmentData,
        totalTokens: alignmentTokenIds.length,
        promptLen: alignmentPromptLen,
        textTokenCount: textTokenIds.length,
        frameCount,
        medianFilterWidth: loaded.modelConfig.medianFilterWidth,
        timePrecisionSeconds: 0.02,
        textTokenRowIndices: alignmentTextRowIndices,
        cropFrameCount,
      });

      return buildWhisperWordTimestampsFromDtwTokens(
        textTokenIds.map((id, i) => ({
          id,
          text: tokenizer.decode([id]),
          sourceIndex: Math.max(0, (rowIndices[i] ?? promptLen) - promptLen),
        })),
        dtwTimestamps,
        { language },
      );
    } catch (error) {
      if (error instanceof Error && error.message) {
        warnings?.push({
          code: 'whisper.decoder-align-fallback',
          message: `Whisper decoder alignment failed; using generated timestamp interpolation (${error.message}).`,
          recoverable: true,
        });
      }
      return buildWhisperWordTimestampsFromTokenDetails(tokenDetails, {
        timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
        timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
        language,
      });
    }
  }

  private async computeAttentionWordTimestamps(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
    tokenizer: WhisperTokenizer,
    tokenDetails: WhisperNativeToken[],
    segments: WhisperNativeSegment[],
    language: string,
    options: WhisperSeq2SeqTranscriptionOptions,
    audioDurationSeconds?: number,
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
    const alignmentTokenIds = buildWhisperForcedAlignmentTokenIds(
      tokenizer,
      resolveWhisperLanguageCode(language, this.config.languages),
      textTokenIds,
      options.task ?? 'transcribe',
    );
    const alignmentPromptLen = alignmentTokenIds.length - textTokenIds.length - 1;

    try {
      const alignment = await this.runForcedAlignment(
        loaded,
        encoderHiddenStates,
        language,
        textTokenIds,
        options.task ?? 'transcribe',
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
        const croppedFrames = audioDurationSeconds && audioDurationSeconds > 0
          ? Math.max(1, Math.min(
              totalFramesPerHead,
              Math.round(audioDurationSeconds / 0.02),
            ))
          : totalFramesPerHead;
        if (totalTokens < alignmentPromptLen + textTokenIds.length) {
          throw new Error(
            `Cross-attention layer ${layer} has ${totalTokens} token rows; expected at least ${alignmentPromptLen + textTokenIds.length}.`,
          );
        }
        // Extract single head: tensor has shape [batch=1, heads, tokens, frames]
        const headSize = totalTokens * totalFramesPerHead;
        const headOffset = head * headSize;
        const headValues = new Float32Array(textTokenIds.length * totalFramesPerHead);
        for (let tokenIndex = 0; tokenIndex < textTokenIds.length; tokenIndex++) {
          const sourceOffset = headOffset + (alignmentPromptLen + tokenIndex) * totalFramesPerHead;
          headValues.set(
            layerTensor.data.subarray(sourceOffset, sourceOffset + totalFramesPerHead),
            tokenIndex * totalFramesPerHead,
          );
        }
        return {
          values: headValues,
          tokenCount: textTokenIds.length,
          frameCount: croppedFrames,
        };
      });

      const croppedFrames = attentionHeads[0]?.frameCount ?? 0;
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

  private async loadDecoderAlignCausalSelfAttention(
    manifestUrl: string | undefined,
  ): Promise<boolean | undefined> {
    if (!manifestUrl) return undefined;
    try {
      const raw = JSON.parse(await fetchText(manifestUrl)) as Record<string, unknown>;
      return parseWhisperManifest(raw).alignmentExport?.causalSelfAttention === true;
    } catch {
      // A custom source may omit the optional manifest or use an older schema.
      // Keep the existing alignment path when metadata cannot be inspected.
      return undefined;
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
    return this.runDecoderStepMultiToken(loaded, [tokenId], pastKv);
  }

  /** Multi-token decoder_step: feeds K tokens at once, returns K logits vectors.
   *  The model was exported with dynamic sequence length — input_ids [1, K]
   *  produces logits [1, K, vocab] and advances present KV by K positions.
   *  GPU ArgMax (next_token_id) is only valid for K=1. */
  private async runDecoderStepMultiToken(
    loaded: Required<Pick<LoadedExecutorState, 'decoderStepSession' | 'ort'>> & LoadedExecutorState,
    tokenIds: number[],
    pastKv: Record<string, OrtTensorLike<Float32Array>>,
  ): Promise<{
    logits: Float32Array;
    vocabSize: number;
    nextTokenId?: number;
    presentKv: Record<string, OrtTensorLike<Float32Array>>;
    timings: DecoderSessionTiming;
  }> {
    const K = tokenIds.length;
    const inputStart = nowMs();
    const inputIdsTensor = new loaded.ort.Tensor(
      'int64',
      new BigInt64Array(tokenIds.map((id) => BigInt(id))),
      [1, K],
    );
    const feeds: Record<string, unknown> = { input_ids: inputIdsTensor };

    // Add all past_key_values (decoder + encoder KV). Step model expects both.
    // CRITICAL: Clone tensor data for cross-session safety. ORT WASM cannot
    // reuse tensor objects from one session as inputs to another.
    for (const [name, tensor] of Object.entries(pastKv)) {
      if (isGpuBufferTensor(tensor)) {
        feeds[name] = tensor;
      } else {
        const cloned = cloneDecoderKvTensorDataForInput(tensor);
        feeds[name] = new loaded.ort.Tensor(cloned.type, cloned.data, tensor.dims);
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
    const featureTensor = await maybeCastWhisperFeatureTensor(new loaded.ort.Tensor(
      'float32',
      paddedFeatures,
      [1, melBins, melInputFrames],
    ), loaded.encoderSession, loaded.ort);

    // 2. Run encoder
    const encoderOutputs = await loaded.encoderSession.run({
      input_features: featureTensor,
    });
    const encoderHiddenStates = await maybeCastEncoderHiddenStates(
      encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>,
      loaded.decoderInitSession ?? loaded.decoderSession!,
      loaded.ort,
    );

    // 3. Detect language if auto
    const tokenizer = loaded.tokenizer;
    let language = options.language ?? this.config.languages[0] ?? 'auto';
    if (language === 'auto' && loaded.decoderSession) {
      language = await this.detectLanguageFromMergedDecoder(loaded, encoderHiddenStates);
    }
    language = resolveWhisperLanguageCode(language, this.config.languages);
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      resolveWhisperLanguageTokenId(tokenizer, language),
      tokenizer.getTokenId(taskToken) ?? 50360,
    ];
    if (noTimestampsToken) {
      const ntId = tokenizer.getTokenId(noTimestampsToken);
      if (ntId !== undefined) {
        promptTokens.push(ntId);
      }
    }

    // 4. Decode loop (greedy by default, beam search when numBeams > 1)
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
    const noSpeechTokenId = loaded.generationConfig.noSpeechTokenId
      ?? tokenizer.getTokenId('<|nospeech|>')
      ?? 50362;
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

    let selectedTokenTraces: Array<{ tokenId: number; logProb: number; entropy: number }> = [];

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
        if (step === 0) {
          options.onDecoderInitLogits?.(new Float32Array(result.lastLogits), {
            tokens: promptTokens,
            beginIndex: promptTokens.length,
            vocabSize: result.vocabSize,
            noSpeechTokenId,
          });
        }
        timestampProcessor.process(result.lastLogits, generatedTokens, promptTokens.length);
        const nextTokenId = argmax(result.lastLogits);
        generatedTokens.push(nextTokenId);

        const quality = tokenQualityFromLogits(
          result.lastLogits,
          nextTokenId,
          result.vocabSize,
        );
        selectedTokenTraces.push({
          tokenId: nextTokenId,
          logProb: quality.logProb,
          entropy: quality.entropy,
        });

        const tokenText = this.formatTokenText(tokenizer, nextTokenId);
        tokenDetails.push({
          index: step,
          id: nextTokenId,
          text: tokenText,
          confidence: quality.confidence,
          special: tokenizer.isSpecialTokenId(nextTokenId),
        });

        if (nextTokenId === eosId) break;
      }
    } else {
      let beams: WhisperBeamState<BeamPayload>[] = [
        createInitialWhisperBeam(promptTokens, 0, { tokenDetails: [], pastKeyValues: {}, tokenTraces: [] }),
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
          if (step === 0) {
            options.onDecoderInitLogits?.(new Float32Array(result.lastLogits), {
              tokens: promptTokens,
              beginIndex: promptTokens.length,
              vocabSize: result.vocabSize,
              noSpeechTokenId,
            });
          }
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
          expandPayload: (beam, tokenId, logProb) => {
            const parentLogits = logitsByBeam[beams.indexOf(beam)] ?? new Float32Array(0);
            const quality = tokenQualityFromLogits(parentLogits, tokenId, vocabSize);
            const tokenText = this.formatTokenText(tokenizer, tokenId);
            return {
              tokenDetails: [
                ...(beam.payload?.tokenDetails ?? []),
                {
                  index: step,
                  id: tokenId,
                  text: tokenText,
                  confidence: quality.confidence,
                  special: tokenizer.isSpecialTokenId(tokenId),
                },
              ],
              pastKeyValues: nextPastByBeam.get(beam) ?? {},
              tokenTraces: [
                ...(beam.payload?.tokenTraces ?? []),
                {
                  tokenId,
                  logProb,
                  entropy: quality.entropy,
                },
              ],
            };
          },
        });
      }

      const bestBeam = selectBestWhisperBeam(beams, lengthPenalty);
      tokenDetails = [...(bestBeam?.payload?.tokenDetails ?? [])];
      selectedTokenTraces = [...(bestBeam?.payload?.tokenTraces ?? [])];
    }

    // 5. Build segments from decoded tokens
    const segments = this.buildSegments(tokenDetails, tokenizer, options.noTimestamps);
    const alignedWords = this.shouldReturnWordTimestamps(options)
      ? await this.computeAttentionWordTimestamps(
          loaded,
          encoderHiddenStates,
          tokenizer,
          tokenDetails,
          segments,
          language,
          options,
          audio.durationSeconds,
        )
      : [];
    const words = this.shouldReturnWordTimestamps(options)
      ? await this.finalizeWordTimestamps(
          alignedWords,
          tokenDetails,
          tokenizer,
          language,
          options,
          audio,
          warnings,
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
      ...(selectedTokenTraces.length > 0 ? { tokenTraces: selectedTokenTraces } : {}),
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

  private resolveWordTimestamps(
    alignedWords: readonly WhisperNativeWord[] | undefined,
    tokens: readonly WhisperNativeToken[],
    tokenizer: WhisperTokenizer,
    language: string,
  ): WhisperNativeWord[] {
    return coalesceWhisperWordTimestamps(alignedWords, tokens, {
      timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
      timestampEnd: tokenizer.getTokenId('<|30.00|>') ?? 51864,
      language,
    });
  }

  private async finalizeWordTimestamps(
    alignedWords: readonly WhisperNativeWord[] | undefined,
    tokens: readonly WhisperNativeToken[],
    tokenizer: WhisperTokenizer,
    language: string,
    options: WhisperSeq2SeqTranscriptionOptions,
    audio: AudioBufferLike,
    warnings: TranscriptWarning[],
  ): Promise<WhisperNativeWord[]> {
    const words = this.resolveWordTimestamps(alignedWords, tokens, tokenizer, language);
    if (!options.wordAligner || words.length === 0) {
      return clipShortWhisperWordDurations(words);
    }

    const groups = splitWhisperWordsByPause(words);
    const refined: WhisperNativeWord[] = [];
    let failedGroups = 0;
    for (const group of groups) {
      try {
        const window = alignmentWindowForWhisperWords(group, audio.durationSeconds);
        const sliced = sliceAudioBufferLike(audio, window.startSeconds, window.endSeconds);
        const aligned = await options.wordAligner.align({
          transcript: group.map((word) => word.text).join(' '),
          audio: sliced,
          durationSeconds: sliced.durationSeconds,
          language,
        });
        if (aligned.length === 0) {
          failedGroups += 1;
          refined.push(...clipShortWhisperWordDurations(group));
          continue;
        }
        const shifted = aligned.map((word) => ({
          ...word,
          startTime: word.startTime + window.startSeconds,
          endTime: word.endTime + window.startSeconds,
        }));
        if (!forcedAlignmentLooksAnchored(group, shifted)) {
          failedGroups += 1;
          refined.push(...clipShortWhisperWordDurations(group));
          continue;
        }
        refined.push(...refineWhisperWordsWithForcedAlignment(group, shifted));
      } catch {
        failedGroups += 1;
        refined.push(...clipShortWhisperWordDurations(group));
      }
    }
    if (failedGroups > 0) {
      warnings.push({
        code: 'whisper.word-alignment-partial',
        message: `Wav2Vec2 word alignment failed for ${failedGroups} of ${groups.length} pause group(s); DTW timestamps were kept for those groups.`,
        recoverable: true,
      });
    }
    return clipShortWhisperWordDurations(refined);
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

        return selectWhisperLanguageFromLogits(loaded.tokenizer, logitsData.data, vocabSize) ?? 'auto';
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

  /**
   * Detect language from encoder output using the merged decoder with single start token.
   * Returns language code (e.g. 'en', 'tr') or 'auto' if detection fails.
   */
  private async detectLanguageFromMergedDecoder(
    loaded: LoadedExecutorState,
    encoderHiddenStates: OrtTensorLike<Float32Array>,
  ): Promise<string> {
    try {
      const sotId = loaded.tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
      const result = await this.runDecoderStep(loaded, encoderHiddenStates, [sotId], {}, true);
      try {
        return selectWhisperLanguageFromLogits(loaded.tokenizer, result.lastLogits, result.vocabSize) ?? 'auto';
      } finally {
        // Dispose present KV tensors from the probe to avoid leaking GPU memory.
        for (const tensor of Object.values(result.pastKeyValues)) {
          disposeGpuTensor(tensor);
        }
      }
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
    readonly onDecoderInitLogits?: WhisperSeq2SeqTranscriptionOptions['onDecoderInitLogits'];
    readonly noSpeechTokenId?: number;
    readonly trackQuality?: boolean;
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
      onDecoderInitLogits,
      noSpeechTokenId,
      trackQuality = false,
      onInitTiming,
      onStepTiming,
    } = params;

    const initStart = nowMs();
    const init = await this.runDecoderInit(loaded, encoderHiddenStates, promptTokens);
    onInitTiming?.(init.timings, nowMs() - initStart);

    const vocabSize = init.vocabSize;
    const firstLogits = init.logits.subarray(init.logits.length - vocabSize);
    onDecoderInitLogits?.(new Float32Array(firstLogits), {
      tokens: promptTokens,
      beginIndex: promptTokens.length,
      vocabSize,
      noSpeechTokenId,
    });
    processLogits?.(firstLogits, promptTokens, promptTokens.length);

    const firstTokenId = argmax(firstLogits);
    const tokens: number[] = [firstTokenId];
    const tokenTraces: Array<{ tokenId: number; logProb: number; entropy: number }> = [];
    if (trackQuality) {
      const firstQuality = tokenQualityFromLogits(firstLogits, firstTokenId);
      tokenTraces.push({
        tokenId: firstTokenId,
        logProb: firstQuality.logProb,
        entropy: firstQuality.entropy,
      });
    }
    onTokenLogits?.(firstTokenId, firstLogits, { tokens, beginIndex: promptTokens.length });

    const finish = (): SplitGraphDecodeResult => (
      trackQuality ? { tokens, tokenTraces } : { tokens }
    );

    let pastKv = mapPresentKvToPastKv(init.presentKv);
    try {
      if (firstTokenId === eosTokenId) {
        return finish();
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
        if (trackQuality) {
          const stepQuality = tokenQualityFromLogits(step.logits, nextTokenId);
          tokenTraces.push({
            tokenId: nextTokenId,
            logProb: stepQuality.logProb,
            entropy: stepQuality.entropy,
          });
        }
        onTokenLogits?.(nextTokenId, step.logits, { tokens, beginIndex: promptTokens.length });

        if (nextTokenId === eosTokenId) {
          break;
        }
      }

      return finish();
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

    const requestedNumBeams = Math.max(1, Math.floor(options.numBeams ?? 1));
    const requestedBestOf = Math.max(1, Math.floor(options.bestOf ?? 1));
    const requestedTemperature = options.temperature ?? 0;
    const useExperimentalGpuKvCache = Boolean(
      splitLoaded.experimentalGpuKvCache &&
      splitLoaded.decoderBackendForOrt === 'webgpu',
    );

    // Reject an unsupported decode policy before allocating mel/encoder work.
    // GPU-KV remains a greedy-only fast path until cache cloning/reordering is
    // proven correct for beams and sampling.
    assertExperimentalGpuKvCacheIsGreedyOnly({
      enabled: useExperimentalGpuKvCache,
      numBeams: requestedNumBeams,
      bestOf: requestedBestOf,
      temperature: requestedTemperature,
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

    const featureTensor = await maybeCastWhisperFeatureTensor(new loaded.ort.Tensor(
      'float32', paddedFeatures,
      [1, melBins, melInputFrames],
    ), loaded.encoderSession, loaded.ort);

    // 2. Run encoder
    const encodeStart = nowMs();
    const encoderRunStart = nowMs();
    const encoderOutputs = await loaded.encoderSession.run({ input_features: featureTensor });
    const encoderRunEnd = nowMs();
    const encoderOutputTensorStart = nowMs();
    let encoderHiddenStates = await maybeCastEncoderHiddenStates(
      encoderOutputs[Object.keys(encoderOutputs)[0]!] as OrtTensorLike<Float32Array>,
      loaded.decoderInitSession ?? loaded.decoderSession!,
      loaded.ort,
    );
    const encoderOutputCastEnd = nowMs();
    const encoderOutputEnd = nowMs();
    const encodeMs = nowMs() - encodeStart;
    // DIAGNOSTIC: sub-timing for encoder run vs output processing
    const encoderRunMs = encoderRunEnd - encoderRunStart;
    const encoderOutputMs = encoderOutputEnd - encoderRunEnd;
    // DIAGNOSTIC (ORT-FLUSH): fp32 path forces GPU flush inside maybeCastEncoderHiddenStates
    // via getData(true).  This hides ~193ms in encoderOutputCastMs instead of decoderInitMs.
    const encoderOutputCastMs = encoderOutputCastEnd - encoderOutputTensorStart;
    const encoderOutputLocation = (encoderHiddenStates as OrtTensorLike<Float32Array>).location ?? 'cpu';
    const encoderOutputDtype = (encoderHiddenStates as OrtTensorLike<Float32Array>).type ?? 'float32';

    // EDGE A DIAGNOSTIC: Re-wrap encoder GPU output as a fresh Tensor.fromGpuBuffer.
    // The encoder output tensor carries the encoder session's downloader/disposer
    // callbacks. Re-wrapping the same GPUBuffer as a fresh tensor strips those
    // callbacks, testing whether the session association causes the fp16 handoff penalty.
    let encoderBufferRewrapMs = 0;
    let encoderGpuFlushMs = 0;
    if (loaded.encoderBufferRewrap && isGpuBufferTensor(encoderHiddenStates as OrtTensorLike<Float32Array>)) {
      const rewrapStart = nowMs();
      const origTensor = encoderHiddenStates as OrtTensorLike<Float32Array>;
      const gpuBuffer = (origTensor as unknown as { gpuBuffer: GPUBuffer }).gpuBuffer;
      if (gpuBuffer && loaded.ort.Tensor.fromGpuBuffer) {
        const rewrapped = loaded.ort.Tensor.fromGpuBuffer(gpuBuffer, {
          dataType: origTensor.type as string,
          dims: [...origTensor.dims] as readonly number[],
          download: origTensor.getData
            ? async () => origTensor.getData!() as Promise<ArrayBufferView>
            : undefined,
          dispose: undefined,
        });
        encoderHiddenStates = rewrapped as unknown as OrtTensorLike<Float32Array>;
      }
      encoderBufferRewrapMs = nowMs() - rewrapStart;
    }

    // EDGE B2 DIAGNOSTIC: Force GPU pipeline flush by calling getData() on the
    // encoder output, then re-wrap the SAME GPUBuffer as a fresh tensor.
    // This tests whether the 197ms penalty is caused by GPU synchronization
    // (encoder compute pass not yet submitted/completed when decoder_init starts).
    // If decoderInitMs drops to ~19ms after the flush, synchronization is the cause.
    if (loaded.encoderGpuFlush && isGpuBufferTensor(encoderHiddenStates as OrtTensorLike<Float32Array>)) {
      const flushStart = nowMs();
      const origTensor = encoderHiddenStates as OrtTensorLike<Float32Array>;
      const gpuBuffer = (origTensor as unknown as { gpuBuffer: GPUBuffer }).gpuBuffer;
      const dims = [...origTensor.dims] as readonly number[];
      const dtype = origTensor.type as string;
      // Force GPU pipeline flush by downloading to CPU
      if (origTensor.getData) {
        await origTensor.getData(false); // false = don't release GPU buffer
      }
      // Re-wrap the SAME GPUBuffer as a fresh tensor (data is already computed on GPU)
      if (gpuBuffer && loaded.ort.Tensor.fromGpuBuffer) {
        encoderHiddenStates = loaded.ort.Tensor.fromGpuBuffer(gpuBuffer, {
          dataType: dtype,
          dims,
          download: undefined,
          dispose: undefined,
        }) as unknown as OrtTensorLike<Float32Array>;
      }
      encoderGpuFlushMs = nowMs() - flushStart;
    }

    // PROFILING (encoderGpuDrain): Force GPU drain after encoder for honest
    // per-phase metrics.  ORT Submit() is non-blocking — the encoder's GPU
    // compute work (~178ms) hasn't finished when session.run() returns.
    // Without this drain, the cost silently appears in decoderInitMs (both
    // sessions share the same device queue).  This forces the GPU to drain
    // so encoderTotalMs tells the truth and decoderInitMs shows only its own
    // compute cost.
    //
    // This calls getData(false) which adds ~18ms staging-buffer overhead vs
    // a native fence.  It is a PROFILING OPTION, not production behavior.
    // In production, the fp16 pass-through path avoids this readback and
    // lets the queue dependency resolve naturally — the total latency is
    // the same, only the metric attribution differs.
    //
    // Skip if Edge B2 already drained (encoderGpuFlushMs > 0).
    let encoderGpuDrainMs = 0;
    if (encoderGpuFlushMs === 0 &&
        loaded.encoderGpuDrain &&
        isGpuBufferTensor(encoderHiddenStates as OrtTensorLike<Float32Array>)) {
      const drainStart = nowMs();
      const origTensor = encoderHiddenStates as OrtTensorLike<Float32Array>;
      const gpuBuffer = (origTensor as unknown as { gpuBuffer: GPUBuffer }).gpuBuffer;
      const dims = [...origTensor.dims] as readonly number[];
      const dtype = origTensor.type as string;
      if (origTensor.getData && gpuBuffer) {
        await origTensor.getData(false); // force GPU drain, keep buffer alive
      }
      if (gpuBuffer && loaded.ort.Tensor.fromGpuBuffer) {
        encoderHiddenStates = loaded.ort.Tensor.fromGpuBuffer(gpuBuffer, {
          dataType: dtype,
          dims,
          download: undefined,
          dispose: undefined,
        }) as unknown as OrtTensorLike<Float32Array>;
      }
      encoderGpuDrainMs = nowMs() - drainStart;
    }
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
      language = resolveWhisperLanguageCode(language, this.config.languages);
    }
    const taskToken = options.task === 'translate' ? '<|translate|>' : '<|transcribe|>';
    const noTimestampsToken = options.noTimestamps ? '<|notimestamps|>' : undefined;

    const promptTokens: number[] = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      resolveWhisperLanguageTokenId(tokenizer, language),
      tokenizer.getTokenId(taskToken) ?? 50360,
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
    const noSpeechTokenId = loaded.generationConfig.noSpeechTokenId
      ?? tokenizer.getTokenId('<|nospeech|>')
      ?? 50362;
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
    let decoderInitTensorCreateMs = 0;
    let decoderInitLogitReadMs = 0;
    let decoderInitKvExtractMs = 0;
    let decoderStepTensorCreateMs = 0;
    let decoderStepLogitReadMs = 0;
    let decoderStepKvMergeMs = 0;
    const decoderStepTimings: number[] = [];
    const requestedDecoderKvCacheLocation = useExperimentalGpuKvCache ? 'gpu-buffer' : 'cpu';
    const recordDecoderTiming = (timings: DecoderSessionTiming): void => {
      decoderGpuTensorInputs += timings.gpuInputCount;
      decoderCpuTensorInputs += timings.cpuInputCount;
      decoderGpuTensorOutputs += timings.gpuOutputCount;
      decoderCpuTensorOutputs += timings.cpuOutputCount;
      decoderGpuTensorDownloads += timings.gpuDownloadCount;
    };

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
          onDecoderInitLogits: options.onDecoderInitLogits,
          noSpeechTokenId,
          trackQuality: options.trackQuality === true,
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
          onDecoderInitLogits: options.onDecoderInitLogits,
          noSpeechTokenId,
          numBeams: requestedNumBeams,
          lengthPenalty: options.lengthPenalty,
          patience: options.patience ?? 1,
          temperature: requestedTemperature,
          bestOf: requestedBestOf,
          experimentalBatchedBeam: options.experimentalBatchedBeam === true,
          trackQuality: options.trackQuality === true,
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
                Object.entries(init.presentKv).map(([k, v]) => [
                  k,
                  {
                    data: v.data as ArrayBufferView,
                    dims: v.dims,
                    type: v.type,
                  },
                ]),
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
            for (const [name, value] of Object.entries(pastKv)) {
              const kvValue = normalizeWhisperKvCacheValue(value);
              const stepName = name.replace(/^present\./, 'past_key_values.');
              const cloned = cloneDecoderKvDataForInput(kvValue.data, kvValue.type ?? kvDtype);
              // Try multiple key formats for dims lookup (init uses present.*, step uses past_key_values.*)
              const dims = kvValue.dims
                ?? kvDims[name]
                ?? kvDims[stepName]
                ?? kvDims[name.replace(/^past_key_values\./, 'present.')];
              if (dims) {
                feeds[stepName] = new splitLoaded.ort.Tensor(cloned.type, cloned.data, dims) as unknown as OrtTensorLike<Float32Array>;
              } else {
                const numHeads = splitLoaded.modelConfig.decoderAttentionHeads;
                const headDim = splitLoaded.modelConfig.headDim;
                const seqLen = Math.round(cloned.data.length / (numHeads * headDim));
                feeds[stepName] = new splitLoaded.ort.Tensor(
                  cloned.type,
                  cloned.data,
                  [1, numHeads, seqLen, headDim],
                ) as unknown as OrtTensorLike<Float32Array>;
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
                Object.entries(step.presentKv).map(([k, v]) => [
                  k,
                  {
                    data: v.data as ArrayBufferView,
                    dims: v.dims,
                    type: v.type,
                  },
                ]),
              ),
            };
          },
          runStepBatch: options.experimentalBatchedBeam
            ? async (tokenIds, pastKvs) => {
                const batchSize = tokenIds.length;
                if (batchSize === 0) return [];
                const decoderStepStart = nowMs();
                const feedBuildStart = nowMs();
                const feeds: Record<string, unknown> = {
                  input_ids: new splitLoaded.ort.Tensor(
                    'int64',
                    new BigInt64Array(tokenIds.map((id) => BigInt(id))),
                    [batchSize, 1],
                  ),
                };

                const firstKv = pastKvs[0] ?? {};
                for (const name of Object.keys(firstKv)) {
                  const stepName = name.replace(/^present\./, 'past_key_values.');
                  const values = pastKvs.map((kv) => {
                    const value = kv[name]
                      ?? kv[stepName]
                      ?? kv[name.replace(/^past_key_values\./, 'present.')];
                    if (!value) {
                      throw new Error(`Missing Whisper batched beam KV "${name}" for one active beam.`);
                    }
                    return normalizeWhisperKvCacheValue(value);
                  });
                  const firstValue = values[0]!;
                  const dims = firstValue.dims
                    ?? kvDims[name]
                    ?? kvDims[stepName]
                    ?? kvDims[name.replace(/^past_key_values\./, 'present.')];
                  let perBeamDims = dims;
                  if (!perBeamDims) {
                    const numHeads = splitLoaded.modelConfig.decoderAttentionHeads;
                    const headDim = splitLoaded.modelConfig.headDim;
                    const clonedFirst = cloneDecoderKvDataForInput(firstValue.data, firstValue.type ?? kvDtype);
                    const seqLen = Math.round(clonedFirst.data.length / (numHeads * headDim));
                    perBeamDims = [1, numHeads, seqLen, headDim];
                  }
                  for (const value of values) {
                    if (!value.dims) continue;
                    const sameRank = value.dims.length === perBeamDims.length;
                    const sameNonBatchDims = value.dims.slice(1).every((dim, i) => dim === perBeamDims![i + 1]);
                    if (!sameRank || !sameNonBatchDims) {
                      throw new Error(`Cannot batch Whisper KV "${name}" with mismatched per-beam dims.`);
                    }
                  }
                  const batched = concatDecoderKvDataForBatch(values, kvDtype);
                  feeds[stepName] = new splitLoaded.ort.Tensor(
                    batched.type,
                    batched.data,
                    [batchSize, ...perBeamDims.slice(1)],
                  );
                }

                decoderStepFeedBuildMs += nowMs() - feedBuildStart;
                const inputLocations = countTensorLocations(Object.values(feeds));
                const runStart = nowMs();
                const outputs = await splitLoaded.decoderStepSession.run(feeds);
                const outputStart = nowMs();
                const outputLocations = countTensorLocations(Object.values(outputs));
                const logitsKey = Object.keys(outputs).find((k) => k.includes('logits')) ?? Object.keys(outputs)[0]!;
                const logitsTensor = outputs[logitsKey] as OrtTensorLike<Float32Array>;
                const logitReadStart = nowMs();
                const logitsData = await readOrtTensorData(logitsTensor, { releaseGpu: true });
                const logitReadMs = nowMs() - logitReadStart;
                const logitsDims = logitsTensor.dims;
                const vocabSize = logitsDims[logitsDims.length - 1] ?? 0;
                const logitsBatch = logitsDims[0] ?? batchSize;
                if (logitsBatch !== batchSize) {
                  throw new Error(`Batched Whisper decoder logits batch ${logitsBatch} does not match active beams ${batchSize}.`);
                }
                const perBeamLogitSpan = logitsData.data.length / batchSize;
                if (!Number.isInteger(perBeamLogitSpan) || perBeamLogitSpan < vocabSize) {
                  throw new Error('Batched Whisper decoder logits shape is not divisible by active beam count.');
                }
                const logitTimeSteps = Math.max(1, Math.floor(perBeamLogitSpan / vocabSize));
                const lastLogitOffset = (logitTimeSteps - 1) * vocabSize;
                const results = Array.from({ length: batchSize }, (_, beamIndex) => ({
                  logits: new Float32Array(
                    logitsData.data.subarray(
                      beamIndex * perBeamLogitSpan + lastLogitOffset,
                      beamIndex * perBeamLogitSpan + lastLogitOffset + vocabSize,
                    ),
                  ),
                  vocabSize,
                  presentKv: {} as WhisperKvCache,
                }));

                const kvStart = nowMs();
                let outputDownloadCount = logitsData.downloaded ? 1 : 0;
                for (const [key, value] of Object.entries(outputs)) {
                  if (!key.startsWith('present')) continue;
                  const pastName = key.replace(/^present/, 'past_key_values');
                  const tensor = value as OrtTensorLike<Float32Array>;
                  const tensorData = await readOrtTensorData(tensor, { releaseGpu: true });
                  if (tensorData.downloaded) outputDownloadCount += 1;
                  const data = tensorData.data as unknown as TensorDataView;
                  const tensorBatch = tensor.dims[0] ?? batchSize;
                  if (tensorBatch !== batchSize) {
                    throw new Error(`Batched Whisper KV "${key}" batch ${tensorBatch} does not match active beams ${batchSize}.`);
                  }
                  const perBeamLength = data.length / batchSize;
                  if (!Number.isInteger(perBeamLength)) {
                    throw new Error(`Batched Whisper KV "${key}" data length is not divisible by active beam count.`);
                  }
                  const perBeamDims = [1, ...tensor.dims.slice(1)];
                  for (let beamIndex = 0; beamIndex < batchSize; beamIndex++) {
                    results[beamIndex]!.presentKv[pastName] = {
                      data: sliceTensorDataView(data, beamIndex * perBeamLength, perBeamLength),
                      dims: perBeamDims,
                      type: tensor.type,
                    };
                  }
                }

                for (let beamIndex = 0; beamIndex < batchSize; beamIndex++) {
                  for (const [name, value] of Object.entries(pastKvs[beamIndex] ?? {})) {
                    const stepName = name.replace(/^present\./, 'past_key_values.');
                    if (!stepName.includes('encoder') || results[beamIndex]!.presentKv[stepName]) continue;
                    const kvValue = normalizeWhisperKvCacheValue(value);
                    results[beamIndex]!.presentKv[stepName] = {
                      data: kvValue.data,
                      ...(kvValue.dims ? { dims: kvValue.dims } : {}),
                      type: kvValue.type ?? kvDtype,
                    };
                  }
                }

                for (const result of results) {
                  for (const [k, v] of Object.entries(result.presentKv)) {
                    const kvValue = normalizeWhisperKvCacheValue(v);
                    if (kvValue.dims) kvDims[k] = kvValue.dims;
                  }
                }

                const kvEnd = nowMs();
                const outputEnd = nowMs();
                const decoderStepTotal = nowMs() - decoderStepStart;
                decoderStepMs += decoderStepTotal;
                decoderStepTensorCloneMs += runStart - feedBuildStart;
                decoderStepRunMs += outputStart - runStart;
                decoderStepOutputMs += outputEnd - outputStart;
                decoderStepTensorCreateMs += runStart - feedBuildStart;
                decoderStepLogitReadMs += logitReadMs;
                decoderStepKvMergeMs += kvEnd - kvStart;
                decoderStepCount += 1;
                decoderStepTimings.push(decoderStepTotal);
                recordDecoderTiming({
                  inputMs: runStart - feedBuildStart,
                  runMs: outputStart - runStart,
                  outputMs: outputEnd - outputStart,
                  tensorCreateMs: runStart - feedBuildStart,
                  logitReadMs,
                  kvExtractMs: kvEnd - kvStart,
                  gpuInputCount: inputLocations.gpu,
                  cpuInputCount: inputLocations.cpu,
                  gpuOutputCount: outputLocations.gpu,
                  cpuOutputCount: outputLocations.cpu,
                  gpuDownloadCount: outputDownloadCount,
                });

                return results;
              }
            : undefined,
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
    const alignedWords = this.shouldReturnWordTimestamps(options)
      ? await this.computeAttentionWordTimestampsSplitGraph(
          loaded,
          encoderHiddenStates,
          tokenizer,
          tokenDetails,
          generatedTokens,
          promptTokens.length,
          language,
          options,
          audio.durationSeconds,
          warnings,
        )
      : [];
    const words = this.shouldReturnWordTimestamps(options)
      ? await this.finalizeWordTimestamps(
          alignedWords,
          tokenDetails,
          tokenizer,
          language,
          options,
          audio,
          warnings,
        )
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
      // DIAGNOSTIC (ORT-FLUSH): cast/download time — fp32 path hides GPU flush here
      encoderOutputCastMs: roundMetric(encoderOutputCastMs),
      encoderOutputLocation,
      encoderOutputDtype,
      // DIAGNOSTIC: Edge A re-wrap timing
      encoderBufferRewrapMs: roundMetric(encoderBufferRewrapMs),
      // DIAGNOSTIC: Edge B2 GPU flush timing (now redundant with encoderGpuDrainMs)
      encoderGpuFlushMs: roundMetric(encoderGpuFlushMs),
      // PROFILING (encoderGpuDrain): GPU drain after encoder — gated, adds ~18ms overhead
      encoderGpuDrainMs: roundMetric(encoderGpuDrainMs),
      // PROFILING: encoder total when drain is active (encoderRunMs + encoderGpuDrainMs)
      encoderTotalMs: roundMetric(encoderRunMs + encoderGpuDrainMs),
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
      ...(result.tokenTraces ? { tokenTraces: result.tokenTraces } : {}),
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
