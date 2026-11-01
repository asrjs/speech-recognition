import type { AbortSignalLike, TranscriptionProgressEvent } from '../../types/index.js';
import { PipelineAbortedError } from '../../pipeline/composition.js';
import { roundMetric } from '../../runtime/timing.js';
import type { WhisperKvCacheValue } from './core.js';
import type {
  TensorLocationCounts,
  WhisperEncoderKvTensorCacheEntry,
  WhisperPresentKvSnapshot,
} from './executor-types.js';
import { tensorDataAsFloat32 } from './executor-utils.js';
import type { WhisperModelConfig } from './generation-config.js';
import type {
  OrtModuleLike,
  OrtPreferredOutputLocation,
  OrtSessionLike,
  OrtTensorLike,
} from './ort.js';
import type { WhisperSeq2SeqTranscriptionOptions } from './types.js';

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

export function roundMiB(bytes: number | undefined): number | undefined {
  if (!Number.isFinite(bytes)) return undefined;
  return roundMetric((bytes as number) / (1024 * 1024), 2);
}

export function clampProgress(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function estimateRemainingMs(elapsedMs: number, progress: number): number | undefined {
  if (progress <= 0 || progress >= 1) {
    return undefined;
  }

  return roundMetric((elapsedMs / progress) * (1 - progress), 2);
}

export function emitTranscriptionProgress(
  options: WhisperSeq2SeqTranscriptionOptions,
  event: TranscriptionProgressEvent,
): void {
  options.onProgress?.(event);
}

export function createWhisperGpuKvOutputLocation(
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

export function createWhisperInt64TokenIds(tokenIds: readonly number[]): BigInt64Array {
  const result = new BigInt64Array(tokenIds.length);
  for (let index = 0; index < tokenIds.length; index++) {
    result[index] = BigInt(tokenIds[index]!);
  }
  return result;
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

export function isOrtTensorLike(value: unknown): value is OrtTensorLike {
  return Boolean(value) && typeof value === 'object' && Array.isArray((value as { dims?: unknown }).dims);
}

export function isGpuBufferTensor(tensor: OrtTensorLike): boolean {
  return tensor.location === 'gpu-buffer';
}

export function countTensorLocations(values: Iterable<unknown>): TensorLocationCounts {
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

export async function readOrtTensorData<TData extends ArrayBufferView>(
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

export function disposeOrtTensor(tensor: OrtTensorLike | undefined): void {
  tensor?.dispose?.();
}

export function throwIfDecodeAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('decode');
  }
}

export function disposeGpuTensor(tensor: OrtTensorLike | undefined): void {
  disposeOrtTensor(tensor);
}

/** Free decoder-step feed tensors that this call owns. Caller cache tensors stay in `retain`. */
export function disposeOwnedDecoderFeeds(
  feeds: Record<string, unknown>,
  retain: ReadonlySet<unknown> = new Set(),
): void {
  for (const value of Object.values(feeds)) {
    if (!value || retain.has(value)) continue;
    disposeOrtTensor(value as OrtTensorLike);
  }
}

export async function withOwnedDecoderFeeds<T>(
  feeds: Record<string, unknown>,
  retain: ReadonlySet<unknown>,
  run: () => Promise<T>,
): Promise<T> {
  try {
    return await run();
  } finally {
    disposeOwnedDecoderFeeds(feeds, retain);
  }
}

export function disposeOwnedWhisperTensors(tensors: Set<OrtTensorLike | undefined>): void {
  for (const tensor of tensors) {
    disposeOrtTensor(tensor);
  }
  tensors.clear();
}

export function disposeWhisperEncoderKvCache(
  cache: Map<string, { readonly tensor: OrtTensorLike }>,
): void {
  disposeWhisperEncoderKvCaches(cache);
}

export function disposeWhisperEncoderKvCaches(
  ...caches: Array<Map<string, { readonly tensor: OrtTensorLike }>>
): void {
  disposeOwnedWhisperTensors(
    new Set(caches.flatMap((cache) => [...cache.values()].map((entry) => entry.tensor))),
  );
  for (const cache of caches) cache.clear();
}

export function disposeReplacedOrtKv(
  previous: Record<string, OrtTensorLike | undefined>,
  next: Record<string, OrtTensorLike | undefined>,
): void {
  for (const [key, tensor] of Object.entries(previous)) {
    if (next[key] !== tensor) disposeOrtTensor(tensor);
  }
}

export function disposeOrtKv(kv: Record<string, OrtTensorLike | undefined>): void {
  for (const tensor of Object.values(kv)) disposeOrtTensor(tensor);
}

/** Copy present-KV views into owned JS buffers, then dispose Ort wrappers except retained encoder cache. */
export function copyAndReleaseWhisperPresentKv(
  presentKv: Record<string, OrtTensorLike<Float32Array>>,
  retained: ReadonlyMap<OrtTensorLike, WhisperPresentKvSnapshot> = new Map(),
): Record<string, WhisperPresentKvSnapshot> {
  const copied: Record<string, WhisperPresentKvSnapshot> = {};
  for (const [key, tensor] of Object.entries(presentKv)) {
    const kept = retained.get(tensor);
    if (kept) {
      copied[key] = kept;
      continue;
    }
    const cloned = cloneDecoderKvDataForInput(tensor.data as ArrayBufferView, tensor.type);
    copied[key] = {
      data: cloned.data,
      dims: [...tensor.dims],
      type: cloned.type,
    };
  }
  for (const tensor of Object.values(presentKv)) {
    if (!retained.has(tensor)) disposeOrtTensor(tensor);
  }
  return copied;
}

export function disposeReplacedGpuKv(
  previous: Record<string, OrtTensorLike>,
  next: Record<string, OrtTensorLike>,
): void {
  disposeReplacedOrtKv(previous, next);
}

export function disposeGpuKv(kv: Record<string, OrtTensorLike>): void {
  disposeOrtKv(kv);
}

export function copyLogitsTail(data: ArrayBufferView, vocabSize: number): Float32Array {
  const logits = tensorDataAsFloat32(data);
  if (!(vocabSize > 0) || logits.length === 0) return Float32Array.from(logits);
  const offset = Math.max(0, logits.length - vocabSize);
  return Float32Array.from(logits.subarray(offset));
}

export function copyTensorData(data: ArrayBufferView): Float32Array {
  return Float32Array.from(tensorDataAsFloat32(data));
}

export function disposeUnusedDecoderOutputs(
  outputs: Record<string, OrtTensorLike>,
  keep: ReadonlySet<OrtTensorLike>,
): void {
  for (const tensor of Object.values(outputs)) {
    if (!keep.has(tensor)) disposeOrtTensor(tensor);
  }
}

export function mapPresentKvToPastKv(
  presentKv: Record<string, OrtTensorLike<Float32Array>>,
): Record<string, OrtTensorLike<Float32Array>> {
  const mapped: Record<string, OrtTensorLike<Float32Array>> = {};
  for (const [name, tensor] of Object.entries(presentKv)) {
    mapped[name.replace(/^present\./, 'past_key_values.')] = tensor;
  }
  return mapped;
}

export type TensorDataView = ArrayBufferView & { readonly length: number };
export type MutableTensorDataView = TensorDataView & {
  [index: number]: number;
  set?: (source: ArrayLike<number>, offset?: number) => void;
};
export type TensorDataConstructor = {
  new(length: number): TensorDataView;
  new(buffer: ArrayBufferLike, byteOffset?: number, length?: number): TensorDataView;
  readonly name?: string;
};

function cloneTensorDataView(data: TensorDataView, ctor: TensorDataConstructor): TensorDataView {
  const buffer = (data.buffer as ArrayBuffer).slice(data.byteOffset, data.byteOffset + data.byteLength);
  return new ctor(buffer, 0, data.length);
}

export function decoderKvDataLayout(
  data: ArrayBufferView,
  tensorType?: string,
): { readonly type: 'float32' | 'float16'; readonly ctor: TensorDataConstructor } {
  const view = data as TensorDataView;
  const dataCtor = view.constructor as TensorDataConstructor;
  const isFloat16 = tensorType === 'float16' || dataCtor.name === 'Float16Array';
  const globalFloat16Ctor = (globalThis as unknown as { readonly Float16Array?: TensorDataConstructor }).Float16Array;
  return {
    type: isFloat16 ? 'float16' : 'float32',
    ctor: isFloat16 ? (globalFloat16Ctor ?? dataCtor) : dataCtor,
  };
}

/**
 * Split one batched decoder output into per-beam views without copying the
 * full KV payload again. The output buffer is retained by each view until the
 * next decoder call, which packs/clones it into fresh input storage. Keep the
 * copy fallback for unusual ArrayBufferView implementations.
 */
export function sliceDecoderKvDataForBatch(
  data: TensorDataView,
  elementOffset: number,
  length: number,
): TensorDataView {
  const sliceable = data as TensorDataView & {
    readonly subarray?: (begin?: number, end?: number) => TensorDataView;
  };
  if (typeof sliceable.subarray === 'function') {
    return sliceable.subarray(elementOffset, elementOffset + length);
  }
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
  const layout = decoderKvDataLayout(view, tensorType);
  return {
    type: layout.type,
    data: cloneTensorDataView(view, layout.ctor),
  };
}

function copyTensorDataViewInto(
  target: MutableTensorDataView,
  source: TensorDataView,
  offset: number,
): void {
  const targetBytesPerElement = target.byteLength / Math.max(1, target.length);
  if (source.byteLength % Math.max(1, targetBytesPerElement) !== 0) {
    // This should not occur for a valid Whisper KV tensor, but preserve a
    // deterministic fallback for unusual test/runtime views.
    const sourceValues = source as unknown as ArrayLike<number>;
    if (target.set) {
      target.set(sourceValues, offset);
      return;
    }
    for (let i = 0; i < source.length; i++) {
      target[offset + i] = sourceValues[i] ?? 0;
    }
    return;
  }
  const sourceBytes = new Uint8Array(
    source.buffer as ArrayBufferLike,
    source.byteOffset,
    source.byteLength,
  );
  const targetBytes = new Uint8Array(
    target.buffer as ArrayBufferLike,
    target.byteOffset + offset * targetBytesPerElement,
    source.byteLength,
  );
  targetBytes.set(sourceBytes);
}

/**
 * Pack per-beam KV inputs into one fresh batch tensor. Allocate the final
 * buffer once and copy each source directly; cloning each beam first and then
 * concatenating the clones doubled the memory bandwidth of every batched step.
 */
export function concatDecoderKvDataForBatch(
  values: readonly { readonly data: ArrayBufferView; readonly type?: string }[],
  fallbackType: 'float32' | 'float16',
): { readonly type: 'float32' | 'float16'; readonly data: TensorDataView } {
  const firstValue = values[0];
  if (!firstValue) {
    throw new Error('Cannot batch empty Whisper KV tensor data.');
  }
  const firstLayout = decoderKvDataLayout(firstValue.data, firstValue.type ?? fallbackType);
  const layouts = values.map((value) => ({
    value,
    layout: decoderKvDataLayout(value.data, value.type ?? fallbackType),
  }));
  for (const { layout } of layouts) {
    if (layout.type !== firstLayout.type) {
      throw new Error(`Cannot batch mixed Whisper KV tensor dtypes: ${firstLayout.type} and ${layout.type}.`);
    }
  }
  const merged = new firstLayout.ctor(
    values.reduce((sum, value) => sum + (value.data as TensorDataView).length, 0),
  ) as MutableTensorDataView;
  let offset = 0;
  for (const { value } of layouts) {
    const source = value.data as TensorDataView;
    copyTensorDataViewInto(merged, source, offset);
    offset += source.length;
  }
  return {
    type: firstLayout.type,
    data: merged,
  };
}

export function cloneDecoderKvTensorDataForInput(
  tensor: OrtTensorLike<Float32Array>,
): { readonly type: 'float32' | 'float16'; readonly data: TensorDataView } {
  const data = tensor.data as unknown as TensorDataView;
  const dataCtor = data.constructor as TensorDataConstructor;
  return cloneDecoderKvDataForInput(data, tensor.type ?? dataCtor.name);
}

export function normalizeWhisperKvCacheValue(
  value: WhisperKvCacheValue,
): { readonly data: ArrayBufferView; readonly dims?: readonly number[]; readonly type?: string } {
  if (ArrayBuffer.isView(value)) {
    return { data: value };
  }
  return value;
}

export function getWhisperDecoderCacheLength(
  pastKv: Record<string, WhisperKvCacheValue>,
  allowEmpty = false,
): number {
  const lengths = Object.entries(pastKv)
    .filter(([name]) => /^(?:past_key_values|present)\.\d+\.decoder\.(?:key|value)$/.test(name))
    .map(([name, value]) => {
      const dims = normalizeWhisperKvCacheValue(value).dims;
      const length = dims?.[2];
      if (!Number.isSafeInteger(length) || length! < 0) {
        throw new Error(`Whisper decoder cache "${name}" has no valid sequence length at dimension 2.`);
      }
      return length!;
    });
  if (lengths.length === 0) {
    if (allowEmpty) return 0;
    throw new Error('Whisper decoder cache_position requires decoder KV cache tensors.');
  }
  const length = lengths[0]!;
  if (lengths.some((candidate) => candidate !== length)) {
    throw new Error('Whisper decoder KV tensors have mismatched sequence lengths.');
  }
  return length;
}

export function createWhisperCachePosition(start: number, tokenCount: number): BigInt64Array {
  if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(tokenCount) || tokenCount < 1) {
    throw new Error('Whisper cache_position requires a non-negative cache length and at least one input token.');
  }
  const positions = new BigInt64Array(tokenCount);
  for (let index = 0; index < tokenCount; index++) positions[index] = BigInt(start + index);
  return positions;
}

export function createWhisperBatchedCachePosition(
  pastKvs: readonly Record<string, WhisperKvCacheValue>[],
  tokenCount: number,
): BigInt64Array {
  const lengths = pastKvs.map((pastKv) => getWhisperDecoderCacheLength(pastKv));
  const cacheLength = lengths[0];
  if (cacheLength === undefined) {
    throw new Error('Whisper batched cache_position requires at least one active beam.');
  }
  if (lengths.some((length) => length !== cacheLength)) {
    throw new Error('Cannot batch Whisper beams with different decoder cache_position lengths.');
  }
  return createWhisperCachePosition(cacheLength, tokenCount);
}

export function createWhisperBatchedStepFeedTensors(
  ort: Pick<OrtModuleLike, 'Tensor'>,
  tokenIds: readonly number[],
  pastKvs: readonly Record<string, WhisperKvCacheValue>[],
  includeCachePosition: boolean,
): Record<string, unknown> {
  const cachePosition = includeCachePosition
    ? createWhisperBatchedCachePosition(pastKvs, 1)
    : undefined;
  const feeds: Record<string, unknown> = {
    input_ids: new ort.Tensor('int64', createWhisperInt64TokenIds(tokenIds), [tokenIds.length, 1]),
  };
  if (cachePosition) {
    feeds.cache_position = new ort.Tensor('int64', cachePosition, [cachePosition.length]);
  }
  return feeds;
}

/**
 * Return true when a batched decoder can feed one immutable encoder-KV slice
 * for all active beams. Identity, rather than value equality, is deliberate:
 * the core decoder shares read-only cache entries, while distinct buffers must
 * remain fully materialized to avoid changing model semantics.
 */
export function canShareWhisperEncoderKvAcrossBatch(
  values: readonly WhisperKvCacheValue[],
): boolean {
  if (values.length < 2) return false;
  const first = normalizeWhisperKvCacheValue(values[0]!);
  return first.dims?.[0] === 1 && values.every((value) => {
    const normalized = normalizeWhisperKvCacheValue(value);
    return normalized.data === first.data && normalized.dims?.[0] === 1;
  });
}

/**
 * Gate the encoder-KV broadcast optimization on the decoder-step graph's
 * token-axis contract. The older public export has a dynamic token axis and
 * its attention graph correctly broadcasts a batch-one encoder cache. The
 * newer fixed-[B,1] export family can accept the same tensor shape but its
 * reshape path treats the singleton batch as real data, producing wrong
 * logits instead of an error. Unknown sessions stay on the materialized
 * path; correctness is more important than a speculative copy elision.
 */
export function supportsWhisperEncoderKvBroadcast(
  session: Pick<OrtSessionLike, 'inputMetadata'>,
): boolean {
  const inputIds = session.inputMetadata?.find((entry) => entry.name === 'input_ids');
  const tokenAxis = inputIds?.shape?.[1];
  return typeof tokenAxis === 'string' && tokenAxis.length > 0;
}

/**
 * Scalar beam decoding can reuse an encoder-KV tensor for the lifetime of one
 * transcription. The decoder contract treats encoder KV as immutable; only
 * decoder self-attention KV advances. Keep shape and dtype checks here so a
 * backend that changes the cache contract falls back to the defensive clone.
 */
export function canReuseWhisperEncoderKvTensor(
  cached: Pick<WhisperEncoderKvTensorCacheEntry, 'sourceData' | 'stableData' | 'dims' | 'type'> | undefined,
  data: ArrayBufferView | undefined,
  dims: readonly number[] | undefined,
  type: 'float32' | 'float16',
): boolean {
  return Boolean(
    cached
    && data
    && (cached.sourceData === data || cached.stableData === data)
    && cached.type === type
    && dims
    && cached.dims.length === dims.length
    && cached.dims.every((dim, index) => dim === dims[index]),
  );
}
