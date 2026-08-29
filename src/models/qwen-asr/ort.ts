import {
  honorAbortAfterCreate,
  withNativeAbortSignalOption,
  withOrtCreateAbort,
} from '../../io/abort.js';
import {
  importNodeModule,
  importNodePackage,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../../io/node-compat.js';
import { resolveOrtExternalDataMounts } from '../../io/ort-external-data.js';
import type {
  Qwen3AsrArtifactSource,
  Qwen3AsrDirectArtifacts,
  Qwen3AsrHuggingFaceSource,
  QwenCacheOutputLocation,
  QwenExecutionBackend,
} from './types.js';

interface OrtEnv {
  readonly wasm: {
    wasmPaths?: string;
    numThreads?: number;
    simd?: boolean;
    proxy?: boolean;
  };
  webgpu?: { profiling?: unknown };
  versions?: { common?: string };
}

export interface QwenOrtTensorLike<TData extends ArrayBufferView = ArrayBufferView> {
  readonly data: TData;
  readonly dims: readonly number[];
  readonly type?: string;
  readonly location?: string;
  readonly gpuBuffer?: unknown;
  getData?(releaseData?: boolean): Promise<TData>;
  dispose?(): void;
}

export interface QwenOrtSessionLike {
  readonly inputNames?: readonly string[];
  readonly outputNames?: readonly string[];
  readonly inputMetadata?: readonly { readonly name?: string; readonly type?: string }[];
  run(
    feeds: Record<string, unknown>,
    fetchesOrOptions?: unknown,
    options?: Record<string, unknown>,
  ): Promise<Record<string, QwenOrtTensorLike>>;
  release?(): void;
}

export interface QwenOrtModuleLike {
  readonly env: OrtEnv;
  readonly Tensor: new <TData extends ArrayBufferView>(
    type: 'float16' | 'float32' | 'int32' | 'int64' | 'bool',
    data: TData,
    dims: readonly number[],
  ) => QwenOrtTensorLike<TData>;
  readonly InferenceSession: {
    create(url: string, options?: Record<string, unknown>): Promise<QwenOrtSessionLike>;
  };
}

export interface ResolvedQwen3AsrArtifacts {
  readonly artifacts: Qwen3AsrDirectArtifacts;
  readonly ortBackend: QwenExecutionBackend;
  readonly encoderBackendForOrt: QwenExecutionBackend;
  readonly decoderBackendForOrt: QwenExecutionBackend;
  readonly cacheOutputLocation: QwenCacheOutputLocation;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

const DEFAULT_REPO_REVISION = 'main';
const DEFAULT_ENCODER_PATH = 'onnx/audio_encoder_fp16.onnx';
const DEFAULT_DECODER_PATH = 'onnx/decoder_with_past_fp16.onnx';
const DEFAULT_TOKENIZER_PATH = 'processor/tokenizer.json';
const DEFAULT_ENCODER_DATA_PATH = 'audio_encoder_fp16.onnx_data';
const DEFAULT_DECODER_DATA_PATH = 'decoder_with_past_fp16.onnx_data';

function buildResolveUrl(repoId: string, revision: string, filename: string): string {
  const repo = repoId
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
  const rev = encodeURIComponent(revision);
  const path = filename
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
  return `https://huggingface.co/${repo}/resolve/${rev}/${path}`;
}

function normalizeBackend(backendId: string): QwenExecutionBackend {
  return backendId.toLowerCase().startsWith('webgpu') ? 'webgpu' : 'wasm';
}

function resolveBackend(
  requested: QwenExecutionBackend | undefined,
  fallback: QwenExecutionBackend,
): QwenExecutionBackend {
  return requested ?? fallback;
}

function resolveHuggingFaceArtifacts(
  source: Qwen3AsrHuggingFaceSource,
  backendId: string,
): ResolvedQwen3AsrArtifacts {
  const revision = source.revision ?? DEFAULT_REPO_REVISION;
  const fallback = normalizeBackend(backendId);
  const encoderBackendForOrt = resolveBackend(source.encoderBackend, fallback);
  const decoderBackendForOrt = resolveBackend(source.decoderBackend, fallback);
  return {
    artifacts: {
      encoderUrl: buildResolveUrl(
        source.repoId,
        revision,
        source.encoderPath ?? DEFAULT_ENCODER_PATH,
      ),
      decoderUrl: buildResolveUrl(
        source.repoId,
        revision,
        source.decoderPath ?? DEFAULT_DECODER_PATH,
      ),
      tokenizerUrl: buildResolveUrl(
        source.repoId,
        revision,
        source.tokenizerPath ?? DEFAULT_TOKENIZER_PATH,
      ),
      encoderDataUrl: buildResolveUrl(
        source.repoId,
        revision,
        `onnx/${source.encoderDataPath ?? DEFAULT_ENCODER_DATA_PATH}`,
      ),
      decoderDataUrl: buildResolveUrl(
        source.repoId,
        revision,
        `onnx/${source.decoderDataPath ?? DEFAULT_DECODER_DATA_PATH}`,
      ),
      encoderDataPath: source.encoderDataPath ?? DEFAULT_ENCODER_DATA_PATH,
      decoderDataPath: source.decoderDataPath ?? DEFAULT_DECODER_DATA_PATH,
    },
    ortBackend:
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    cacheOutputLocation: source.cacheOutputLocation ?? 'gpu-buffer',
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function resolveDirectArtifacts(
  source: Extract<Qwen3AsrArtifactSource, { kind: 'direct' }>,
  backendId: string,
): ResolvedQwen3AsrArtifacts {
  const fallback = normalizeBackend(backendId);
  const encoderBackendForOrt = resolveBackend(source.encoderBackend, fallback);
  const decoderBackendForOrt = resolveBackend(source.decoderBackend, fallback);
  return {
    artifacts: source.artifacts,
    ortBackend:
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    cacheOutputLocation: source.cacheOutputLocation ?? 'gpu-buffer',
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

export function resolveQwen3AsrArtifacts(
  source: Qwen3AsrArtifactSource,
  backendId: string,
): ResolvedQwen3AsrArtifacts {
  return source.kind === 'huggingface'
    ? resolveHuggingFaceArtifacts(source, backendId)
    : resolveDirectArtifacts(source, backendId);
}

export async function initQwenOrt(
  backendId: QwenExecutionBackend,
  options: {
    readonly wasmPaths?: string;
    readonly cpuThreads?: number;
    readonly enableProfiling?: boolean;
    readonly signal?: { readonly aborted: boolean } | null;
  } = {},
): Promise<QwenOrtModuleLike> {
  if (backendId === 'webgpu' && isNodeLikeRuntime()) {
    // onnxruntime-web resolves WebGPU through navigator.gpu, which Node does
    // not provide; the native package ships its own wgpu adapter. Fall back to
    // the web build when the native package is unavailable so callers classify
    // the backend as before.
    try {
      const imported = importNodePackage<
        QwenOrtModuleLike & {
          readonly default?: QwenOrtModuleLike;
        }
      >('onnxruntime-node');
      return withOrtCreateAbort(imported.default ?? imported, options.signal);
    } catch {
      // fall through to onnxruntime-web.
    }
  }
  const imported = (await (backendId === 'webgpu'
    ? import('onnxruntime-web/webgpu')
    : import('onnxruntime-web'))) as unknown as QwenOrtModuleLike & {
    readonly default?: QwenOrtModuleLike;
  };
  const ort = imported.default ?? imported;
  if (typeof SharedArrayBuffer !== 'undefined') {
    ort.env.wasm.numThreads =
      options.cpuThreads ??
      (typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
        ? navigator.hardwareConcurrency
        : 4);
    ort.env.wasm.simd = true;
  } else {
    ort.env.wasm.numThreads = 1;
  }
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths =
    options.wasmPaths ??
    (isNodeLikeRuntime()
      ? await resolveNodePackageSubpathUrl('onnxruntime-web', 'dist')
      : '/node_modules/onnxruntime-web/dist/');
  if (backendId === 'webgpu' && options.enableProfiling) {
    ort.env.webgpu ??= {};
    ort.env.webgpu.profiling = { mode: 'default' };
  }
  return withOrtCreateAbort(ort, options.signal);
}

/** Fire-and-forget ORT session teardown. onnxruntime-web `release()` frees GPU/WASM heaps. */
export function releaseQwenOrtSession(session: QwenOrtSessionLike | undefined): void {
  void session?.release?.();
}

export async function createQwenOrtSession(
  ort: QwenOrtModuleLike,
  url: string,
  options: {
    readonly backendId: QwenExecutionBackend;
    readonly enableProfiling?: boolean;
    readonly externalDataUrl?: string;
    readonly externalDataPath?: string;
    readonly preferredOutputLocation?:
      | QwenCacheOutputLocation
      | Record<string, QwenCacheOutputLocation>;
    readonly lowMemory?: boolean;
    readonly signal?: { readonly aborted: boolean } | null;
  },
): Promise<QwenOrtSessionLike> {
  let modelUrl = url;
  let externalDataUrl = options.externalDataUrl;
  const executionProviders =
    options.backendId === 'webgpu'
      ? isNodeLikeRuntime()
        ? ['webgpu']
        : [{ name: 'webgpu', deviceType: 'gpu', powerPreference: 'high-performance' }]
      : ['wasm'];
  const lowMemory = options.lowMemory === true;
  const sessionOptions: Record<string, unknown> = {
    executionProviders,
    graphOptimizationLevel: 'all',
    executionMode: lowMemory ? 'sequential' : 'parallel',
    enableCpuMemArena: !lowMemory,
    enableMemPattern: !lowMemory,
    enableProfiling: options.enableProfiling ?? false,
  };
  if (options.preferredOutputLocation) {
    // Native ORT in Node does not implement gpu-buffer output locations; keep
    // every output on the CPU so parity runs stay on the copy-to-JS path.
    sessionOptions.preferredOutputLocation = isNodeLikeRuntime()
      ? typeof options.preferredOutputLocation === 'string'
        ? 'cpu'
        : Object.fromEntries(
            Object.entries(options.preferredOutputLocation).map(([name]) => [name, 'cpu']),
          )
      : options.preferredOutputLocation;
  }
  if (isNodeLikeRuntime()) {
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    if (/^file:/i.test(modelUrl)) modelUrl = fileURLToPath(modelUrl);
    if (externalDataUrl && /^file:/i.test(externalDataUrl))
      externalDataUrl = fileURLToPath(externalDataUrl);
  }
  if (externalDataUrl && options.externalDataPath) {
    const externalData = await resolveOrtExternalDataMounts({
      backendId: options.backendId,
      sessionModelUrl: modelUrl,
      externalDataUrl,
      externalDataPath: options.externalDataPath,
    });
    if (externalData) {
      sessionOptions.externalData = externalData;
    }
  }
  const createOptions =
    withNativeAbortSignalOption(sessionOptions, options.signal) ?? sessionOptions;
  return honorAbortAfterCreate(
    () => ort.InferenceSession.create(modelUrl, createOptions),
    options.signal,
    (session) => releaseQwenOrtSession(session),
  );
}
