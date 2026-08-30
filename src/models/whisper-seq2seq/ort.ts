import type {
  WhisperArtifactSource,
  WhisperDirectArtifactSource,
  WhisperDirectArtifacts,
  WhisperExecutionBackend,
  WhisperHuggingFaceSource,
  WhisperQuantization,
  WhisperSplitGraphArtifactSource,
} from './types.js';
import {
  honorAbortAfterCreate,
  withNativeAbortSignalOption,
  withOrtCreateAbort,
} from '../../io/abort.js';
import {
  importNodeModule,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../../io/node-compat.js';

interface OrtEnv {
  wasm: {
    wasmPaths?: string;
    numThreads?: number;
    simd?: boolean;
    proxy?: boolean;
  };
  webgpu?: {
    profiling?: unknown;
    device?: unknown;
  };
  versions?: {
    common?: string;
  };
}

export interface OrtTensorLike<TData extends ArrayBufferView = ArrayBufferView> {
  readonly data: TData;
  readonly dims: readonly number[];
  readonly type?: string;
  readonly location?: string;
  readonly gpuBuffer?: unknown;
  getData?(releaseData?: boolean): Promise<TData>;
  dispose?(): void;
}

export interface OrtSessionLike {
  readonly inputNames?: readonly string[];
  readonly outputNames?: readonly string[];
  readonly inputMetadata?: readonly {
    readonly name?: string;
    readonly type?: string;
    readonly shape?: readonly (number | string)[];
  }[];
  run(
    feeds: Record<string, unknown>,
    fetchesOrOptions?: unknown,
    options?: Record<string, unknown>,
  ): Promise<Record<string, OrtTensorLike>>;
  release?(): void | Promise<void>;
}

export interface OrtTensorConstructor {
  new <TData extends ArrayBufferView>(
    type: 'float32' | 'float16' | 'int32' | 'int64' | 'bool',
    data: TData,
    dims: readonly number[],
  ): OrtTensorLike<TData>;
  fromGpuBuffer?(
    buffer: unknown,
    options: {
      dataType?: string;
      dims: readonly number[];
      download?: () => Promise<ArrayBufferView>;
      dispose?: () => void;
    },
  ): OrtTensorLike;
}

export interface OrtModuleLike {
  readonly env: OrtEnv;
  readonly Tensor: OrtTensorConstructor;
  readonly InferenceSession: {
    create(url: string, options?: Record<string, unknown>): Promise<OrtSessionLike>;
  };
}

export type OrtOutputLocation = 'cpu' | 'gpu-buffer';
export type OrtPreferredOutputLocation =
  | OrtOutputLocation
  | Record<string, OrtOutputLocation>;

export interface WhisperExternalDataFile {
  readonly dataUrl: string;
  readonly path: string;
}

export interface ExternalDataMap {
  readonly [graphName: string]: readonly WhisperExternalDataFile[];
}

export interface ResolvedWhisperAlignmentReference {
  readonly encoderUrl: string;
  readonly decoderAlignUrl: string;
  readonly manifestUrl?: string;
  readonly backendForOrt: string;
  readonly externalData?: Partial<Pick<ExternalDataMap, 'encoder' | 'decoder_align'>>;
}

export interface ResolvedWhisperArtifacts {
  readonly artifacts: WhisperDirectArtifacts;
  readonly warnings: readonly { readonly code: string; readonly message: string }[];
  readonly ortBackend: WhisperExecutionBackend;
  readonly encoderBackendForOrt: string;
  readonly decoderBackendForOrt: string;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly experimentalWebGpuEncoderGraphCapture?: boolean;
  readonly experimentalGpuKvCache?: boolean;
  readonly experimentalGpuKvBeam?: boolean;
  /** DIAGNOSTIC: Force encoder output to CPU (Track A2). */
  readonly encoderOutputCpu?: boolean;
  /** DIAGNOSTIC (B2-C): Enable graph capture for decoder_step session. */
  readonly decoderGraphCapture?: boolean;
  /** DIAGNOSTIC (B2-B): freeDimensionOverrides for decoder_step session. */
  readonly decoderFreeDimensionOverrides?: Record<string, number>;
  /** DIAGNOSTIC (Edge A): Re-wrap encoder GPU output as fresh Tensor.fromGpuBuffer. */
  readonly encoderBufferRewrap?: boolean;
  /** DIAGNOSTIC (Edge B2): Force GPU flush before decoder_init. */
  readonly encoderGpuFlush?: boolean;
  /** PROFILING (encoderGpuDrain): Force GPU drain + re-wrap after encoder. */
  readonly encoderGpuDrain?: boolean;
  readonly isSplitGraph: boolean;
  readonly decoderInitUrl?: string;
  readonly decoderStepUrl?: string;
  readonly decoderAlignUrl?: string;
  readonly manifestUrl?: string;
  readonly alignmentReference?: ResolvedWhisperAlignmentReference;
  /** Per-graph external data mappings: graph name → [{ dataUrl, path }].
   *  For Node.js, ORT loads co-located .data files automatically. For browser,
   *  the executor passes these to InferenceSession.create() as externalData. */
  readonly externalData?: ExternalDataMap;
}

const QUANTIZATION_SUFFIX: Record<WhisperQuantization, string> = {
  fp32: '_model.onnx',
  fp16: '_model_fp16.onnx',
  int8: '_model_int8.onnx',
  q4: '_model_q4.onnx',
  uint8: '_model_uint8.onnx',
};

function buildResolveUrl(repoId: string, revision: string, filename: string): string {
  const encodedRepo = repoId
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  const encodedRevision = encodeURIComponent(revision);
  const encodedFilename = filename
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');

  return `https://huggingface.co/${encodedRepo}/resolve/${encodedRevision}/${encodedFilename}`;
}

function getQuantizedFilename(baseName: string, quantization: WhisperQuantization): string {
  return `${baseName}${QUANTIZATION_SUFFIX[quantization]}`;
}

function resolveQuantization(
  requested: WhisperQuantization | undefined,
  backendForOrt: WhisperExecutionBackend,
  role: 'encoder' | 'decoder',
): WhisperQuantization {
  if (requested) {
    return requested;
  }
  if (backendForOrt === 'webgpu') {
    return role === 'encoder' ? 'fp16' : 'int8';
  }
  return 'int8';
}

function resolveComponentBackend(
  requested: WhisperExecutionBackend | undefined,
  fallback: WhisperExecutionBackend,
  role: 'encoder' | 'decoder',
): WhisperExecutionBackend {
  if (requested) {
    return requested;
  }
  return role === 'decoder' ? 'wasm' : fallback;
}

function normalizeWhisperWeightBackend(backendId: string): WhisperExecutionBackend {
  const normalized = String(backendId || '').toLowerCase();
  if (normalized.startsWith('webgpu')) {
    return 'webgpu';
  }
  return 'wasm';
}

function resolveHuggingFaceArtifacts(
  source: WhisperHuggingFaceSource,
  backendId: string,
): ResolvedWhisperArtifacts {
  const revision = source.revision ?? 'main';
  const fallbackBackend = normalizeWhisperWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(source.encoderBackend, fallbackBackend, 'encoder');
  const decoderBackendForOrt = resolveComponentBackend(source.decoderBackend, fallbackBackend, 'decoder');
  const ortBackend =
    encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm';
  const encoderQuant = resolveQuantization(source.encoderQuant, encoderBackendForOrt, 'encoder');
  const decoderQuant = resolveQuantization(source.decoderQuant, decoderBackendForOrt, 'decoder');
  const encoderFilename = getQuantizedFilename('onnx/encoder', encoderQuant);
  const decoderFilename = getQuantizedFilename('onnx/decoder_model_merged', decoderQuant);

  return {
    artifacts: {
      encoderUrl: buildResolveUrl(source.repoId, revision, encoderFilename),
      decoderUrl: buildResolveUrl(source.repoId, revision, decoderFilename),
      tokenizerUrl: buildResolveUrl(source.repoId, revision, 'tokenizer.json'),
    },
    warnings: [],
    ortBackend,
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
    experimentalWebGpuEncoderGraphCapture: source.experimentalWebGpuEncoderGraphCapture,
    experimentalGpuKvCache: source.experimentalGpuKvCache,
    experimentalGpuKvBeam: source.experimentalGpuKvBeam,
    isSplitGraph: false,
  };
}

function resolveDirectArtifacts(
  source: WhisperDirectArtifactSource,
  backendId: string,
): ResolvedWhisperArtifacts {
  const fallbackBackend = normalizeWhisperWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(source.encoderBackend, fallbackBackend, 'encoder');
  const decoderBackendForOrt = resolveComponentBackend(source.decoderBackend, fallbackBackend, 'decoder');
  return {
    artifacts: source.artifacts,
    warnings: [],
    ortBackend:
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
    experimentalWebGpuEncoderGraphCapture: source.experimentalWebGpuEncoderGraphCapture,
    experimentalGpuKvCache: source.experimentalGpuKvCache,
    experimentalGpuKvBeam: source.experimentalGpuKvBeam,
    isSplitGraph: false,
  };
}

function resolveDataUrl(graphUrl: string, externalFile: string): string {
  const normalizedFile = externalFile.replace(/^\.\//, '');
  return new URL(normalizedFile, graphUrl.replace(/[^/]*$/, '')).toString();
}

function resolveSplitGraphArtifacts(
  source: WhisperSplitGraphArtifactSource,
  backendId: string,
): ResolvedWhisperArtifacts {
  const fallbackBackend = normalizeWhisperWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(source.encoderBackend, fallbackBackend, 'encoder');
  const decoderBackendForOrt = resolveComponentBackend(source.decoderBackend, fallbackBackend, 'decoder');

  const externalDataBuild: Record<string, readonly { dataUrl: string; path: string }[]> = {};

  function addExternalData(
    graphName: 'encoder' | 'decoder_init' | 'decoder_step' | 'decoder_align',
    graphUrl: string,
  ): void {
    const entries = source.artifacts.externalDataUrls?.[graphName];
    if (!entries || entries.length === 0) return;
    externalDataBuild[graphName] = entries.map((entry) => ({
      dataUrl: resolveDataUrl(graphUrl, entry.file),
      path: entry.path,
    }));
  }

  const encoderUrl = source.artifacts.encoderUrl;
  const decoderInitUrl = source.artifacts.decoderInitUrl;
  const decoderStepUrl = source.artifacts.decoderStepUrl;

  addExternalData('encoder', encoderUrl);
  addExternalData('decoder_init', decoderInitUrl);
  addExternalData('decoder_step', decoderStepUrl);
  if (source.artifacts.decoderAlignUrl) {
    addExternalData('decoder_align', source.artifacts.decoderAlignUrl);
  }

  let alignmentReference: ResolvedWhisperAlignmentReference | undefined;
  const reference = source.artifacts.alignmentReference;
  if (reference) {
    const referenceExternalDataBuild: Partial<Pick<ExternalDataMap, 'encoder' | 'decoder_align'>> = {};
    const addReferenceExternalData = (
      graphName: 'encoder' | 'decoder_align',
      graphUrl: string,
    ): void => {
      const entries = reference.externalDataUrls?.[graphName];
      if (!entries || entries.length === 0) return;
      referenceExternalDataBuild[graphName] = entries.map((entry) => ({
        dataUrl: resolveDataUrl(graphUrl, entry.file),
        path: entry.path,
      }));
    };
    addReferenceExternalData('encoder', reference.encoderUrl);
    addReferenceExternalData('decoder_align', reference.decoderAlignUrl);

    alignmentReference = {
      encoderUrl: reference.encoderUrl,
      decoderAlignUrl: reference.decoderAlignUrl,
      manifestUrl: reference.manifestUrl ?? source.artifacts.manifestUrl,
      backendForOrt: source.alignmentReferenceBackend ?? encoderBackendForOrt,
      ...(Object.keys(referenceExternalDataBuild).length > 0
        ? { externalData: referenceExternalDataBuild }
        : {}),
    };
  }

  const externalData: ExternalDataMap | undefined = Object.keys(externalDataBuild).length > 0
    ? externalDataBuild
    : undefined;

  return {
    artifacts: {
      encoderUrl,
      decoderUrl: decoderInitUrl,
      tokenizerUrl: source.artifacts.tokenizerUrl,
    },
    warnings: [],
    ortBackend:
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' || alignmentReference?.backendForOrt === 'webgpu'
        ? 'webgpu'
        : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
    experimentalWebGpuEncoderGraphCapture: source.experimentalWebGpuEncoderGraphCapture,
    experimentalGpuKvCache: source.experimentalGpuKvCache,
    experimentalGpuKvBeam: source.experimentalGpuKvBeam,
    encoderOutputCpu: source.encoderOutputCpu,
    decoderGraphCapture: source.decoderGraphCapture,
    decoderFreeDimensionOverrides: source.decoderFreeDimensionOverrides,
    encoderBufferRewrap: source.encoderBufferRewrap,
    encoderGpuFlush: source.encoderGpuFlush,
    encoderGpuDrain: source.encoderGpuDrain,
    isSplitGraph: true,
    decoderInitUrl,
    decoderStepUrl,
    decoderAlignUrl: source.artifacts.decoderAlignUrl,
    manifestUrl: source.artifacts.manifestUrl,
    alignmentReference,
    externalData: externalData,
  };
}

export function resolveWhisperArtifacts(
  source: WhisperArtifactSource,
  backendId: string,
): ResolvedWhisperArtifacts {
  if (source.kind === 'huggingface') {
    return resolveHuggingFaceArtifacts(source, backendId);
  }
  if (source.kind === 'splitgraph') {
    return resolveSplitGraphArtifacts(source, backendId);
  }
  return resolveDirectArtifacts(source, backendId);
}

export async function initWhisperOrt(
  backendId: string,
  options: {
    readonly wasmPaths?: string;
    readonly cpuThreads?: number;
    readonly enableProfiling?: boolean;
    readonly signal?: { readonly aborted: boolean } | null;
  } = {},
): Promise<OrtModuleLike> {
  const imported = (await (
    normalizeWhisperWeightBackend(backendId) === 'webgpu'
      ? import('onnxruntime-web/webgpu')
      : import('onnxruntime-web')
  )) as unknown as OrtModuleLike & {
    readonly default?: OrtModuleLike;
  };
  const ort = imported.default ?? imported;

  if (typeof SharedArrayBuffer !== 'undefined') {
    ort.env.wasm.numThreads = options.cpuThreads ?? 1;
    ort.env.wasm.simd = true;
  } else {
    ort.env.wasm.numThreads = 1;
  }

  ort.env.wasm.proxy = false;

  // Always override ORT's default wasmPaths (which is CDN and blocked by COEP).
  // User-provided path takes precedence; otherwise use local dist/.
  if (options.wasmPaths) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  } else if (isNodeLikeRuntime()) {
    if (!ort.env.wasm.wasmPaths) {
      ort.env.wasm.wasmPaths = await resolveNodePackageSubpathUrl('onnxruntime-web', 'dist');
    }
  } else {
    // Browser — force local path to avoid COEP-blocked CDN worker imports.
    ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
  }

  if (normalizeWhisperWeightBackend(backendId) === 'webgpu' && options.enableProfiling) {
    ort.env.webgpu ??= {};
    ort.env.webgpu.profiling = { mode: 'default' };
  }

  return withOrtCreateAbort(ort, options.signal);
}

export async function createWhisperOrtSession(
  ort: OrtModuleLike,
  url: string,
  options: {
    readonly backendId: string;
    readonly enableProfiling?: boolean;
    /** All external-data shards declared by the graph manifest. */
    readonly externalData?: readonly WhisperExternalDataFile[];
    /** @deprecated Use externalData for graphs with one or more shards. */
    readonly externalDataUrl?: string;
    /** @deprecated Use externalData for graphs with one or more shards. */
    readonly externalDataPath?: string;
    readonly preferredOutputLocation?: OrtPreferredOutputLocation;
    readonly enableGraphCapture?: boolean;
    /** DIAGNOSTIC (B2-B): Override symbolic dimensions at session creation. */
    readonly freeDimensionOverrides?: Record<string, number>;
    readonly signal?: { readonly aborted: boolean } | null;
  },
): Promise<OrtSessionLike> {
  let modelUrl = url;
  const externalDataFiles = [...(options.externalData ?? [])];
  let externalDataUrl = options.externalDataUrl;
  let externalDataPath = options.externalDataPath;
  let fileUrlToPath: ((url: string) => string) | undefined;
  const executionProviders = options.backendId.startsWith('webgpu')
    ? [
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
      ]
    : ['wasm'];

  const sessionOptions: Record<string, unknown> = {
    executionProviders,
    graphOptimizationLevel: 'all',
    executionMode: 'parallel',
    enableCpuMemArena: true,
    enableMemPattern: true,
    enableProfiling: options.enableProfiling ?? false,
  };

  if (options.preferredOutputLocation) {
    sessionOptions.preferredOutputLocation = options.preferredOutputLocation;
  }
  if (options.enableGraphCapture && options.backendId.startsWith('webgpu')) {
    sessionOptions.enableGraphCapture = true;
  }
  // DIAGNOSTIC (B2-B): freeDimensionOverrides for shape specialization
  if (options.freeDimensionOverrides) {
    sessionOptions.freeDimensionOverrides = options.freeDimensionOverrides;
  }

  if (isNodeLikeRuntime()) {
    const nodeUrl = await importNodeModule<typeof import('node:url')>('node:url');
    const { existsSync: fsExists } = await importNodeModule<typeof import('node:fs')>('node:fs');
    fileUrlToPath = nodeUrl.fileURLToPath;
    if (/^file:/i.test(modelUrl)) {
      modelUrl = fileUrlToPath(modelUrl);
    }
    // Auto-detect co-located external data file
    if (externalDataFiles.length === 0 && !externalDataUrl) {
      const dataPath = modelUrl + '.data';
      if (fsExists(dataPath)) {
        externalDataUrl = dataPath;
        // Derive the relative path for ORT
        if (!externalDataPath) {
          const basename = modelUrl.replace(/\\/g, '/').split('/').pop() ?? 'model.onnx';
          externalDataPath = basename + '.data';
        }
      }
    }
    if (externalDataUrl && /^file:/i.test(externalDataUrl)) {
      externalDataUrl = fileUrlToPath(externalDataUrl);
    }
  }

  if (externalDataUrl && externalDataPath) {
    externalDataFiles.push({ dataUrl: externalDataUrl, path: externalDataPath });
  }

  if (externalDataFiles.length > 0) {
    sessionOptions.externalData = externalDataFiles.map((entry) => ({
      data: fileUrlToPath && /^file:/i.test(entry.dataUrl)
        ? fileUrlToPath(entry.dataUrl)
        : entry.dataUrl,
      path: entry.path,
    }));
  }

  const createOptions = withNativeAbortSignalOption(sessionOptions, options.signal) ?? sessionOptions;
  return honorAbortAfterCreate(
    () => ort.InferenceSession.create(modelUrl, createOptions),
    options.signal,
    (session) => releaseOrtSession(session),
  );
}

/** Fire-and-forget ORT session teardown. onnxruntime-web `release()` frees GPU/WASM heaps. */
export function releaseOrtSession(session: OrtSessionLike | undefined): void {
  void session?.release?.();
}
