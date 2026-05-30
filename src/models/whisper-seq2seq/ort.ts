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
  importNodeModule,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../../io/node.js';

interface OrtEnv {
  wasm: {
    wasmPaths?: string;
    numThreads?: number;
    simd?: boolean;
    proxy?: boolean;
  };
  versions?: {
    common?: string;
  };
}

export interface OrtTensorLike<TData extends ArrayBufferView = ArrayBufferView> {
  readonly data: TData;
  readonly dims: readonly number[];
  dispose?(): void;
}

export interface OrtSessionLike {
  readonly inputNames?: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
}

export interface OrtModuleLike {
  readonly env: OrtEnv;
  readonly Tensor: new <TData extends ArrayBufferView>(
    type: 'float32' | 'int32' | 'int64' | 'bool',
    data: TData,
    dims: readonly number[],
  ) => OrtTensorLike<TData>;
  readonly InferenceSession: {
    create(url: string, options?: Record<string, unknown>): Promise<OrtSessionLike>;
  };
}

export interface ExternalDataMap {
  readonly [graphName: string]: readonly { readonly dataUrl: string; readonly path: string }[];
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
  readonly isSplitGraph: boolean;
  readonly decoderInitUrl?: string;
  readonly decoderStepUrl?: string;
  readonly decoderAlignUrl?: string;
  readonly manifestUrl?: string;
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
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
    isSplitGraph: true,
    decoderInitUrl,
    decoderStepUrl,
    decoderAlignUrl: source.artifacts.decoderAlignUrl,
    manifestUrl: source.artifacts.manifestUrl,
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
  } = {},
): Promise<OrtModuleLike> {
  const imported = (await import('onnxruntime-web')) as unknown as OrtModuleLike & {
    readonly default?: OrtModuleLike;
  };
  const ort = imported.default ?? imported;

  if (!ort.env.wasm.wasmPaths) {
    ort.env.wasm.wasmPaths =
      options.wasmPaths ??
      (isNodeLikeRuntime()
        ? await resolveNodePackageSubpathUrl('onnxruntime-web', 'dist')
        : `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions?.common ?? '1.24.1'}/dist/`);
  } else if (options.wasmPaths) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  }

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

  if (
    normalizeWhisperWeightBackend(backendId) === 'webgpu' &&
    typeof navigator !== 'undefined' &&
    !('gpu' in navigator)
  ) {
    return ort;
  }

  return ort;
}

export async function createWhisperOrtSession(
  ort: OrtModuleLike,
  url: string,
  options: {
    readonly backendId: string;
    readonly enableProfiling?: boolean;
    readonly externalDataUrl?: string;
    readonly externalDataPath?: string;
  },
): Promise<OrtSessionLike> {
  let modelUrl = url;
  let externalDataUrl = options.externalDataUrl;
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

  if (isNodeLikeRuntime()) {
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    if (/^file:/i.test(modelUrl)) {
      modelUrl = fileURLToPath(modelUrl);
    }
    // Also handle bare file paths — ONNX Runtime Node.js accepts them natively,
    // but ensure they're absolute for consistency
    if (typeof modelUrl === 'string' && modelUrl.startsWith('/')) {
      // Already a file path, no conversion needed
    }
    if (externalDataUrl && /^file:/i.test(externalDataUrl)) {
      externalDataUrl = fileURLToPath(externalDataUrl);
    }
  }

  if (externalDataUrl && options.externalDataPath) {
    sessionOptions.externalData = [
      {
        data: externalDataUrl,
        path: options.externalDataPath,
      },
    ];
  }

  return ort.InferenceSession.create(modelUrl, sessionOptions);
}
