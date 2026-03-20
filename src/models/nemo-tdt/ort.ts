import type {
  NemoTdtArtifactSource,
  NemoTdtDirectArtifacts,
  NemoTdtExecutionBackend,
  NemoTdtHuggingFaceSource,
  NemoTdtPreprocessorBackend,
  NemoTdtQuantization,
} from './types.js';
import { getDefaultNemoTdtWeightSetup, normalizeNemoTdtWeightBackend } from './weights.js';
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
    type: 'float32' | 'int32' | 'int64',
    data: TData,
    dims: readonly number[],
  ) => OrtTensorLike<TData>;
  readonly InferenceSession: {
    create(url: string, options?: Record<string, unknown>): Promise<OrtSessionLike>;
  };
}

export interface ResolvedNemoTdtArtifacts {
  readonly artifacts: NemoTdtDirectArtifacts;
  readonly preprocessorBackend: NemoTdtPreprocessorBackend;
  readonly warnings: readonly { readonly code: string; readonly message: string }[];
  readonly ortBackend: NemoTdtExecutionBackend;
  readonly encoderBackendForOrt: NemoTdtExecutionBackend;
  readonly decoderBackendForOrt: NemoTdtExecutionBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

const QUANTIZATION_SUFFIX: Record<NemoTdtQuantization, string> = {
  int8: '.int8.onnx',
  fp16: '.fp16.onnx',
  fp32: '.onnx',
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

function getQuantizedFilename(baseName: string, quantization: NemoTdtQuantization): string {
  return `${baseName}${QUANTIZATION_SUFFIX[quantization]}`;
}

function resolveQuantization(
  requested: NemoTdtQuantization | undefined,
  backendForOrt: NemoTdtExecutionBackend,
  role: 'encoder' | 'decoder',
): NemoTdtQuantization {
  if (requested) {
    return requested;
  }
  const setup = getDefaultNemoTdtWeightSetup(backendForOrt);
  return role === 'encoder' ? setup.encoderDefault : setup.decoderDefault;
}

function resolveComponentBackend(
  requested: NemoTdtExecutionBackend | undefined,
  fallback: NemoTdtExecutionBackend,
  role: 'encoder' | 'decoder',
): NemoTdtExecutionBackend {
  if (requested) {
    return requested;
  }
  return role === 'decoder' ? 'wasm' : fallback;
}

function resolveHuggingFaceArtifacts(
  source: NemoTdtHuggingFaceSource,
  backendId: string,
): ResolvedNemoTdtArtifacts {
  const revision = source.revision ?? 'main';
  const fallbackBackend = normalizeNemoTdtWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(
    source.encoderBackend,
    fallbackBackend,
    'encoder',
  );
  const decoderBackendForOrt = resolveComponentBackend(
    source.decoderBackend,
    fallbackBackend,
    'decoder',
  );
  const ortBackend =
    encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm';
  const encoderQuant = resolveQuantization(source.encoderQuant, encoderBackendForOrt, 'encoder');
  const decoderQuant = resolveQuantization(source.decoderQuant, decoderBackendForOrt, 'decoder');
  const encoderFilename = getQuantizedFilename('encoder-model', encoderQuant);
  const decoderFilename = getQuantizedFilename('decoder_joint-model', decoderQuant);
  const preprocessorName = source.preprocessorName ?? 'nemo128';
  const preprocessorBackend = source.preprocessorBackend ?? 'onnx';

  return {
    artifacts: {
      encoderUrl: buildResolveUrl(source.repoId, revision, encoderFilename),
      decoderUrl: buildResolveUrl(source.repoId, revision, decoderFilename),
      tokenizerUrl: buildResolveUrl(source.repoId, revision, 'vocab.txt'),
      preprocessorUrl:
        preprocessorBackend === 'onnx'
          ? buildResolveUrl(source.repoId, revision, `${preprocessorName}.onnx`)
          : undefined,
      encoderFilename,
      decoderFilename,
    },
    preprocessorBackend,
    warnings: [],
    ortBackend,
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function resolveDirectArtifacts(
  source: Extract<NemoTdtArtifactSource, { kind: 'direct' }>,
  backendId: string,
): ResolvedNemoTdtArtifacts {
  const fallbackBackend = normalizeNemoTdtWeightBackend(backendId);
  const encoderBackendForOrt = resolveComponentBackend(
    source.encoderBackend,
    fallbackBackend,
    'encoder',
  );
  const decoderBackendForOrt = resolveComponentBackend(
    source.decoderBackend,
    fallbackBackend,
    'decoder',
  );
  return {
    artifacts: source.artifacts,
    preprocessorBackend: source.preprocessorBackend ?? 'onnx',
    warnings: [],
    ortBackend:
      encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu' ? 'webgpu' : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

export function resolveNemoTdtArtifacts(
  source: NemoTdtArtifactSource,
  backendId: string,
): ResolvedNemoTdtArtifacts {
  return source.kind === 'huggingface'
    ? resolveHuggingFaceArtifacts(source, backendId)
    : resolveDirectArtifacts(source, backendId);
}

export async function initOrt(
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
    normalizeNemoTdtWeightBackend(backendId) === 'webgpu' &&
    typeof navigator !== 'undefined' &&
    !('gpu' in navigator)
  ) {
    return ort;
  }

  return ort;
}

export async function createOrtSession(
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
