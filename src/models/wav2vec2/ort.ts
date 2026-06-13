import type { Wav2Vec2ArtifactSource, Wav2Vec2DirectArtifacts } from './types.js';
import {
  importNodeModule,
  isNodeLikeRuntime,
  resolveNodePackageSubpathUrl,
} from '../../io/node.js';

// ---------------------------------------------------------------------------
// ORT type shims (mirror lasr-ctc/ort.ts)
// ---------------------------------------------------------------------------

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
  readonly type?: string;
  dispose?(): void;
}

export interface OrtSessionLike {
  readonly inputMetadata?: Record<string, { readonly type?: string }>;
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
}

export interface OrtModuleLike {
  readonly env: OrtEnv;
  readonly Tensor: new <TData extends ArrayBufferView>(
    type: 'float32' | 'float16' | 'int32' | 'int64',
    data: TData,
    dims: readonly number[],
  ) => OrtTensorLike<TData>;
  readonly InferenceSession: {
    create(url: string, options?: Record<string, unknown>): Promise<OrtSessionLike>;
  };
}

// ---------------------------------------------------------------------------
// Resolved artifacts
// ---------------------------------------------------------------------------

export interface ResolvedWav2Vec2Artifacts {
  readonly artifacts: Wav2Vec2DirectArtifacts & {
    readonly modelDataFilename?: string;
  };
  readonly backendForOrt: 'webgpu' | 'wasm';
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalizeBackendForOrt(backendId: string): 'webgpu' | 'wasm' {
  return backendId.startsWith('webgpu') ? 'webgpu' : 'wasm';
}

function buildResolveUrl(
  repoId: string,
  revision: string,
  filename: string,
  subfolder?: string,
): string {
  const encodedRepo = repoId
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
  const encodedRevision = encodeURIComponent(revision);
  const encodedSubfolder = subfolder
    ? subfolder
        .split('/')
        .map((segment) => encodeURIComponent(segment))
        .join('/')
    : undefined;
  const encodedFilename = filename
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');

  const segments = [`https://huggingface.co/${encodedRepo}/resolve/${encodedRevision}`];
  if (encodedSubfolder) {
    segments.push(encodedSubfolder);
  }
  segments.push(encodedFilename);
  return segments.join('/');
}

function resolveDirectArtifacts(
  source: Extract<Wav2Vec2ArtifactSource, { kind: 'direct' }>,
  backendId: string,
): ResolvedWav2Vec2Artifacts {
  const modelDataFilename =
    source.artifacts.modelDataFilename ?? source.artifacts.modelDataUrl?.split('/').pop();

  return {
    artifacts: {
      modelUrl: source.artifacts.modelUrl,
      modelDataUrl: source.artifacts.modelDataUrl,
      modelDataFilename,
      tokenizerUrl: source.artifacts.tokenizerUrl,
    },
    backendForOrt: normalizeBackendForOrt(backendId),
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function resolveHuggingFaceArtifacts(
  source: Extract<Wav2Vec2ArtifactSource, { kind: 'huggingface' }>,
  backendId: string,
): ResolvedWav2Vec2Artifacts {
  const revision = source.revision ?? 'main';
  const modelFilename = source.modelFilename ?? 'model.onnx';
  const modelDataFilename = source.modelDataFilename ?? 'model.onnx.data';
  const tokenizerFilename = source.tokenizerFilename ?? 'vocab.json';

  return {
    artifacts: {
      modelUrl: buildResolveUrl(source.repoId, revision, modelFilename, source.subfolder),
      modelDataUrl: buildResolveUrl(source.repoId, revision, modelDataFilename, source.subfolder),
      modelDataFilename,
      tokenizerUrl: buildResolveUrl(source.repoId, revision, tokenizerFilename, source.subfolder),
    },
    backendForOrt: normalizeBackendForOrt(backendId),
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function resolveWav2Vec2Artifacts(
  source: Wav2Vec2ArtifactSource,
  backendId: string,
): ResolvedWav2Vec2Artifacts {
  if (source.kind === 'huggingface') {
    return resolveHuggingFaceArtifacts(source, backendId);
  }

  return resolveDirectArtifacts(source, backendId);
}

export async function initOrt(
  _backendId: string,
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

  return ort;
}

export async function createOrtSession(
  ort: OrtModuleLike,
  modelUrl: string,
  options: {
    readonly backendId?: 'webgpu' | 'wasm';
    readonly enableProfiling?: boolean;
    readonly externalDataUrl?: string;
    readonly externalDataPath?: string;
  } = {},
): Promise<OrtSessionLike> {
  const backend = options.backendId ?? 'wasm';

  const executionProviders =
    backend === 'webgpu'
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

  let sessionModelUrl = modelUrl;
  let externalDataUrl = options.externalDataUrl;

  if (isNodeLikeRuntime()) {
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    if (/^file:/i.test(sessionModelUrl)) {
      sessionModelUrl = fileURLToPath(sessionModelUrl);
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

  return ort.InferenceSession.create(sessionModelUrl, sessionOptions);
}
