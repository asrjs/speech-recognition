import type {
  NemoTdtArtifactSource,
  NemoTdtDirectArtifacts,
  NemoTdtExecutionBackend,
  NemoTdtHuggingFaceSource,
  NemoTdtOutputLocation,
  NemoTdtPreprocessorBackend,
  NemoTdtQuantization,
  NemoTdtWebGpuOptions,
} from './types.js';
import { getDefaultNemoTdtWeightSetup, normalizeNemoTdtWeightBackend } from './weights.js';
import { loadNativeCpuOrt } from '../../inference/backends/native-cpu/ort.js';
import { CapabilityMismatchError } from '../../runtime/errors.js';
import {
  honorAbortAfterCreate,
  withNativeAbortSignalOption,
  withOrtCreateAbort,
  throwIfAssetAborted,
} from '../../io/abort.js';
import {
  importNodeModule,
  isNodeLikeRuntime,
} from '../../io/node-compat.js';
import {
  configureUnifiedOrtEnv,
  releaseUnifiedOrtSession,
} from '../../inference/ort/index.js';

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

export type OrtOutputLocation = NemoTdtOutputLocation;

export interface OrtSessionLike {
  readonly inputNames?: readonly string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
  release?(): void | Promise<void>;
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
  readonly decoderStateOutputLocation?: NemoTdtOutputLocation;
  readonly webgpuOptions?: NemoTdtWebGpuOptions;
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
  return role === 'decoder' && fallback !== 'native-cpu' ? 'wasm' : fallback;
}

function resolveHuggingFaceArtifacts(
  source: NemoTdtHuggingFaceSource,
  backendId: string,
): ResolvedNemoTdtArtifacts {
  const revision = source.revision ?? 'main';
  const fallbackBackend =
    backendId === 'native-cpu' ? 'native-cpu' : normalizeNemoTdtWeightBackend(backendId);
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
    fallbackBackend === 'native-cpu'
      ? 'native-cpu'
      : encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu'
        ? 'webgpu'
        : 'wasm';
  const encoderQuant = resolveQuantization(source.encoderQuant, encoderBackendForOrt, 'encoder');
  const decoderQuant = resolveQuantization(source.decoderQuant, decoderBackendForOrt, 'decoder');
  const encoderFilename = getQuantizedFilename('encoder-model', encoderQuant);
  const decoderFilename = getQuantizedFilename('decoder_joint-model', decoderQuant);
  const preprocessorName = source.preprocessorName ?? 'nemo128';
  const preprocessorBackend =
    source.preprocessorBackend ?? (fallbackBackend === 'native-cpu' ? 'js' : 'onnx');

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
    decoderStateOutputLocation: source.decoderStateOutputLocation,
    webgpuOptions: source.webgpuOptions,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

function resolveDirectArtifacts(
  source: Extract<NemoTdtArtifactSource, { kind: 'direct' }>,
  backendId: string,
): ResolvedNemoTdtArtifacts {
  const fallbackBackend =
    backendId === 'native-cpu' ? 'native-cpu' : normalizeNemoTdtWeightBackend(backendId);
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
    preprocessorBackend:
      source.preprocessorBackend ?? (fallbackBackend === 'native-cpu' ? 'js' : 'onnx'),
    warnings: [],
    ortBackend:
      fallbackBackend === 'native-cpu'
        ? 'native-cpu'
        : encoderBackendForOrt === 'webgpu' || decoderBackendForOrt === 'webgpu'
          ? 'webgpu'
          : 'wasm',
    encoderBackendForOrt,
    decoderBackendForOrt,
    decoderStateOutputLocation: source.decoderStateOutputLocation,
    webgpuOptions: source.webgpuOptions,
    wasmPaths: source.wasmPaths,
    cpuThreads: source.cpuThreads,
    enableProfiling: source.enableProfiling,
  };
}

export function resolveNemoTdtArtifacts(
  source: NemoTdtArtifactSource,
  backendId: string,
): ResolvedNemoTdtArtifacts {
  const native = backendId === 'native-cpu';
  const components = [source.encoderBackend, source.decoderBackend];
  if (components.some((component) => component && (component === 'native-cpu') !== native)) {
    throw new CapabilityMismatchError(
      'native-cpu requires both graph components to use native-cpu; mixed native/Web engines are not supported.',
      { backendId, encoderBackend: source.encoderBackend, decoderBackend: source.decoderBackend },
    );
  }
  if (
    native &&
    (source.webgpuOptions || source.decoderStateOutputLocation === 'gpu-buffer' || source.wasmPaths)
  ) {
    throw new CapabilityMismatchError(
      'native-cpu cannot honor WebGPU output/tuning or WASM path options.',
    );
  }
  return source.kind === 'huggingface'
    ? resolveHuggingFaceArtifacts(source, backendId)
    : resolveDirectArtifacts(source, backendId);
}

export async function initOrt(
  backendId: string,
  options: {
    readonly wasmPaths?: string;
    readonly cpuThreads?: number;
    readonly signal?: { readonly aborted: boolean } | null;
  } = {},
): Promise<OrtModuleLike> {
  throwIfAssetAborted(options.signal, 'session-create');
  if (backendId === 'native-cpu') {
    return withOrtCreateAbort(loadNativeCpuOrt() as unknown as OrtModuleLike, options.signal);
  }
  const imported = (await import('onnxruntime-web')) as unknown as OrtModuleLike & {
    readonly default?: OrtModuleLike;
  };
  const ort = imported.default ?? imported;

  configureUnifiedOrtEnv(ort as any, {
    wasmPaths: options.wasmPaths,
    cpuThreads: options.cpuThreads,
    signal: options.signal,
  });

  if (
    normalizeNemoTdtWeightBackend(backendId) === 'webgpu' &&
    typeof navigator !== 'undefined' &&
    !('gpu' in navigator)
  ) {
    return withOrtCreateAbort(ort, options.signal);
  }

  return withOrtCreateAbort(ort, options.signal);
}

export async function createOrtSession(
  ort: OrtModuleLike,
  url: string,
  options: {
    readonly backendId: string;
    readonly cpuThreads?: number;
    readonly enableProfiling?: boolean;
    readonly externalDataUrl?: string;
    readonly externalDataPath?: string;
    readonly preferredOutputLocation?:
      | OrtOutputLocation
      | Readonly<Record<string, OrtOutputLocation>>;
    readonly webgpuOptions?: NemoTdtWebGpuOptions;
    readonly signal?: { readonly aborted: boolean } | null;
  },
): Promise<OrtSessionLike> {
  let modelUrl = url;
  let externalDataUrl = options.externalDataUrl;
  const native = options.backendId === 'native-cpu';
  if (
    native &&
    options.cpuThreads !== undefined &&
    (!Number.isInteger(options.cpuThreads) || options.cpuThreads < 1)
  ) {
    throw new Error('Native CPU thread count must be a positive integer.');
  }
  if (
    native &&
    (options.webgpuOptions ||
      (typeof options.preferredOutputLocation === 'string'
        ? options.preferredOutputLocation === 'gpu-buffer'
        : Object.values(options.preferredOutputLocation ?? {}).includes('gpu-buffer')))
  ) {
    throw new CapabilityMismatchError(
      'native-cpu does not support WebGPU output placement or tuning.',
    );
  }
  const executionProviders = native
    ? ['cpu']
    : options.backendId.startsWith('webgpu')
      ? [
          {
            name: 'webgpu',
            deviceType: 'gpu',
            powerPreference: 'high-performance',
            ...(options.webgpuOptions ?? {}),
          },
        ]
      : ['wasm'];

  const sessionOptions: Record<string, unknown> = {
    executionProviders,
    graphOptimizationLevel: 'all',
    executionMode: native ? 'sequential' : 'parallel',
    enableCpuMemArena: true,
    enableMemPattern: true,
    enableProfiling: options.enableProfiling ?? false,
    ...(native ? { intraOpNumThreads: options.cpuThreads ?? 4, interOpNumThreads: 1 } : {}),
  };

  if (options.preferredOutputLocation) {
    // The browser WebGPU EP supports gpu-buffer outputs. Node's onnxruntime-
    // web path does not, so keep diagnostic options CPU-backed in local runs.
    sessionOptions.preferredOutputLocation = isNodeLikeRuntime()
      ? typeof options.preferredOutputLocation === 'string'
        ? 'cpu'
        : Object.fromEntries(
            Object.keys(options.preferredOutputLocation).map((name) => [name, 'cpu']),
          )
      : options.preferredOutputLocation;
  }

  if (isNodeLikeRuntime()) {
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    if (/^file:/i.test(modelUrl)) {
      modelUrl = fileURLToPath(modelUrl);
    }
    if (externalDataUrl && /^file:/i.test(externalDataUrl)) {
      externalDataUrl = fileURLToPath(externalDataUrl);
    }
  }

  if (native) {
    const nodePath = await importNodeModule<typeof import('node:path')>('node:path');
    if (!nodePath.isAbsolute(modelUrl)) {
      throw new Error('Native CPU inference requires a local absolute model path or file URL.');
    }
    if (externalDataUrl) {
      const graphRelativePath = options.externalDataPath;
      const expectedPath = graphRelativePath
        ? nodePath.resolve(nodePath.dirname(modelUrl), graphRelativePath)
        : undefined;
      if (!expectedPath || nodePath.resolve(externalDataUrl) !== expectedPath) {
        throw new Error('Native external data must be colocated at its graph-relative location.');
      }
      const fs = await importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');
      if (!(await fs.stat(expectedPath)).isFile()) {
        throw new Error('Native external data locator is not a regular file.');
      }
    }
    // Native ORT resolves graph-referenced sidecars from the model's directory.
    // Do not read multi-GB data into JS or pass ORT Web's externalData mounts.
  } else if (externalDataUrl && options.externalDataPath) {
    sessionOptions.externalData = [
      {
        data: externalDataUrl,
        path: options.externalDataPath,
      },
    ];
  }

  const createOptions =
    withNativeAbortSignalOption(sessionOptions, options.signal) ?? sessionOptions;
  return honorAbortAfterCreate(
    () => ort.InferenceSession.create(modelUrl, createOptions),
    options.signal,
    (session) => session.release?.(),
  );
}

/** Return ORT teardown so an owning lifecycle can await native/GPU/WASM release. */
export function releaseOrtSession(session: OrtSessionLike | undefined): void | Promise<void> {
  return releaseUnifiedOrtSession(session);
}

/** Drop `session.run()` output tensors after their JS data has been copied. */
export function disposeOrtOutputs(
  outputs: Record<string, { dispose?: () => void } | undefined> | undefined,
): void {
  if (!outputs) return;
  for (const tensor of Object.values(outputs)) tensor?.dispose?.();
}
