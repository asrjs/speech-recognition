import { importNodeModule, isNodeLikeRuntime } from '../../io/node-compat.js';
import {
  resolveOrtExternalDataMounts,
  validateNativeOrtExternalDataFiles,
} from '../../io/ort-external-data.js';
import {
  honorAbortAfterCreate,
  throwIfAssetAborted,
  withNativeAbortSignalOption,
  withOrtCreateAbort,
} from '../../io/abort.js';
import type {
  OrtExecutionProviderOption,
  OrtModuleLike,
  OrtSessionLike,
  UnifiedOrtSessionOptions,
  UnifiedSessionCreateResult,
} from './types.js';

/**
 * Resolves standard execution provider options for ONNX Runtime sessions.
 */
export function resolveUnifiedExecutionProviders(
  backendId: string,
  options?: {
    deviceType?: 'gpu' | 'cpu';
    powerPreference?: 'high-performance' | 'low-power';
    allowWasmFallback?: boolean;
  },
): readonly OrtExecutionProviderOption[] {
  const normalized = String(backendId || '').toLowerCase();

  if (normalized === 'native-cpu') {
    return ['cpu'];
  }

  if (normalized.startsWith('webgpu')) {
    const webgpuEp: OrtExecutionProviderOption = {
      name: 'webgpu',
      deviceType: options?.deviceType ?? 'gpu',
      powerPreference: options?.powerPreference ?? 'high-performance',
    };
    if (options?.allowWasmFallback) {
      return [webgpuEp, 'wasm'];
    }
    return [webgpuEp];
  }

  return ['wasm'];
}

/**
 * Creates an ONNX Runtime session with unified backend selection, optional WebGPU-to-WASM fallback,
 * colocated external data mounting, and abort signal tracking.
 */
export async function createUnifiedOrtSession(
  ort: OrtModuleLike,
  modelSource: string | Uint8Array,
  options: UnifiedOrtSessionOptions = {},
): Promise<UnifiedSessionCreateResult> {
  throwIfAssetAborted(options.signal, 'session-create');

  let sessionModel: string | Uint8Array = modelSource;
  if (typeof sessionModel === 'string' && isNodeLikeRuntime()) {
    const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
    if (/^file:/i.test(sessionModel)) {
      sessionModel = fileURLToPath(sessionModel);
    }
  }

  const requestedBackend = String(options.backendId || 'wasm').toLowerCase();
  const isNativeCpu = requestedBackend === 'native-cpu';
  const isWebGpu = requestedBackend.startsWith('webgpu');

  const baseSessionOptions: Record<string, unknown> = {
    graphOptimizationLevel: options.graphOptimizationLevel ?? 'all',
    executionMode: options.executionMode ?? (isNativeCpu ? 'sequential' : 'parallel'),
    enableCpuMemArena: options.enableCpuMemArena ?? true,
    enableMemPattern: options.enableMemPattern ?? true,
    enableProfiling: options.enableProfiling ?? false,
    ...(options.sessionOptions ?? {}),
  };

  if (options.cpuThreads !== undefined && isNativeCpu) {
    baseSessionOptions.intraOpNumThreads = options.cpuThreads;
    baseSessionOptions.interOpNumThreads = 1;
  }

  if (options.preferredOutputLocation) {
    baseSessionOptions.preferredOutputLocation =
      isNodeLikeRuntime() && !isWebGpu
        ? typeof options.preferredOutputLocation === 'string'
          ? 'cpu'
          : Object.fromEntries(
              Object.keys(options.preferredOutputLocation).map((key) => [key, 'cpu']),
            )
        : options.preferredOutputLocation;
  }

  if (options.enableGraphCapture && isWebGpu) {
    baseSessionOptions.enableGraphCapture = true;
  }

  if (options.freeDimensionOverrides) {
    baseSessionOptions.freeDimensionOverrides = options.freeDimensionOverrides;
  }

  // Native ORT resolves sidecars from the graph directory. Validate those
  // paths without loading potentially multi-gigabyte files into JS memory.
  const hasExternalDataLocator = Boolean(options.externalDataUrl || options.externalDataPath);
  if (isNativeCpu && ((options.externalData?.length ?? 0) > 0 || hasExternalDataLocator)) {
    if (typeof sessionModel !== 'string') {
      throw new Error('Native external data requires a local model path or file URL.');
    }
    if (Boolean(options.externalDataUrl) !== Boolean(options.externalDataPath)) {
      throw new Error('Native external data requires both a locator and graph-relative path.');
    }
    await validateNativeOrtExternalDataFiles(sessionModel, [
      ...(options.externalData ?? []),
      ...(options.externalDataUrl && options.externalDataPath
        ? [{ dataUrl: options.externalDataUrl, path: options.externalDataPath }]
        : []),
    ]);
  }

  // ORT Web external data handling
  if (!isNativeCpu && options.externalData && options.externalData.length > 0) {
    let fileUrlToPath: ((url: string) => string) | undefined;
    if (isNodeLikeRuntime()) {
      const nodeUrl = await importNodeModule<typeof import('node:url')>('node:url');
      fileUrlToPath = nodeUrl.fileURLToPath;
    }
    baseSessionOptions.externalData = options.externalData.map((entry) => ({
      data:
        fileUrlToPath && /^file:/i.test(entry.dataUrl)
          ? fileUrlToPath(entry.dataUrl)
          : entry.dataUrl,
      path: entry.path,
    }));
  } else if (
    !isNativeCpu &&
    options.externalDataUrl &&
    options.externalDataPath &&
    typeof sessionModel === 'string'
  ) {
    const backendCategory = isWebGpu ? 'webgpu' : 'wasm';
    const mounts = await resolveOrtExternalDataMounts({
      backendId: backendCategory,
      sessionModelUrl: sessionModel,
      externalDataUrl: options.externalDataUrl,
      externalDataPath: options.externalDataPath,
    });
    if (mounts) {
      baseSessionOptions.externalData = mounts;
    }
  }

  const ortWithAbort = withOrtCreateAbort(ort as any, options.signal);
  const createSession = (
    source: string | Uint8Array,
    opts: Record<string, unknown>,
  ): Promise<OrtSessionLike> => {
    const createOpts = withNativeAbortSignalOption(opts, options.signal) ?? opts;
    return honorAbortAfterCreate(
      () => ortWithAbort.InferenceSession.create(source, createOpts),
      options.signal,
      (session) => releaseUnifiedOrtSession(session),
    );
  };

  if (isNativeCpu) {
    const nativeOptions = {
      ...baseSessionOptions,
      executionProviders: resolveUnifiedExecutionProviders('native-cpu', options),
    };
    const session = await createSession(sessionModel, nativeOptions);
    return { session, resolvedBackendId: 'native-cpu', fellBackToWasm: false };
  }

  if (isWebGpu) {
    const webgpuOptions = {
      ...baseSessionOptions,
      executionProviders: resolveUnifiedExecutionProviders('webgpu', {
        deviceType: options.deviceType,
        powerPreference: options.powerPreference,
        allowWasmFallback: false,
      }),
      ...(options.webgpuOptions ?? {}),
    };

    try {
      const session = await createSession(sessionModel, webgpuOptions);
      return { session, resolvedBackendId: 'webgpu', fellBackToWasm: false };
    } catch (error) {
      if (options.allowWasmFallback) {
        const wasmOptions: Record<string, unknown> = {
          ...baseSessionOptions,
          executionProviders: resolveUnifiedExecutionProviders('wasm'),
        };
        if (
          options.externalDataUrl &&
          options.externalDataPath &&
          typeof sessionModel === 'string'
        ) {
          const wasmMounts = await resolveOrtExternalDataMounts({
            backendId: 'wasm',
            sessionModelUrl: sessionModel,
            externalDataUrl: options.externalDataUrl,
            externalDataPath: options.externalDataPath,
          });
          if (wasmMounts) {
            wasmOptions.externalData = wasmMounts;
          }
        }
        const session = await createSession(sessionModel, wasmOptions);
        return { session, resolvedBackendId: 'wasm', fellBackToWasm: true };
      }
      throw error;
    }
  }

  // WASM fallback or explicitly requested
  const wasmOptions = {
    ...baseSessionOptions,
    executionProviders: resolveUnifiedExecutionProviders('wasm'),
  };
  const session = await createSession(sessionModel, wasmOptions);
  return { session, resolvedBackendId: 'wasm', fellBackToWasm: false };
}

/**
 * Releases an ONNX Runtime session, awaiting release if async and ignoring missing/null sessions safely.
 */
export async function releaseUnifiedOrtSession(
  session: OrtSessionLike | null | undefined,
): Promise<void> {
  if (!session) {
    return;
  }
  await session.release?.();
}
