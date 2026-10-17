import { resolveWasmThreads } from '../backends/wasm-threads.js';
import { loadNativeCpuOrt } from '../backends/native-cpu/ort.js';
import { importNodePackage, isNodeLikeRuntime } from '../../io/node-compat.js';
import { throwIfAssetAborted } from '../../io/abort.js';
import { CapabilityMismatchError } from '../../runtime/errors.js';
import type { OrtModuleLike, UnifiedOrtInitOptions } from './types.js';

function resolveViteDevOrtWasmPath(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  const marker = '/@fs/';
  const metaUrl = import.meta.url;
  const markerIndex = metaUrl.indexOf(marker);
  if (markerIndex < 0) {
    return null;
  }
  const afterMarker = metaUrl.slice(markerIndex + marker.length).split('?')[0]?.split('#')[0] ?? '';
  const decodedPath = decodeURIComponent(afterMarker);
  const sourceSuffix = '/src/inference/ort/env.ts';
  if (decodedPath.endsWith(sourceSuffix)) {
    const repoRoot = decodedPath.slice(0, -sourceSuffix.length);
    const ortDist = `${repoRoot}/node_modules/onnxruntime-web/dist/`;
    return `/@fs/${encodeURI(ortDist)}`;
  }
  const distSuffix = '/dist/inference/ort/env.js';
  if (decodedPath.endsWith(distSuffix)) {
    const repoRoot = decodedPath.slice(0, -distSuffix.length);
    const ortDist = `${repoRoot}/node_modules/onnxruntime-web/dist/`;
    return `/@fs/${encodeURI(ortDist)}`;
  }
  return null;
}

/**
 * Resolves the WASM directory URL prefix across local node_modules, Vite dev server, and CDN.
 * Does not clobber already-configured paths unless custom path is explicitly provided.
 */
export function resolveUnifiedWasmPaths(customWasmPaths?: string, ortVersion?: string): string {
  if (customWasmPaths) {
    return customWasmPaths;
  }
  const version = ortVersion ?? '1.29.0';
  if (typeof window !== 'undefined') {
    const vitePath = resolveViteDevOrtWasmPath();
    if (vitePath) {
      return vitePath;
    }
    // Browser local dist
    return '/node_modules/onnxruntime-web/dist/';
  }
  // Node / Server / CDN fallback
  return `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;
}

/**
 * Canonical configuration for ONNX Runtime environment options.
 * Configures wasmPaths, simd = true, worker thread count, proxy = false, and logSeverityLevel = 3.
 */
export function configureUnifiedOrtEnv(
  ort: OrtModuleLike,
  options: UnifiedOrtInitOptions = {},
): void {
  throwIfAssetAborted(options.signal, 'session-create');

  if (!ort.env) {
    (ort as { env: OrtModuleLike['env'] }).env = { wasm: {} };
  }
  ort.env.wasm ??= {};

  // 1. wasmPaths: do not clobber existing unless requested
  const existingWasm = ort.env.wasm.wasmPaths;
  const requestedWasm = options.wasmPaths ?? options.customWasmPaths;
  if (requestedWasm) {
    ort.env.wasm.wasmPaths = requestedWasm;
  } else if (!existingWasm) {
    ort.env.wasm.wasmPaths = resolveUnifiedWasmPaths(
      undefined,
      options.ortVersion ?? ort.env.versions?.common,
    );
  }

  // 2. simd = true
  ort.env.wasm.simd = options.simd ?? true;

  // 3. numThreads based on SharedArrayBuffer / crossOriginIsolated and navigator.hardwareConcurrency
  const hasSharedArrayBuffer =
    typeof SharedArrayBuffer !== 'undefined' &&
    (typeof crossOriginIsolated === 'undefined' || crossOriginIsolated);

  if (hasSharedArrayBuffer) {
    ort.env.wasm.numThreads = resolveWasmThreads(options.cpuThreads, 1);
  } else {
    ort.env.wasm.numThreads = 1;
  }

  // 4. proxy = false
  ort.env.wasm.proxy = options.proxy ?? false;

  // 5. logSeverityLevel = 3
  ort.env.logSeverityLevel = options.logSeverityLevel ?? 3;

  // WebGPU profiling if requested
  if (options.enableProfiling) {
    ort.env.webgpu ??= {};
    (ort.env.webgpu as Record<string, unknown>).profiling = { mode: 'default' };
  }
}

/**
 * Dynamically imports the appropriate ONNX Runtime module for the target backend and host runtime.
 */
export async function importUnifiedOrtModule(backendId: string): Promise<OrtModuleLike> {
  const normalized = String(backendId || '').toLowerCase();

  if (normalized === 'native-cpu') {
    if (isNodeLikeRuntime()) {
      try {
        const nativeModule = loadNativeCpuOrt();
        return nativeModule as unknown as OrtModuleLike;
      } catch (error) {
        try {
          const imported = importNodePackage<
            OrtModuleLike & { readonly default?: OrtModuleLike }
          >('onnxruntime-node');
          return imported.default ?? imported;
        } catch {
          throw error;
        }
      }
    }
    throw new CapabilityMismatchError(
      'native-cpu backend is only available in Node-compatible environments.',
      { backendId },
    );
  }

  const isWebGpu = normalized.startsWith('webgpu');
  if (isWebGpu && typeof window !== 'undefined') {
    try {
      const imported = (await import('onnxruntime-web/webgpu')) as unknown as OrtModuleLike & {
        readonly default?: OrtModuleLike;
      };
      return imported.default ?? imported;
    } catch {
      // Fall through to standard onnxruntime-web if subpath is not resolvable
    }
  }

  const imported = (await import('onnxruntime-web')) as unknown as OrtModuleLike & {
    readonly default?: OrtModuleLike;
  };
  return imported.default ?? imported;
}
