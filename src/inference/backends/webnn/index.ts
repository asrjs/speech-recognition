import type {
  BackendCapabilities,
  BackendEnvironment,
  BackendExecutionRequest,
  ExecutionBackend,
} from '../../../types/index.js';
import { createBackendExecutionContext } from '../execution-context.js';

interface NavigatorWithMl extends Navigator {
  ml?: object;
}

function detectEnvironments(): BackendEnvironment[] {
  const environments: BackendEnvironment[] = [];

  if (typeof window !== 'undefined') {
    environments.push('browser');
  }
  if (
    typeof (globalThis as { importScripts?: unknown }).importScripts === 'function' &&
    typeof window === 'undefined'
  ) {
    environments.push('worker');
  }

  return environments;
}

export async function probeWebNnCapabilities(): Promise<BackendCapabilities> {
  const navigatorLike =
    typeof navigator !== 'undefined' ? (navigator as NavigatorWithMl) : undefined;
  const available = !!navigatorLike?.ml;

  return {
    id: 'webnn',
    displayName: 'WebNN',
    available,
    priority: 95,
    environments: detectEnvironments(),
    acceleration: ['npu', 'gpu', 'cpu'],
    supportedPrecisions: ['fp32', 'fp16', 'int8'],
    supportsFp16: true,
    supportsInt8: true,
    supportsSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    requiresSharedArrayBuffer: false,
    fallbackSuitable: false,
    experimental: true,
    notes: available
      ? ['WebNN detected. Treat provider behavior as capability-probed and experimental.']
      : ['navigator.ml is not available. WebNN should be treated as optional.'],
  };
}

export function createWebNnBackend(): ExecutionBackend {
  return {
    id: 'webnn',
    displayName: 'WebNN',
    probeCapabilities: probeWebNnCapabilities,
    async createExecutionContext(request: BackendExecutionRequest) {
      const capabilities = await probeWebNnCapabilities();
      return createBackendExecutionContext(request, capabilities);
    },
  };
}
