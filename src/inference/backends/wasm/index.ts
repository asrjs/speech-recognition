import { NotImplementedSpeechFeatureError } from '../../../runtime/errors.js';
import type {
  BackendCapabilities,
  BackendEnvironment,
  BackendExecutionRequest,
  ExecutionBackend,
} from '../../../types/index.js';

function detectEnvironments(): BackendEnvironment[] {
  const environments: BackendEnvironment[] = [];

  if (typeof window !== 'undefined') {
    environments.push('browser');
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    environments.push('node');
  }
  if (
    typeof (globalThis as { importScripts?: unknown }).importScripts === 'function' &&
    typeof window === 'undefined'
  ) {
    environments.push('worker');
  }

  return environments;
}

function hasSharedArrayBufferSupport(): boolean {
  return typeof SharedArrayBuffer !== 'undefined';
}

export async function probeWasmCapabilities(): Promise<BackendCapabilities> {
  const supportsThreads = hasSharedArrayBufferSupport();
  const notes = [
    'Universal baseline backend for browser and local inference.',
    supportsThreads
      ? 'SharedArrayBuffer is available, so threaded WASM is possible.'
      : 'SharedArrayBuffer is unavailable, so threaded WASM should not be assumed.',
  ];

  return {
    id: 'wasm',
    displayName: 'WASM',
    available: typeof WebAssembly !== 'undefined',
    priority: 60,
    environments: detectEnvironments(),
    acceleration: ['cpu'],
    supportedPrecisions: ['fp32', 'int8'],
    supportsFp16: false,
    supportsInt8: true,
    supportsSharedArrayBuffer: supportsThreads,
    requiresSharedArrayBuffer: false,
    fallbackSuitable: true,
    notes,
  };
}

export function createWasmBackend(): ExecutionBackend {
  return {
    id: 'wasm',
    displayName: 'WASM',
    probeCapabilities: probeWasmCapabilities,
    async createExecutionContext(request: BackendExecutionRequest) {
      const capabilities = await probeWasmCapabilities();
      throw new NotImplementedSpeechFeatureError(
        'WASM backend execution context creation is not implemented in this scaffold.',
        {
          backendId: 'wasm',
          modelId: request.modelId,
          capabilities,
        },
      );
    },
  };
}
