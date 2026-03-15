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
  if (
    typeof (globalThis as { importScripts?: unknown }).importScripts === 'function' &&
    typeof window === 'undefined'
  ) {
    environments.push('worker');
  }

  return environments;
}

function detectWebGlAvailability(): boolean {
  if (typeof document === 'undefined') {
    return false;
  }

  const canvas = document.createElement('canvas');
  return !!canvas.getContext('webgl2') || !!canvas.getContext('webgl');
}

export async function probeWebGlCapabilities(): Promise<BackendCapabilities> {
  const available = detectWebGlAvailability();

  return {
    id: 'webgl',
    displayName: 'WebGL',
    available,
    priority: 20,
    environments: detectEnvironments(),
    acceleration: ['gpu'],
    supportedPrecisions: ['fp32'],
    supportsFp16: false,
    supportsInt8: false,
    supportsSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    requiresSharedArrayBuffer: false,
    fallbackSuitable: true,
    notes: available
      ? ['Compatibility fallback backend. Lower strategic priority than WebGPU and WASM.']
      : ['WebGL context creation failed or is unavailable in this environment.'],
  };
}

export function createWebGlBackend(): ExecutionBackend {
  return {
    id: 'webgl',
    displayName: 'WebGL',
    probeCapabilities: probeWebGlCapabilities,
    async createExecutionContext(request: BackendExecutionRequest) {
      const capabilities = await probeWebGlCapabilities();
      throw new NotImplementedSpeechFeatureError(
        'WebGL backend execution context creation is not implemented in this scaffold.',
        {
          backendId: 'webgl',
          modelId: request.modelId,
          capabilities,
        },
      );
    },
  };
}
