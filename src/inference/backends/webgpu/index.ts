import { NotImplementedSpeechFeatureError } from '../../../runtime/errors.js';
import type {
  BackendCapabilities,
  BackendEnvironment,
  BackendExecutionRequest,
  ExecutionBackend,
} from '../../../types/index.js';

interface NavigatorWithGpu extends Navigator {
  gpu?: {
    requestAdapter?: () => Promise<{
      features?: Set<string> | { has(feature: string): boolean };
      info?: { architecture?: string; vendor?: string };
    } | null>;
  };
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

export async function probeWebGpuCapabilities(): Promise<BackendCapabilities> {
  const navigatorLike =
    typeof navigator !== 'undefined' ? (navigator as NavigatorWithGpu) : undefined;
  const notes = ['Primary browser acceleration path when WebGPU is available.'];
  let available = false;
  let supportsFp16 = false;
  let adapterInfo: { architecture?: string; vendor?: string } | undefined;

  if (navigatorLike?.gpu?.requestAdapter) {
    try {
      const adapter = await navigatorLike.gpu.requestAdapter();
      available = adapter !== null;
      supportsFp16 = !!adapter?.features?.has?.('shader-f16');
      adapterInfo = adapter?.info;
      if (!available) {
        notes.push('navigator.gpu exists but requestAdapter() returned null.');
      }
    } catch (error) {
      notes.push(`WebGPU adapter probe failed: ${String(error)}`);
    }
  } else {
    notes.push('navigator.gpu is not available.');
  }

  if (!supportsFp16) {
    notes.push('FP16 should be treated as capability-probed rather than assumed.');
  }

  return {
    id: 'webgpu',
    displayName: 'WebGPU',
    available,
    priority: 100,
    environments: detectEnvironments(),
    acceleration: ['gpu'],
    supportedPrecisions: supportsFp16 ? ['fp32', 'fp16', 'int8'] : ['fp32', 'int8'],
    supportsFp16,
    supportsInt8: true,
    supportsSharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    requiresSharedArrayBuffer: false,
    fallbackSuitable: true,
    ...(adapterInfo?.architecture ? { adapter: adapterInfo.architecture } : {}),
    ...(adapterInfo?.vendor ? { provider: adapterInfo.vendor } : {}),
    notes,
  };
}

export function createWebGpuBackend(): ExecutionBackend {
  return {
    id: 'webgpu',
    displayName: 'WebGPU',
    probeCapabilities: probeWebGpuCapabilities,
    async createExecutionContext(request: BackendExecutionRequest) {
      const capabilities = await probeWebGpuCapabilities();
      throw new NotImplementedSpeechFeatureError(
        'WebGPU backend execution context creation is not implemented in this scaffold.',
        {
          backendId: 'webgpu',
          modelId: request.modelId,
          capabilities,
        },
      );
    },
  };
}
