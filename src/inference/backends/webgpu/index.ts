import type {
  BackendCapabilities,
  BackendEnvironment,
  BackendExecutionRequest,
  ExecutionBackend,
} from '../../../types/index.js';
import { createBackendExecutionContext } from '../execution-context.js';

type WebGpuAdapterRequestOptions = {
  readonly powerPreference?: 'low-power' | 'high-performance';
};

type WebGpuAdapterLike = {
  features?: Set<string> | { has(feature: string): boolean };
  info?: { architecture?: string; vendor?: string };
};

interface NavigatorWithGpu extends Navigator {
  gpu?: {
    requestAdapter?: (options?: WebGpuAdapterRequestOptions) => Promise<WebGpuAdapterLike | null>;
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
    // ASR is a sustained workload. Prefer the discrete/high-performance
    // adapter, but keep a default-selection retry for browsers that expose
    // WebGPU while rejecting the preference or returning null for it.
    let adapter: WebGpuAdapterLike | null = null;
    try {
      adapter = await navigatorLike.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
    } catch (error) {
      notes.push(
        `High-performance WebGPU adapter probe failed: ${String(error)}; retrying default adapter selection.`,
      );
    }
    if (adapter === null) {
      if (!notes.some((note) => note.includes('retrying default adapter selection'))) {
        notes.push(
          'High-performance adapter selection returned null; retrying default WebGPU adapter selection.',
        );
      }
      try {
        adapter = await navigatorLike.gpu.requestAdapter();
      } catch (error) {
        notes.push(`Default WebGPU adapter probe failed: ${String(error)}`);
      }
    }
    available = adapter !== null;
    supportsFp16 = !!adapter?.features?.has?.('shader-f16');
    adapterInfo = adapter?.info;
    if (!available) {
      notes.push('navigator.gpu exists but both adapter selections returned null.');
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
      return createBackendExecutionContext(request, capabilities);
    },
  };
}
