import type { BackendCapabilities, BackendProbeContext, ExecutionBackend } from '../../types/index.js';

export async function probeWasmCapabilities(): Promise<BackendCapabilities> {
  const impl = await import('./wasm/probe.js');
  return impl.probeWasmCapabilities();
}

export function createWasmBackend(): ExecutionBackend {
  return {
    id: 'wasm',
    displayName: 'WASM',
    probeCapabilities: probeWasmCapabilities,
    async createExecutionContext(request) {
      const capabilities = await probeWasmCapabilities();
      const { createBackendExecutionContext } = await import('./execution-context.js');
      return createBackendExecutionContext(request, capabilities);
    },
  };
}

export async function probeWebGpuCapabilities(): Promise<BackendCapabilities> {
  const impl = await import('./webgpu/probe.js');
  return impl.probeWebGpuCapabilities();
}

export function createWebGpuBackend(): ExecutionBackend {
  return {
    id: 'webgpu',
    displayName: 'WebGPU',
    probeCapabilities: probeWebGpuCapabilities,
    async createExecutionContext(request) {
      const capabilities = await probeWebGpuCapabilities();
      const { createBackendExecutionContext } = await import('./execution-context.js');
      return createBackendExecutionContext(request, capabilities);
    },
  };
}

export async function probeWebNnCapabilities(): Promise<BackendCapabilities> {
  const impl = await import('./webnn/probe.js');
  return impl.probeWebNnCapabilities();
}

export function createWebNnBackend(): ExecutionBackend {
  return {
    id: 'webnn',
    displayName: 'WebNN',
    probeCapabilities: probeWebNnCapabilities,
    async createExecutionContext(request) {
      const capabilities = await probeWebNnCapabilities();
      const { createBackendExecutionContext } = await import('./execution-context.js');
      return createBackendExecutionContext(request, capabilities);
    },
  };
}

export async function probeWebGlCapabilities(): Promise<BackendCapabilities> {
  const impl = await import('./webgl/probe.js');
  return impl.probeWebGlCapabilities();
}

export function createWebGlBackend(): ExecutionBackend {
  return {
    id: 'webgl',
    displayName: 'WebGL',
    probeCapabilities: probeWebGlCapabilities,
    async createExecutionContext(request) {
      const capabilities = await probeWebGlCapabilities();
      const { createBackendExecutionContext } = await import('./execution-context.js');
      return createBackendExecutionContext(request, capabilities);
    },
  };
}

export async function probeNativeCpuCapabilities(context: BackendProbeContext = {}): Promise<BackendCapabilities> {
  const impl = await import('./native-cpu/probe.js');
  return impl.probeNativeCpuCapabilities(context);
}

export function createNativeCpuBackend(): ExecutionBackend {
  return {
    id: 'native-cpu',
    displayName: 'Native ORT CPU',
    requiresModelOptIn: true,
    probeCapabilities: probeNativeCpuCapabilities,
    async createExecutionContext(request) {
      const capabilities = await probeNativeCpuCapabilities();
      const { createBackendExecutionContext } = await import('./execution-context.js');
      return createBackendExecutionContext(request, capabilities);
    },
  };
}
