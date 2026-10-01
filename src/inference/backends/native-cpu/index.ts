import type {
  BackendCapabilities,
  BackendProbeContext,
  ExecutionBackend,
} from '../../../types/index.js';
import { isNodeLikeRuntime } from '../../../io/node-compat.js';
import { createBackendExecutionContext } from '../execution-context.js';
import { loadNativeCpuOrt } from './ort.js';

export async function probeNativeCpuCapabilities(
  context: BackendProbeContext = {},
): Promise<BackendCapabilities> {
  let available = false;
  const notes = ['Native ORT CPU; model-family opt-in is required.'];
  if (!isNodeLikeRuntime() || (context.environment && context.environment !== 'node')) {
    notes.push('A Node-compatible host is required; browser WASM is not a native fallback.');
  } else {
    try {
      loadNativeCpuOrt();
      available = true;
    } catch {
      notes.push('The installed native CPU binding is unavailable.');
    }
  }
  return {
    id: 'native-cpu',
    displayName: 'Native ORT CPU',
    available,
    priority: 70,
    environments: isNodeLikeRuntime() ? ['node'] : [],
    acceleration: ['cpu'],
    supportedPrecisions: available ? ['fp32'] : [],
    supportsFp16: false,
    supportsInt8: false,
    supportsSharedArrayBuffer: false,
    requiresSharedArrayBuffer: false,
    fallbackSuitable: false,
    experimental: true,
    provider: 'onnxruntime-node/cpu',
    notes,
  };
}

/** Opt-in only: intentionally not registered by createBuiltInSpeechRuntime(). */
export function createNativeCpuBackend(): ExecutionBackend {
  return {
    id: 'native-cpu',
    displayName: 'Native ORT CPU',
    requiresModelOptIn: true,
    probeCapabilities: probeNativeCpuCapabilities,
    async createExecutionContext(request) {
      return createBackendExecutionContext(request, await probeNativeCpuCapabilities());
    },
  };
}
