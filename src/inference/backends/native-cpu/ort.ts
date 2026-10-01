import { BackendUnavailableError } from '../../../runtime/errors.js';
import { importNodePackage, isNodeLikeRuntime } from '../../../io/node-compat.js';

/** Lazy native binding access: no native package is imported by a browser entrypoint. */
export function loadNativeCpuOrt(): typeof import('onnxruntime-node') {
  if (!isNodeLikeRuntime()) {
    throw new BackendUnavailableError('Native CPU inference requires a Node-compatible runtime.', {
      backendId: 'native-cpu',
    });
  }
  try {
    const ort = importNodePackage<typeof import('onnxruntime-node')>('onnxruntime-node');
    if (!ort.listSupportedBackends().some((backend) => backend.name === 'cpu' && backend.bundled)) {
      throw new Error('The installed native binding does not provide the CPU execution provider.');
    }
    return ort;
  } catch (error) {
    throw new BackendUnavailableError('The native ORT CPU binding could not be loaded.', {
      backendId: 'native-cpu',
      cause: error,
    });
  }
}
