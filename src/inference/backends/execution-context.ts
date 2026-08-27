import { BackendUnavailableError, CapabilityMismatchError } from '../../runtime/errors.js';
import type {
  BackendCapabilities,
  BackendExecutionContext,
  BackendExecutionRequest,
} from '../../types/index.js';

/**
 * Creates the capability-scoped context exposed by the backend contract.
 *
 * Model-family executors deliberately own their ORT sessions because their
 * graph lifecycles differ (for example, Whisper split graphs versus a
 * stateful RNN-T stream). This context therefore owns the probed backend
 * capability lease, not model sessions or tensors.
 */
export function createBackendExecutionContext(
  request: BackendExecutionRequest,
  capabilities: BackendCapabilities,
): BackendExecutionContext {
  if (!capabilities.available) {
    throw new BackendUnavailableError(
      `Backend "${capabilities.id}" is unavailable for model "${request.modelId}".`,
      {
        backendId: capabilities.id,
        modelFamily: request.modelFamily,
        modelId: request.modelId,
        capabilities,
      },
    );
  }

  if (request.precision && !capabilities.supportedPrecisions.includes(request.precision)) {
    throw new CapabilityMismatchError(
      `Backend "${capabilities.id}" does not support ${request.precision} precision for model "${request.modelId}".`,
      {
        backendId: capabilities.id,
        modelFamily: request.modelFamily,
        modelId: request.modelId,
        requestedPrecision: request.precision,
        supportedPrecisions: capabilities.supportedPrecisions,
      },
    );
  }

  let disposed = false;
  return {
    backendId: capabilities.id,
    capabilities,
    ...(capabilities.provider ? { provider: capabilities.provider } : {}),
    dispose(): void {
      if (disposed) {
        return;
      }
      disposed = true;
    },
  };
}
