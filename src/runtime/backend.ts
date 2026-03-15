import type {
  BackendCapabilities,
  BackendSelectionCriteria,
  ExecutionBackend,
} from '../types/index.js';
import { BackendUnavailableError, CapabilityMismatchError } from './errors.js';

export interface BackendCandidate {
  readonly backend: ExecutionBackend;
  readonly capabilities: BackendCapabilities;
}

function precisionScore(
  capabilities: BackendCapabilities,
  criteria: BackendSelectionCriteria,
): number {
  if (!criteria.requiredPrecision) {
    return 0;
  }
  return capabilities.supportedPrecisions.includes(criteria.requiredPrecision) ? 0 : 1;
}

function preferredIndex(backendId: string, criteria: BackendSelectionCriteria): number {
  if (!criteria.preferredBackendIds || criteria.preferredBackendIds.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  const index = criteria.preferredBackendIds.indexOf(backendId);
  return index === -1 ? Number.POSITIVE_INFINITY : index;
}

function accelerationScore(capabilities: BackendCapabilities): number {
  if (capabilities.acceleration.includes('npu')) return 0;
  if (capabilities.acceleration.includes('gpu')) return 1;
  if (capabilities.acceleration.includes('hybrid')) return 2;
  return 3;
}

export function matchesBackendCriteria(
  capabilities: BackendCapabilities,
  criteria: BackendSelectionCriteria,
): boolean {
  if (!capabilities.available) {
    return false;
  }

  if (!criteria.allowExperimental && capabilities.experimental) {
    return false;
  }

  if (
    criteria.requiredPrecision &&
    !capabilities.supportedPrecisions.includes(criteria.requiredPrecision)
  ) {
    return false;
  }

  if (criteria.requireSharedArrayBuffer && !capabilities.supportsSharedArrayBuffer) {
    return false;
  }

  if (criteria.environment && !capabilities.environments.includes(criteria.environment)) {
    return false;
  }

  return true;
}

export function sortBackendCandidates(
  candidates: readonly BackendCandidate[],
  criteria: BackendSelectionCriteria = {},
): BackendCandidate[] {
  const preferAcceleration = criteria.preferAcceleration ?? true;

  return [...candidates].sort((left, right) => {
    const preferredDelta =
      preferredIndex(left.backend.id, criteria) - preferredIndex(right.backend.id, criteria);
    if (preferredDelta !== 0) {
      return preferredDelta;
    }

    const precisionDelta =
      precisionScore(left.capabilities, criteria) - precisionScore(right.capabilities, criteria);
    if (precisionDelta !== 0) {
      return precisionDelta;
    }

    if (preferAcceleration) {
      const accelerationDelta =
        accelerationScore(left.capabilities) - accelerationScore(right.capabilities);
      if (accelerationDelta !== 0) {
        return accelerationDelta;
      }
    }

    if (
      left.capabilities.requiresSharedArrayBuffer !== right.capabilities.requiresSharedArrayBuffer
    ) {
      return left.capabilities.requiresSharedArrayBuffer ? 1 : -1;
    }

    if (left.capabilities.fallbackSuitable !== right.capabilities.fallbackSuitable) {
      return left.capabilities.fallbackSuitable ? -1 : 1;
    }

    return right.capabilities.priority - left.capabilities.priority;
  });
}

export function ensureAvailableBackendCandidates(
  candidates: readonly BackendCandidate[],
  criteria: BackendSelectionCriteria = {},
): BackendCandidate[] {
  if (candidates.length === 0) {
    throw new BackendUnavailableError('No execution backends are registered in the runtime.');
  }

  const available = candidates.filter((candidate) => candidate.capabilities.available);
  if (available.length === 0) {
    throw new BackendUnavailableError('No registered execution backend is currently available.', {
      backendIds: candidates.map((candidate) => candidate.backend.id),
    });
  }

  const supported = available.filter((candidate) =>
    matchesBackendCriteria(candidate.capabilities, criteria),
  );
  if (supported.length === 0) {
    throw new CapabilityMismatchError(
      'No available backend satisfies the requested capabilities.',
      {
        criteria,
        availableBackends: available.map((candidate) => ({
          id: candidate.backend.id,
          precisions: candidate.capabilities.supportedPrecisions,
          requiresSharedArrayBuffer: candidate.capabilities.requiresSharedArrayBuffer,
        })),
      },
    );
  }

  return supported;
}
