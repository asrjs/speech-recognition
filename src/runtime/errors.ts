export interface SpeechRuntimeErrorDetails {
  readonly [key: string]: unknown;
}

export const EXPERIMENTAL_ARTIFACT_MISSING_CODE = 'experimental-artifact-missing' as const;
export type ExperimentalArtifactMissingCode = typeof EXPERIMENTAL_ARTIFACT_MISSING_CODE;

export class SpeechRuntimeError extends Error {
  readonly code: string;
  readonly details?: SpeechRuntimeErrorDetails;

  constructor(message: string, code = 'speech-runtime-error', details?: SpeechRuntimeErrorDetails) {
    super(message);
    this.name = new.target.name;
    this.code = code;
    this.details = details;
  }
}

export class BackendUnavailableError extends SpeechRuntimeError {
  constructor(message: string, details?: SpeechRuntimeErrorDetails) {
    super(message, 'backend-unavailable', details);
  }
}

export class CapabilityMismatchError extends SpeechRuntimeError {
  constructor(message: string, details?: SpeechRuntimeErrorDetails) {
    super(message, 'capability-mismatch', details);
  }
}

export class ModelLoadError extends SpeechRuntimeError {
  constructor(message: string, details?: SpeechRuntimeErrorDetails, code = 'model-load-error') {
    super(message, code, details);
  }
}

export class ExperimentalArtifactMissingError extends ModelLoadError {
  override readonly code: ExperimentalArtifactMissingCode = EXPERIMENTAL_ARTIFACT_MISSING_CODE;

  constructor(message: string, details?: SpeechRuntimeErrorDetails) {
    super(message, details, EXPERIMENTAL_ARTIFACT_MISSING_CODE);
  }
}

export function isExperimentalArtifactMissingError(
  error: unknown,
): error is ExperimentalArtifactMissingError {
  if (error instanceof ExperimentalArtifactMissingError) {
    return true;
  }
  if (!error || typeof error !== 'object') {
    return false;
  }
  const candidate = error as { name?: unknown; code?: unknown };
  return (
    candidate.name === 'ExperimentalArtifactMissingError' &&
    candidate.code === EXPERIMENTAL_ARTIFACT_MISSING_CODE
  );
}

export class NotImplementedSpeechFeatureError extends SpeechRuntimeError {
  constructor(message: string, details?: SpeechRuntimeErrorDetails) {
    super(message, 'not-implemented-speech-feature', details);
  }
}
