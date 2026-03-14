export interface SpeechRuntimeErrorDetails {
  readonly [key: string]: unknown;
}

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
  constructor(message: string, details?: SpeechRuntimeErrorDetails) {
    super(message, 'model-load-error', details);
  }
}

export class NotImplementedSpeechFeatureError extends SpeechRuntimeError {
  constructor(message: string, details?: SpeechRuntimeErrorDetails) {
    super(message, 'not-implemented-speech-feature', details);
  }
}

