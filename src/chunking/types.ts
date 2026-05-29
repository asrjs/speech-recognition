/**
 * Chunking Types — audio pre-segmentation contracts.
 * Model-agnostic. Works with any ASR model.
 */

/** A speech segment detected by VAD. */
export interface VadSpeechSegment {
  readonly startSeconds: number;
  readonly endSeconds: number;
  readonly durationSeconds: number;
}

/**
 * Backend interface for VAD-based audio segmentation.
 * Implementations wrap TenVAD, FireRed VAD, or any other VAD model.
 */
export interface WhisperVadBackend {
  segment(audio: Float32Array, sampleRate: number, threshold: number): Promise<VadSpeechSegment[]>;
}

/** Drift correction result from seek-based timestamp adjustment. */
export interface DriftCorrectionResult {
  readonly start: number;
  readonly end: number;
  readonly corrected: boolean;
}
