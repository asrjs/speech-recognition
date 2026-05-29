/**
 * Chunking Module — audio pre-segmentation for long-form ASR.
 *
 * Model-agnostic. Works with any ASR model.
 * Independently importable: @asrjs/speech-recognition/chunking
 */

export type { VadSpeechSegment, WhisperVadBackend, DriftCorrectionResult } from './types.js';
export { DriftHandler } from './drift-handler.js';
export { mergeVadSegments } from './vad-segmenter.js';
