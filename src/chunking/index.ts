/**
 * Chunking Module — audio pre-segmentation for long-form ASR.
 *
 * Model-agnostic. Works with any ASR model.
 * Independently importable: @asrjs/speech-recognition/chunking
 */

export type { VadSpeechSegment, WhisperVadBackend, DriftCorrectionResult } from './types.js';
export type { VadMergeConfig, VadBinarizeOptions, NoiseGateOptions, SegmentAudioOptions } from './vad-segmenter.js';
export { DriftHandler } from './drift-handler.js';
export { mergeVadSegments, vadBinarize, noiseGate, segmentAudio } from './vad-segmenter.js';
export { FixedWindowChunker, type FixedWindowConfig, type AudioWindow } from './fixed-window.js';
