/**
 * Post-Processing Module — transcript refinement for any ASR output.
 *
 * Model-agnostic. Works with any ASR model's output format.
 * Independently importable: @asrjs/speech-recognition/post-processing
 */

export { mergeSegments, type MergedTranscription } from './segment-merger.js';
export { deduplicateWords, normalizeText, buildSentences, type DedupWord, type Sentence } from './extras.js';
