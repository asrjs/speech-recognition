/**
 * Alignment Module — DTW-based timestamp alignment for ASR output.
 *
 * Model-agnostic. Works with attention matrices from any encoder-decoder
 * ASR model (Whisper, etc.).
 *
 * Independently importable: @asrjs/speech-recognition/alignment
 */

export { crossAttentionDtwTimestamps } from './cross-attention-dtw.js';
