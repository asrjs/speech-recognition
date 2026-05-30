/**
 * VAD Segmenter — audio pre-segmentation for any ASR model.
 *
 * Uses existing VAD backends (TenVAD or FireRed) in the project to detect
 * speech regions in long audio, then feeds each segment to the ASR model.
 *
 * Architecture: Dependency injection via WhisperVadBackend interface.
 * The actual backend adapter is wired at construction time.
 *
 * Backends already in the project:
 *   - TenVAD: WASM-based, fast, bundled src/runtime/ten-vad-browser.ts
 *   - FireRed VAD: ONNX-based, streaming + file-mode src/runtime/firered-vad/
 *
 * ## WhisperX pipeline reference
 *
 * WhisperX VAD + merge flow:
 *   1. VAD detects speech segments (Silero or Pyannote)
 *   2. `merge_chunks`: merge adjacent segments, start new chunk when
 *      accumulated duration exceeds chunk_size (30s default)
 *   3. vad_onset (0-1): VAD threshold for speech detection
 *   4. vad_offset (optional): not used in current WhisperX merge
 *   5. No overlap between chunks in WhisperX — each chunk is independent
 *
 * Our enhanced version adds:
 *   - Overlap between consecutive chunks for boundary safety
 *   - Energy RMS binarization from VAD probabilities
 *   - Noise floor gating (zero out silence frames)
 */

import type { VadSpeechSegment, WhisperVadBackend } from './types.js';

export type { VadSpeechSegment, WhisperVadBackend };

// ---------------------------------------------------------------------------
// Merge configuration (WhisperX-compatible)
// ---------------------------------------------------------------------------

export interface VadMergeConfig {
  /** Minimum silence gap to split segments (ms). Default: 100 */
  readonly minSilenceDurationMs?: number;
  /** Pad speech segments on each side (ms). Default: 400 (200ms each side) */
  readonly speechPadMs?: number;
  /** Maximum segment duration before forced split (ms). Default: 29000 */
  readonly maxSegmentDurationMs?: number;
  /** Minimum speech duration to keep a segment (ms). Default: 250 */
  readonly minSpeechDurationMs?: number;
  /**
   * Overlap between consecutive chunks from the same segment (ms).
   * Default: 0 (no overlap). Set to ~500-1000ms for boundary safety.
   *
   * When a segment is split into multiple chunks (because it exceeds
   * maxSegmentDurationMs), each consecutive chunk will overlap by this
   * amount to prevent word cutting at boundaries.
   */
  readonly overlapDurationMs?: number;
  /**
   * VAD onset threshold — speech start sensitivity (0-1).
   * Lower = more sensitive. Default: 0.5
   * Equivalent to WhisperX `--vad_onset`.
   */
  readonly vadOnset?: number;
  /**
   * VAD offset — not used in WhisperX merge. Reserved for future use.
   * Equivalent to WhisperX `--vad_offset`.
   */
  readonly vadOffset?: number;
}

// ---------------------------------------------------------------------------
// Defaults (matching WhisperX)
// ---------------------------------------------------------------------------

const DEFAULT_MIN_SILENCE_MS = 100;
const DEFAULT_SPEECH_PAD_MS = 400;
const DEFAULT_MAX_SEGMENT_MS = 29_000;
const DEFAULT_MIN_SPEECH_MS = 250;
const DEFAULT_OVERLAP_MS = 0;
const DEFAULT_VAD_ONSET = 0.5;

// ---------------------------------------------------------------------------
// mergeVadSegments — enhanced with overlap support
// ---------------------------------------------------------------------------

/**
 * Merge, pad, cap, and (optionally) overlap VAD segments for ASR processing.
 *
 * Pipeline:
 *   1. Filter segments shorter than minSpeechDurationMs
 *   2. Merge adjacent segments with gaps < minSilenceDurationMs
 *   3. Pad each segment by speechPadMs on both sides
 *   4. Clamp padded segments to [0, audioEnd]
 *   5. Split segments longer than maxSegmentDurationMs
 *   6. Apply overlap between consecutive chunks from split segments
 */
export function mergeVadSegments(
  segments: readonly VadSpeechSegment[],
  minSilenceDurationMs: number = DEFAULT_MIN_SILENCE_MS,
  speechPadMs: number = DEFAULT_SPEECH_PAD_MS,
  maxSegmentDurationMs: number = DEFAULT_MAX_SEGMENT_MS,
  minSpeechDurationMs: number = DEFAULT_MIN_SPEECH_MS,
  overlapDurationMs: number = DEFAULT_OVERLAP_MS,
): VadSpeechSegment[] {
  if (segments.length === 0) return [];

  const minSilenceSec = minSilenceDurationMs / 1000;
  const padSec = speechPadMs / 1000;
  const maxSec = maxSegmentDurationMs / 1000;
  const minSec = minSpeechDurationMs / 1000;
  const overlapSec = Math.max(0, overlapDurationMs / 1000);

  // ── 1. Filter too-short segments ──
  const filtered = segments.filter((s) => s.durationSeconds >= minSec);
  if (filtered.length === 0) return [];

  // ── 2. Merge adjacent segments with small gaps ──
  const merged: VadSpeechSegment[] = [];
  let current = { ...filtered[0]! };

  for (let i = 1; i < filtered.length; i++) {
    const next = filtered[i]!;
    const gap = next.startSeconds - current.endSeconds;
    if (gap < minSilenceSec) {
      // Merge: extend current to cover both
      current = {
        startSeconds: current.startSeconds,
        endSeconds: next.endSeconds,
        durationSeconds: next.endSeconds - current.startSeconds,
      };
    } else {
      merged.push(current);
      current = { ...next };
    }
  }
  merged.push(current);

  // ── 3. Pad on both sides, clamp to [0, ∞) ──
  // Note: padding can extend beyond the last segment's end.
  // The audio extraction step handles bounds at sample level.

  const padded = merged.map((seg) => {
    const start = Math.max(0, seg.startSeconds - padSec);
    const end = seg.endSeconds + padSec;
    return {
      startSeconds: start,
      endSeconds: end,
      durationSeconds: end - start,
    };
  });

  // ── 4 & 5. Cap long segments, split with optional overlap ──
  const result: VadSpeechSegment[] = [];

  for (const seg of padded) {
    if (seg.durationSeconds <= maxSec) {
      result.push(seg);
    } else {
      // Split into maxSec-sized chunks with overlap
      let cursor = seg.startSeconds;
      while (cursor < seg.endSeconds) {
        const chunkEnd = Math.min(cursor + maxSec, seg.endSeconds);
        const chunkDuration = chunkEnd - cursor;

        // Only add if the chunk is meaningful (don't add trailing slivers)
        if (chunkDuration >= 0.1) { // 100ms minimum chunk
          result.push({
            startSeconds: cursor,
            endSeconds: chunkEnd,
            durationSeconds: chunkDuration,
          });
        }

        // Advance cursor with overlap
        cursor = overlapSec > 0 && chunkEnd < seg.endSeconds
          ? chunkEnd - overlapSec
          : chunkEnd;

        // Safety: ensure forward progress
        if (cursor <= chunkEnd - maxSec + overlapSec) {
          cursor = chunkEnd; // fallback to no-overlap to prevent infinite loop
        }
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// vadBinarize — convert VAD probabilities to binary speech/silence mask
// ---------------------------------------------------------------------------

export interface VadBinarizeOptions {
  /** Speech probability threshold (0-1). Default: 0.5 */
  readonly threshold?: number;
  /** Minimum consecutive speech hops to confirm onset (default: 5) */
  readonly minSpeechHops?: number;
  /** Minimum consecutive silence hops to confirm offset (default: 10) */
  readonly minSilenceHops?: number;
  /** Speech hangover in hops — extend speech end by this many hops (default: 5) */
  readonly hangoverHops?: number;
}

/**
 * Convert a VAD probability array into binary speech segments.
 *
 * Returns segments with start/end times. Uses hysteresis:
 *   - Speech onset: probability >= threshold for `minSpeechHops` consecutive hops
 *   - Speech offset: probability < threshold for `minSilenceHops` consecutive hops
 *   - Hangover: extend speech end by `hangoverHops` to avoid clipped words
 *
 * @param probabilities — VAD probability per hop (Float32Array, values 0-1)
 * @param hopDurationSec — duration of one hop in seconds (e.g., 0.032 for 512@16k)
 * @param options — binarization options
 */
export function vadBinarize(
  probabilities: Float32Array,
  hopDurationSec: number,
  options: VadBinarizeOptions = {},
): VadSpeechSegment[] {
  const threshold = options.threshold ?? 0.5;
  const minSpeechHops = options.minSpeechHops ?? 5;
  const minSilenceHops = options.minSilenceHops ?? 10;
  const hangoverHops = options.hangoverHops ?? 5;

  if (probabilities.length === 0) return [];

  const segments: VadSpeechSegment[] = [];
  let inSpeech = false;
  let speechStartHop = 0;
  let silenceRun = 0;
  let speechRun = 0;

  for (let i = 0; i < probabilities.length; i++) {
    const isSpeech = probabilities[i]! >= threshold;

    if (!inSpeech) {
      if (isSpeech) {
        speechRun++;
        if (speechRun >= minSpeechHops) {
          // Speech onset confirmed
          speechStartHop = i - speechRun + 1;
          inSpeech = true;
          silenceRun = 0;
        }
      } else {
        speechRun = 0;
      }
    } else {
      if (isSpeech) {
        silenceRun = 0;
      } else {
        silenceRun++;
        if (silenceRun >= minSilenceHops) {
          // Speech offset confirmed
          const speechEndHop = Math.min(
            probabilities.length,
            i - silenceRun + hangoverHops,
          );
          const startSec = speechStartHop * hopDurationSec;
          const endSec = speechEndHop * hopDurationSec;
          if (endSec - startSec > 0) {
            segments.push({
              startSeconds: startSec,
              endSeconds: endSec,
              durationSeconds: endSec - startSec,
            });
          }
          inSpeech = false;
          speechRun = 0;
        }
      }
    }
  }

  // Close trailing speech segment
  if (inSpeech) {
    const endHop = Math.min(probabilities.length, probabilities.length + hangoverHops);
    const startSec = speechStartHop * hopDurationSec;
    const endSec = endHop * hopDurationSec;
    if (endSec > startSec) {
      segments.push({
        startSeconds: startSec,
        endSeconds: endSec,
        durationSeconds: endSec - startSec,
      });
    }
  }

  return segments;
}

// ---------------------------------------------------------------------------
// noiseGate — suppress low-energy frames using VAD probabilities
// ---------------------------------------------------------------------------

export interface NoiseGateOptions {
  /** Energy threshold multiplier of noise floor. Default: 2.0 */
  readonly noiseFloorMultiplier?: number;
  /** Window size in samples for RMS calculation. Default: 512 */
  readonly windowSize?: number;
  /**
   * Attenuation factor for noise frames (0 = silence, 1 = pass-through).
   * Default: 0.1 (reduce noise by 20dB). Use 0.0 to completely silence.
   */
  readonly attenuation?: number;
  /**
   * Use smooth crossfade at window boundaries to avoid discontinuities.
   * Default: true. Prevents VAD from detecting artificial silence at boundaries.
   */
  readonly smoothEdges?: boolean;
}

/**
 * Apply a noise gate to audio using energy-based noise floor estimation.
 *
 * Computes RMS per window, estimates noise floor from quietest windows,
 * and attenuates windows below noiseFloor * noiseFloorMultiplier.
 *
 * This is a lightweight noise suppressor — it zeroes out frames that are
 * below the estimated noise floor. For spectral noise suppression, use a
 * dedicated denoiser (e.g., RNNoise, spectral gating).
 *
 * @returns gated audio (same length as input)
 */
export function noiseGate(
  audio: Float32Array,
  options: NoiseGateOptions = {},
): Float32Array {
  const multiplier = options.noiseFloorMultiplier ?? 2.0;
  const windowSize = options.windowSize ?? 512;
  const attenuation = options.attenuation ?? 0.1;
  const smoothEdges = options.smoothEdges !== false;

  if (audio.length === 0) return new Float32Array(0);

  const numWindows = Math.floor(audio.length / windowSize);
  if (numWindows === 0) return new Float32Array(audio); // too short

  // Compute per-window RMS
  const rmsValues = new Float32Array(numWindows);
  for (let w = 0; w < numWindows; w++) {
    const start = w * windowSize;
    const end = start + windowSize;
    let energy = 0;
    for (let j = start; j < end; j++) {
      energy += audio[j]! * audio[j]!;
    }
    rmsValues[w] = Math.sqrt(energy / windowSize);
  }

  // Estimate noise floor: median of the quietest 20% of windows
  const sorted = new Float32Array(rmsValues).sort();
  const noiseFloorIdx = Math.floor(numWindows * 0.2);
  const noiseFloor = sorted[noiseFloorIdx] ?? 0;

  const threshold = noiseFloor * multiplier;

  // Compute per-window gains with smoothing
  const rawGains = new Float32Array(numWindows);
  for (let w = 0; w < numWindows; w++) {
    const rms = rmsValues[w] ?? 0;
    rawGains[w] = rms >= threshold ? 1.0 : attenuation;
  }

  // Smooth gains across windows to avoid discontinuities
  const gains = smoothEdges ? smoothGains(rawGains) : rawGains;

  // Gate: apply smoothed gains sample-by-sample with crossfade
  const output = new Float32Array(audio.length);
  for (let w = 0; w < numWindows; w++) {
    const start = w * windowSize;
    const end = Math.min(start + windowSize, audio.length);
    const gain = gains[w] ?? attenuation;

    if (smoothEdges && w > 0 && w < numWindows - 1) {
      const prevGain = gains[w - 1] ?? gain;
      const nextGain = gains[w + 1] ?? gain;
      const halfWin = Math.floor(windowSize / 2);
      for (let j = start; j < end; j++) {
        const localPos = j - start;
        // Crossfade from prevGain to gain in first half, gain to nextGain in second half
        const localGain = localPos < halfWin
          ? prevGain + (gain - prevGain) * (localPos / halfWin)
          : gain + (nextGain - gain) * ((localPos - halfWin) / (windowSize - halfWin));
        output[j] = audio[j]! * localGain;
      }
    } else {
      for (let j = start; j < end; j++) {
        output[j] = audio[j]! * gain;
      }
    }
  }

  // Copy trailing samples (if audio.length not divisible by windowSize)
  const fullWindows = numWindows * windowSize;
  for (let j = fullWindows; j < audio.length; j++) {
    output[j] = audio[j]! * attenuation; // trailing silence → attenuate
  }

  return output;
}

/**
 * Smooth a gain array with a 3-point moving average to avoid
 * discontinuities between adjacent windows.
 */
function smoothGains(gains: Float32Array): Float32Array {
  const smoothed = new Float32Array(gains.length);
  if (gains.length <= 2) return new Float32Array(gains);

  smoothed[0] = (gains[0]! * 2 + gains[1]!) / 3;
  for (let i = 1; i < gains.length - 1; i++) {
    smoothed[i] = (gains[i - 1]! + gains[i]! + gains[i + 1]!) / 3;
  }
  smoothed[gains.length - 1] =
    (gains[gains.length - 2]! + gains[gains.length - 1]! * 2) / 3;

  return smoothed;
}

// ---------------------------------------------------------------------------
// segmentAudio — full pre-processing pipeline
// ---------------------------------------------------------------------------

export interface SegmentAudioOptions {
  /** VAD backend to use */
  readonly vad: WhisperVadBackend;
  /** Audio sample rate (default: 16000) */
  readonly sampleRate?: number;
  /** VAD speech threshold (default: 0.5) */
  readonly threshold?: number;
  /** Merge configuration */
  readonly merge?: VadMergeConfig;
  /** Apply noise gate before segmentation (default: false — opt-in) */
  readonly noiseGate?: boolean | NoiseGateOptions;
}

/**
 * Full preprocessing pipeline: VAD → binarize → merge → split → overlap.
 *
 * This is the recommended entry point for ASR pre-segmentation.
 * It handles the entire WhisperX-style VAD pipeline in one call.
 */
export async function segmentAudio(
  audio: Float32Array,
  options: SegmentAudioOptions,
): Promise<VadSpeechSegment[]> {
  const sampleRate = options.sampleRate ?? 16_000;
  const threshold = options.threshold ?? DEFAULT_VAD_ONSET;

  // Optional noise gating (opt-in)
  const gatedAudio = typeof options.noiseGate === 'object'
    ? noiseGate(audio, options.noiseGate)
    : options.noiseGate === true
      ? noiseGate(audio)
      : audio;

  // VAD segmentation
  const rawSegments = await options.vad.segment(gatedAudio, sampleRate, threshold);

  // Merge + pad + cap + overlap
  const merge = options.merge ?? {};
  return mergeVadSegments(
    rawSegments,
    merge.minSilenceDurationMs ?? DEFAULT_MIN_SILENCE_MS,
    merge.speechPadMs ?? DEFAULT_SPEECH_PAD_MS,
    merge.maxSegmentDurationMs ?? DEFAULT_MAX_SEGMENT_MS,
    merge.minSpeechDurationMs ?? DEFAULT_MIN_SPEECH_MS,
    merge.overlapDurationMs ?? DEFAULT_OVERLAP_MS,
  );
}
