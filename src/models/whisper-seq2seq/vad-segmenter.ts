/**
 * VAD Segmenter — audio pre-segmentation for Whisper chunking.
 *
 * Uses existing VAD backends (TenVAD or FireRed) in the project to detect
 * speech regions in long audio, then feeds each segment to Whisper independently.
 *
 * This replaces the simple 30s fixed-window chunking with intelligent
 * speech-aware segmentation — matching the strategy used in faster-whisper
 * and WhisperX.
 *
 * Architecture: Dependency injection via WhisperVadBackend interface.
 * The actual backend adapter is wired in enhanced-executor.ts (Phase 8).
 *
 * Backends already in the project:
 *   - TenVAD: WASM-based, fast, bundled src/runtime/ten-vad-browser.ts
 *   - FireRed VAD: ONNX-based, streaming + file-mode src/runtime/firered-vad/
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  /**
   * Segment audio into speech regions.
   * @param audio — PCM 16-bit-equivalent float audio samples
   * @param sampleRate — audio sample rate (typically 16000)
   * @param threshold — speech probability threshold (0-1)
   */
  segment(
    audio: Float32Array,
    sampleRate: number,
    threshold: number,
  ): Promise<VadSpeechSegment[]>;
}

// ---------------------------------------------------------------------------
// Segment Merging
// ---------------------------------------------------------------------------

/**
 * Merge, pad, and cap VAD segments for Whisper processing.
 *
 * Algorithm:
 *   1. Filter segments shorter than minSpeechDurationMs
 *   2. Merge segments with gaps < minSilenceDurationMs
 *   3. Pad each segment by speechPadMs on both sides
 *   4. Clamp padded segments to [0, audioEnd]
 *   5. Split segments longer than maxSegmentDurationMs
 *
 * @param segments — raw VAD segments
 * @param minSilenceDurationMs — merge segments with gap < this (default: 100)
 * @param speechPadMs — padding on each side of speech (default: 400)
 * @param maxSegmentDurationMs — cap at ~29s for Whisper's 30s window (default: 29000)
 * @param minSpeechDurationMs — filter segments shorter than this, applied per raw segment (default: 250)
 */
export function mergeVadSegments(
  segments: readonly VadSpeechSegment[],
  minSilenceDurationMs: number = 100,
  speechPadMs: number = 400,
  maxSegmentDurationMs: number = 29000,
  minSpeechDurationMs: number = 250,
): VadSpeechSegment[] {
  if (segments.length === 0) return [];

  const minSilenceSec = minSilenceDurationMs / 1000;
  const padSec = speechPadMs / 1000;
  const maxSec = maxSegmentDurationMs / 1000;
  const minSec = minSpeechDurationMs / 1000;

  // 1. Filter too-short segments
  const filtered = segments.filter((s) => s.durationSeconds >= minSec);
  if (filtered.length === 0) return [];

  // 2. Merge close segments
  const merged: VadSpeechSegment[] = [];
  let current = { ...filtered[0]! };

  for (let i = 1; i < filtered.length; i++) {
    const next = filtered[i]!;
    const gap = next.startSeconds - current.endSeconds;

    if (gap < minSilenceSec) {
      // Merge: extend current
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

  // 3-4. Pad and clamp
  const padded = merged.map((seg) => ({
    startSeconds: Math.max(0, seg.startSeconds - padSec),
    endSeconds: seg.endSeconds + padSec,
    durationSeconds: 0, // computed after
  }));

  // 5. Split long segments
  const result: VadSpeechSegment[] = [];
  for (const seg of padded) {
    const totalDuration = seg.endSeconds - seg.startSeconds;
    if (totalDuration <= maxSec) {
      result.push({
        startSeconds: seg.startSeconds,
        endSeconds: seg.endSeconds,
        durationSeconds: totalDuration,
      });
    } else {
      // Split into chunks of maxSec
      let cursor = seg.startSeconds;
      while (cursor < seg.endSeconds) {
        const chunkEnd = Math.min(cursor + maxSec, seg.endSeconds);
        result.push({
          startSeconds: cursor,
          endSeconds: chunkEnd,
          durationSeconds: chunkEnd - cursor,
        });
        cursor = chunkEnd;
      }
    }
  }

  return result;
}
