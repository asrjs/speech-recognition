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
 */

import type { VadSpeechSegment, WhisperVadBackend } from './types.js';

export type { VadSpeechSegment, WhisperVadBackend };

/**
 * Merge, pad, and cap VAD segments for ASR processing.
 *
 *   1. Filter segments shorter than minSpeechDurationMs
 *   2. Merge segments with gaps < minSilenceDurationMs
 *   3. Pad each segment by speechPadMs on both sides
 *   4. Clamp padded segments to [0, audioEnd]
 *   5. Split segments longer than maxSegmentDurationMs
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

  const filtered = segments.filter((s) => s.durationSeconds >= minSec);
  if (filtered.length === 0) return [];

  const merged: VadSpeechSegment[] = [];
  let current = { ...filtered[0]! };

  for (let i = 1; i < filtered.length; i++) {
    const next = filtered[i]!;
    const gap = next.startSeconds - current.endSeconds;
    if (gap < minSilenceSec) {
      current = { startSeconds: current.startSeconds, endSeconds: next.endSeconds, durationSeconds: next.endSeconds - current.startSeconds };
    } else {
      merged.push(current);
      current = { ...next };
    }
  }
  merged.push(current);

  const padded = merged.map((seg) => ({
    startSeconds: Math.max(0, seg.startSeconds - padSec),
    endSeconds: seg.endSeconds + padSec,
    durationSeconds: 0,
  }));

  const result: VadSpeechSegment[] = [];
  for (const seg of padded) {
    const totalDuration = seg.endSeconds - seg.startSeconds;
    if (totalDuration <= maxSec) {
      result.push({ startSeconds: seg.startSeconds, endSeconds: seg.endSeconds, durationSeconds: totalDuration });
    } else {
      let cursor = seg.startSeconds;
      while (cursor < seg.endSeconds) {
        const chunkEnd = Math.min(cursor + maxSec, seg.endSeconds);
        result.push({ startSeconds: cursor, endSeconds: chunkEnd, durationSeconds: chunkEnd - cursor });
        cursor = chunkEnd;
      }
    }
  }

  return result;
}
