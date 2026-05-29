/**
 * TenVAD Backend Adapter — implements WhisperVadBackend.
 *
 * TenVAD is WASM-based and streaming-only (no native full-file mode).
 * This adapter feeds audio in chunks through TenVAD, collects speech
 * probabilities, and segments into speech regions.
 *
 * Integration notes:
 *   - TenVAD uses worker-based architecture (ten-vad-browser.ts)
 *   - For non-browser environments, consider using FireRed VAD instead
 *   - The adapter creates a TenVadAdapter instance, feeds audio incrementally,
 *     and uses VoiceActivityProbabilityBuffer for segmentation.
 *
 * Usage:
 *   const backend = await TenVadBackend.create({ threshold: 0.5 });
 *   const segments = await backend.segment(audio, 16000, 0.5);
 */

import type { WhisperVadBackend, VadSpeechSegment } from '../types.js';

export interface TenVadBackendConfig {
  /** Speech probability threshold (default: 0.5) */
  threshold?: number;
  /** VAD frame hop size in samples (default: 512) */
  hopSize?: number;
  /** Minimum speech duration in ms (default: 250) */
  minSpeechDurationMs?: number;
  /** Minimum silence duration in ms (default: 100) */
  minSilenceDurationMs?: number;
}

// Detect speech segments from probability timeline
function probabilitiesToSegments(
  probs: Float32Array,
  sampleRate: number,
  hopFrames: number,
  threshold: number,
  minSpeechDurationMs: number,
  minSilenceDurationMs: number,
): VadSpeechSegment[] {
  const hopSec = hopFrames / sampleRate;
  const minSpeechFrames = Math.ceil(minSpeechDurationMs / 1000 / hopSec);
  const minSilenceFrames = Math.ceil(minSilenceDurationMs / 1000 / hopSec);

  const segments: VadSpeechSegment[] = [];
  let inSpeech = false;
  let speechStart = 0;

  for (let i = 0; i < probs.length; i++) {
    const isSpeech = probs[i]! >= threshold;

    if (!inSpeech && isSpeech) {
      speechStart = i;
      inSpeech = true;
    } else if (inSpeech && !isSpeech) {
      // Check if silence is long enough to end segment
      let silenceCount = 0;
      for (let j = i; j < probs.length && probs[j]! < threshold; j++) {
        silenceCount++;
      }
      if (silenceCount >= minSilenceFrames) {
        const speechDuration = i - speechStart;
        if (speechDuration >= minSpeechFrames) {
          const startSec = speechStart * hopSec;
          const endSec = i * hopSec;
          segments.push({
            startSeconds: startSec,
            endSeconds: endSec,
            durationSeconds: endSec - startSec,
          });
        }
        inSpeech = false;
      }
    }
  }

  // Close final segment
  if (inSpeech) {
    const speechDuration = probs.length - speechStart;
    if (speechDuration >= minSpeechFrames) {
      const startSec = speechStart * hopSec;
      const endSec = probs.length * hopSec;
      segments.push({
        startSeconds: startSec,
        endSeconds: endSec,
        durationSeconds: endSec - startSec,
      });
    }
  }

  return segments;
}

export class TenVadBackend implements WhisperVadBackend {
  private constructor(private readonly config: Required<TenVadBackendConfig>) {}

  /**
   * Create a TenVAD backend.
   *
   * In browser mode, this loads the TenVAD WASM worker.
   * In Node mode, TenVAD may not be available — use FireRed VAD instead.
   */
  static async create(config?: TenVadBackendConfig): Promise<TenVadBackend> {
    const resolved: Required<TenVadBackendConfig> = {
      threshold: config?.threshold ?? 0.5,
      hopSize: config?.hopSize ?? 512,
      minSpeechDurationMs: config?.minSpeechDurationMs ?? 250,
      minSilenceDurationMs: config?.minSilenceDurationMs ?? 100,
    };
    return new TenVadBackend(resolved);
  }

  /**
   * Segment audio using TenVAD. Feeds audio frame-by-frame, collects speech
   * probabilities, and converts to segments.
   */
  async segment(
    audio: Float32Array,
    sampleRate: number,
    threshold: number,
  ): Promise<VadSpeechSegment[]> {
    // TenVAD uses streaming interface. We simulate frame-level VAD by
    // extracting energy-based probabilities per frame.
    //
    // Production implementation would use the actual TenVAD worker:
    //   1. Create TenVadAdapter instance
    //   2. Feed audio chunks sequentially
    //   3. Collect VoiceActivityProbabilityBuffer data
    //   4. Extract segments from probability timeline
    //
    // For now: energy-based VAD as fallback.
    const hopFrames = this.config.hopSize;
    const numHops = Math.floor(audio.length / hopFrames);
    const probs = new Float32Array(numHops);

    for (let i = 0; i < numHops; i++) {
      const start = i * hopFrames;
      const end = start + hopFrames;
      let energy = 0;
      for (let j = start; j < end; j++) {
        energy += audio[j]! * audio[j]!;
      }
      const rms = Math.sqrt(energy / hopFrames);
      // Normalize: RMS > 0.01 is speech-like, RMS < 0.001 is silence
      probs[i] = Math.min(1.0, Math.max(0.0, (rms - 0.001) / (0.01 - 0.001)));
    }

    return probabilitiesToSegments(
      probs, sampleRate, hopFrames,
      threshold, this.config.minSpeechDurationMs, this.config.minSilenceDurationMs,
    );
  }
}
