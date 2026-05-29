/**
 * Drift Handler — whisper.cpp-style seek counter for long audio.
 *
 * Problem: Whisper processes audio in fixed windows. After processing a 30s chunk,
 * the model's start/end timestamps should be 0-30s for the first chunk, 30-60s for
 * the second, etc. But due to audio padding, VAD segmentation, or model inaccuracy,
 * the model's timestamps can drift from the true audio position.
 *
 * Solution (from whisper.cpp):
 *   1. Maintain an external seek counter tracking absolute audio position.
 *   2. After each chunk, advance the seek counter by the WHISPER-reported duration.
 *   3. When returning timestamps, check if the model's start time deviates from seek.
 *   4. If drift > maxDriftSec (default 1.0s), use seek position instead.
 *
 * This prevents cumulative timestamp errors in very long recordings.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DriftCorrectionResult {
  readonly start: number;
  readonly end: number;
  /** True if timestamps were corrected to use seek position. */
  readonly corrected: boolean;
}

// ---------------------------------------------------------------------------
// DriftHandler
// ---------------------------------------------------------------------------

export class DriftHandler {
  /** External seek counter in sample units. */
  private seekSamples: number = 0;

  /**
   * Reset the seek counter for a new transcription session.
   * @param audioLength Samples — not used by handler, provided for logging context.
   */
  reset(_audioLengthSamples: number): void {
    this.seekSamples = 0;
  }

  /**
   * Get current seek position in seconds.
   */
  getSeekSeconds(sampleRate: number): number {
    return this.seekSamples / sampleRate;
  }

  /**
   * Advance the seek counter by a processed duration.
   * Called after each chunk is successfully transcribed.
   */
  advanceBy(durationSeconds: number, sampleRate: number): void {
    this.seekSamples += Math.round(durationSeconds * sampleRate);
  }

  /**
   * Correct model-reported timestamps if they have drifted from seek.
   *
   * Algorithm (matches whisper.cpp):
   *   1. Compute model-reported duration: end - start
   *   2. Check drift: |model_start - seek| > maxDriftSec ?
   *   3. If drift detected: return (seek, seek + duration) with corrected=true
   *   4. Otherwise: return (start, end) with corrected=false
   *
   * @param modelStartSec — model's reported segment start in seconds
   * @param modelEndSec — model's reported segment end in seconds
   * @param sampleRate — audio sample rate
   * @param maxDriftSec — max allowed drift before correction (default: 1.0)
   */
  correctTimestamps(
    modelStartSec: number,
    modelEndSec: number,
    sampleRate: number,
    maxDriftSec: number = 1.0,
  ): DriftCorrectionResult {
    const seekSec = this.getSeekSeconds(sampleRate);
    const modelDuration = Math.max(0, modelEndSec - modelStartSec);
    const drift = Math.abs(modelStartSec - seekSec);

    if (drift > maxDriftSec) {
      return {
        start: seekSec,
        end: seekSec + modelDuration,
        corrected: true,
      };
    }

    return {
      start: modelStartSec,
      end: modelEndSec,
      corrected: false,
    };
  }
}
