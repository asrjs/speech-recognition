/**
 * Drift Handler — whisper.cpp-style seek counter for long audio.
 * Model-agnostic. Prevents cumulative timestamp drift in multi-chunk transcription.
 */

import type { DriftCorrectionResult } from './types.js';

export class DriftHandler {
  private seekSamples: number = 0;

  reset(_audioLengthSamples: number): void {
    this.seekSamples = 0;
  }

  getSeekSeconds(sampleRate: number): number {
    return this.seekSamples / sampleRate;
  }

  advanceBy(durationSeconds: number, sampleRate: number): void {
    this.seekSamples += Math.round(durationSeconds * sampleRate);
  }

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

    return { start: modelStartSec, end: modelEndSec, corrected: false };
  }
}
