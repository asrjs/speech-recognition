/**
 * Fixed Window Chunker — sliding window for long audio.
 *
 * For models without VAD, splits audio into overlapping 30s windows.
 * Default: 30s window, 28s hop, 2s overlap.
 *
 * Model-agnostic. Pure function. No model dependencies.
 */

export interface FixedWindowConfig {
  /** Window duration in ms (default: 30000 = 30s) */
  windowDurationMs?: number;
  /** Hop duration between windows in ms (default: 28000 = 28s) */
  hopDurationMs?: number;
}

export interface AudioWindow {
  /** Audio data for this window */
  readonly audio: Float32Array;
  /** Start time in seconds (absolute, from original audio) */
  readonly startSeconds: number;
  /** End time in seconds (absolute) */
  readonly endSeconds: number;
  /** Duration in seconds */
  readonly durationSeconds: number;
}

export class FixedWindowChunker {
  private readonly config: Required<FixedWindowConfig>;

  constructor(config: FixedWindowConfig = {}) {
    this.config = {
      windowDurationMs: config.windowDurationMs ?? 30000,
      hopDurationMs: config.hopDurationMs ?? 28000,
    };
  }

  /**
   * Split audio into sliding windows.
   * Returns windows that cover the full audio.
   */
  chunk(audio: Float32Array, sampleRate: number): AudioWindow[] {
    const windowSamples = Math.floor((this.config.windowDurationMs / 1000) * sampleRate);
    const hopSamples = Math.floor((this.config.hopDurationMs / 1000) * sampleRate);

    if (audio.length <= windowSamples) {
      return [{
        audio,
        startSeconds: 0,
        endSeconds: audio.length / sampleRate,
        durationSeconds: audio.length / sampleRate,
      }];
    }

    const windows: AudioWindow[] = [];
    let start = 0;

    while (start < audio.length) {
      const end = Math.min(start + windowSamples, audio.length);
      const window = audio.subarray(start, end);
      windows.push({
        audio: new Float32Array(window),
        startSeconds: start / sampleRate,
        endSeconds: end / sampleRate,
        durationSeconds: (end - start) / sampleRate,
      });
      start += hopSamples;
    }

    return windows;
  }
}
