/**
 * FireRed VAD Backend Adapter — implements WhisperVadBackend.
 *
 * FireRed VAD has a native full-file mode (`detect()` method) that returns
 * speech segment timestamps directly. This adapter wraps it into the
 * chunking module's WhisperVadBackend interface.
 *
 * Usage:
 *   const backend = await FireRedVadBackend.create('/path/to/model');
 *   const segments = await backend.segment(audio, 16000, 0.5);
 */

import type { WhisperVadBackend, VadSpeechSegment } from '../types.js';

export class FireRedVadBackend implements WhisperVadBackend {
  private constructor(private readonly vad: any) {}

  /**
   * Create a FireRed VAD backend from a pretrained model directory.
   * The model directory should contain ONNX model + CMVN + config.
   *
   * @param modelDir — path to pretrained FireRed VAD model
   * @param config — VAD config overrides (optional)
   */
  static async create(modelDir?: string, config?: Record<string, unknown>): Promise<FireRedVadBackend> {
    // Dynamic import to avoid hard dependency on runtime VAD module
    const { FireRedVad } = await import('../../runtime/firered-vad/api/classes.js');
    const vad = await FireRedVad.from_pretrained(modelDir, config);
    return new FireRedVadBackend(vad);
  }

  /**
   * Segment audio into speech regions using FireRed VAD.
   *
   * @param audio — PCM 16-bit-equivalent float audio samples
   * @param sampleRate — audio sample rate (typically 16000)
   * @param threshold — speech probability threshold (0-1)
   */
  async segment(
    audio: Float32Array,
    sampleRate: number,
    _threshold: number,
  ): Promise<VadSpeechSegment[]> {
    // FireRed expects { pcm16, sampleRate } format
    const [result] = await this.vad.detect({ pcm16: audio, sampleRate }, true);

    if (!result || !result.timestamps) return [];

    return result.timestamps.map((ts: { start: number; end: number }) => ({
      startSeconds: ts.start,
      endSeconds: ts.end,
      durationSeconds: ts.end - ts.start,
    }));
  }
}
