/**
 * Canonical streaming transcriber for the chunked Moonshine session.
 *
 * The chunked design decodes whole accumulated hidden states rather than
 * emitting token-by-token stream updates, so this wrapper maps it onto the
 * core StreamingTranscriber contract as follows:
 *
 * - `pushAudio` buffers audio (and emits a repeatable preview decode when
 *   `emitPartials` is on); `previewText` always mirrors the latest decode.
 * - `committedText` stays empty until `finalize`, which flushes the tail
 *   through the whole-utterance encoder and publishes the final transcript
 *   as both committed and preview text.
 * - Revisions increase with every published update; `reset()` starts a new
 *   stream on the same session.
 *
 * Input audio must already be 16 kHz mono PCM: this wrapper never resamples
 * silently (a mismatched rate is a caller error, reported with the received
 * rate). Canonical meta carries the model identity so downstream consumers
 * never need to know which family produced the transcript.
 */
import { normalizePcmInput } from '../../audio/audio.js';
import type { AudioInputLike } from '../../types/audio.js';
import type { PartialTranscript, TranscriptDetailLevel } from '../../types/transcript.js';
import type { StreamingTranscriber, StreamingTranscriberState } from '../../types/streaming.js';
import type { MoonshineStreamingText } from './streaming-session.js';

const TARGET_SAMPLE_RATE = 16000;

/** The subset of MoonshineStreamingSession this wrapper drives. */
export interface MoonshineStreamingSessionLike {
  readonly committedEncoderFrames: number;
  pushAudio(pcm: Float32Array): Promise<number>;
  preview(): Promise<MoonshineStreamingText>;
  finalize(): Promise<MoonshineStreamingText>;
  reset(): void;
}

export interface MoonshineStreamingTranscriberOptions {
  readonly modelId?: string;
  readonly modelFamily?: string;
  readonly detailLevel?: TranscriptDetailLevel;
  /** Decode a preview after each pushAudio; disable for lower CPU cost. */
  readonly emitPartials?: boolean;
}

export class MoonshineStreamingTranscriber implements StreamingTranscriber {
  private revision = 0;
  private bufferedDurationSeconds = 0;
  private committedText = '';
  private previewText = '';
  private previewTokens: number | undefined;
  private isFinalized = false;
  private isDisposed = false;

  constructor(
    private readonly session: MoonshineStreamingSessionLike,
    private readonly options: MoonshineStreamingTranscriberOptions = {},
  ) {}

  async pushAudio(input: AudioInputLike): Promise<PartialTranscript> {
    this.assertUsable();
    const audio = normalizePcmInput(input);
    if (audio.sampleRate !== TARGET_SAMPLE_RATE) {
      throw new Error(
        `Moonshine streaming requires ${TARGET_SAMPLE_RATE} Hz mono PCM; received ${audio.sampleRate} Hz. Resample before pushing.`,
      );
    }
    const samples = audio.channels[0] ?? new Float32Array(0);
    this.bufferedDurationSeconds += samples.length / TARGET_SAMPLE_RATE;
    await this.session.pushAudio(Float32Array.from(samples));
    if (this.options.emitPartials ?? true) {
      await this.refreshPreview();
    }
    return this.partial('partial');
  }

  async flush(): Promise<PartialTranscript> {
    this.assertUsable();
    await this.refreshPreview();
    return this.partial('partial');
  }

  async finalize(): Promise<PartialTranscript> {
    this.assertUsable();
    const result = await this.session.finalize();
    this.committedText = result.text;
    this.previewText = result.text;
    this.previewTokens = result.tokenIds.length;
    this.isFinalized = true;
    return this.partial('final');
  }

  reset(): void {
    if (this.isDisposed) {
      throw new Error('Moonshine streaming transcriber is disposed.');
    }
    this.session.reset();
    this.revision += 1;
    this.bufferedDurationSeconds = 0;
    this.committedText = '';
    this.previewText = '';
    this.previewTokens = undefined;
    this.isFinalized = false;
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    try {
      this.session.reset();
    } finally {
      this.isDisposed = true;
    }
  }

  getState(): StreamingTranscriberState {
    return {
      revision: this.revision,
      bufferedDurationSeconds: this.bufferedDurationSeconds,
      committedText: this.committedText,
      previewText: this.previewText,
      isFinalized: this.isFinalized,
    };
  }

  private async refreshPreview(): Promise<void> {
    const preview = await this.session.preview();
    this.previewText = preview.text;
    this.previewTokens = preview.tokenIds.length;
  }

  private partial(kind: 'partial' | 'final'): PartialTranscript {
    this.revision += 1;
    return {
      kind,
      revision: this.revision,
      text: this.previewText,
      committedText: this.committedText,
      previewText: this.previewText,
      warnings: [],
      meta: {
        detailLevel: this.options.detailLevel ?? 'segments',
        isFinal: kind === 'final',
        modelFamily: this.options.modelFamily ?? 'moonshine-streaming',
        modelId: this.options.modelId ?? 'moonshine-streaming-tiny',
        sampleRate: TARGET_SAMPLE_RATE,
        durationSeconds: this.bufferedDurationSeconds,
        ...(this.previewTokens === undefined ? {} : { tokenCount: this.previewTokens }),
      },
    };
  }

  private assertUsable(): void {
    if (this.isDisposed) {
      throw new Error('Moonshine streaming transcriber is disposed.');
    }
    if (this.isFinalized) {
      throw new Error('Streaming transcriber is finalized. Call reset() before pushing new audio.');
    }
  }
}
