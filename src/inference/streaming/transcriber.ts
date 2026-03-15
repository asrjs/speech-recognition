import { normalizePcmInput } from '../../audio/index.js';
import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  PartialTranscript,
  SpeechSession,
  StreamingSessionOptions,
  StreamingTranscriber,
  StreamingTranscriberState,
  TranscriptDetailLevel,
  TranscriptResult,
  VoiceActivityDetector,
} from '../../types/index.js';
import { TranscriptAccumulator } from './accumulator.js';
import { RollingAudioWindow } from './rolling-window.js';

export interface DefaultStreamingTranscriberOptions extends StreamingSessionOptions {
  readonly sampleRate?: number;
}

export class DefaultStreamingTranscriber<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> implements StreamingTranscriber {
  private readonly window: RollingAudioWindow;
  private readonly accumulator = new TranscriptAccumulator();
  private readonly vad?: VoiceActivityDetector;
  private readonly detail: TranscriptDetailLevel;
  private readonly emitPartials: boolean;
  private readonly minFinalSilenceMs: number;
  private isFinalized = false;
  private totalDurationSeconds = 0;
  private heardSpeech = false;

  constructor(
    private readonly session: SpeechSession<TOptions, TNative>,
    options: DefaultStreamingTranscriberOptions = {},
  ) {
    this.window = new RollingAudioWindow({
      maxWindowMs: options.maxWindowMs,
      overlapMs: options.overlapMs,
    });
    this.vad = options.vad;
    this.detail = options.detail ?? 'segments';
    this.emitPartials = options.emitPartials ?? true;
    this.minFinalSilenceMs = options.minFinalSilenceMs ?? 350;
  }

  async pushAudio(input: AudioInputLike): Promise<PartialTranscript> {
    this.assertNotFinalized();

    const normalized = normalizePcmInput(input);
    const chunkStartTime = this.totalDurationSeconds;
    this.totalDurationSeconds += normalized.durationSeconds;
    this.window.push(normalized, chunkStartTime);

    if (this.vad) {
      const event = await this.vad.analyze(normalized);
      this.heardSpeech ||= event.isSpeech;

      if (
        !event.isSpeech &&
        this.heardSpeech &&
        normalized.durationSeconds * 1000 >= this.minFinalSilenceMs
      ) {
        return this.finalize();
      }
    }

    if (!this.emitPartials) {
      return this.accumulator.update(this.blankResult(), 'partial');
    }

    return this.transcribeBuffered('partial');
  }

  async flush(): Promise<PartialTranscript> {
    this.assertNotFinalized();
    return this.transcribeBuffered('partial');
  }

  async finalize(): Promise<PartialTranscript> {
    this.assertNotFinalized();
    const update = await this.transcribeBuffered('final');
    this.isFinalized = true;
    return update;
  }

  async reset(): Promise<void> {
    this.window.reset();
    this.accumulator.reset();
    this.totalDurationSeconds = 0;
    this.heardSpeech = false;
    this.isFinalized = false;
    await this.session.dispose();
  }

  getState(): StreamingTranscriberState {
    const state = this.accumulator.getState();
    return {
      revision: state.revision,
      bufferedDurationSeconds: this.window.getBufferedDurationSeconds(),
      committedText: state.committedText,
      previewText: state.previewText,
      isFinalized: this.isFinalized,
    };
  }

  private async transcribeBuffered(kind: 'partial' | 'final'): Promise<PartialTranscript> {
    const audio = this.window.toPcmAudioBuffer();
    const canonical =
      audio.numberOfFrames > 0 ? await this.canonicalTranscribe(audio) : this.blankResult();

    return this.accumulator.update(canonical, kind);
  }

  private async canonicalTranscribe(input: AudioInputLike): Promise<TranscriptResult> {
    return this.session.transcribe(input, {
      detail: this.detail,
      responseFlavor: 'canonical',
    } as TOptions & { readonly responseFlavor: 'canonical' });
  }

  private blankResult(): TranscriptResult {
    return {
      text: '',
      warnings: [],
      meta: {
        detailLevel: this.detail,
        isFinal: false,
        durationSeconds: this.window.getBufferedDurationSeconds(),
      },
    };
  }

  private assertNotFinalized(): void {
    if (this.isFinalized) {
      throw new Error('Streaming transcriber is finalized. Call reset() before pushing new audio.');
    }
  }
}
