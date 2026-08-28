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
import { joinTranscriptFragments } from './merge.js';
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
  private stateGeneration = 0;
  private operationAbortController = new AbortController();
  private operationTail: Promise<void> = Promise.resolve();
  private disposed = false;

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
    const generation = this.stateGeneration;
    return this.enqueue(() => this.pushAudioInternal(input, generation));
  }

  private async pushAudioInternal(
    input: AudioInputLike,
    generation: number,
  ): Promise<PartialTranscript> {
    this.assertNotFinalized();
    if (generation !== this.stateGeneration) return this.staleUpdate();

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
        return this.finalizeInternal(generation);
      }
    }

    if (generation !== this.stateGeneration) return this.staleUpdate();

    if (!this.emitPartials) {
      return this.accumulator.update(this.blankResult(), 'partial');
    }

    return this.transcribeBuffered('partial', generation);
  }

  async flush(): Promise<PartialTranscript> {
    const generation = this.stateGeneration;
    return this.enqueue(() => {
      this.assertNotFinalized();
      return this.transcribeBuffered('partial', generation);
    });
  }

  async finalize(): Promise<PartialTranscript> {
    const generation = this.stateGeneration;
    return this.enqueue(() => this.finalizeInternal(generation));
  }

  /** Reset transcript/window state while retaining the injected session. */
  async reset(): Promise<void> {
    this.assertNotDisposed();
    this.stateGeneration += 1;
    this.operationAbortController.abort();
    this.operationAbortController = new AbortController();
    this.window.reset();
    this.accumulator.reset();
    this.totalDurationSeconds = 0;
    this.heardSpeech = false;
    this.isFinalized = false;
  }

  /** Release the injected session; repeated disposal is harmless. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.stateGeneration += 1;
    this.operationAbortController.abort();
    await this.operationTail;
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

  private async transcribeBuffered(
    kind: 'partial' | 'final',
    generation: number,
  ): Promise<PartialTranscript> {
    const audio = this.window.toPcmAudioBuffer();
    const signal = this.operationAbortController.signal;
    let canonical: TranscriptResult;
    try {
      canonical =
        audio.numberOfFrames > 0
          ? await this.canonicalTranscribe(audio, signal)
          : this.blankResult();
    } catch (error) {
      if (generation !== this.stateGeneration || signal.aborted) {
        return this.staleUpdate();
      }
      throw error;
    }

    if (generation !== this.stateGeneration) return this.staleUpdate();

    return this.accumulator.update(canonical, kind);
  }

  private async finalizeInternal(generation: number): Promise<PartialTranscript> {
    this.assertNotFinalized();
    const update = await this.transcribeBuffered('final', generation);
    if (generation !== this.stateGeneration) return update;
    this.isFinalized = true;
    return update;
  }

  private async canonicalTranscribe(
    input: AudioInputLike,
    signal: AbortSignal,
  ): Promise<TranscriptResult> {
    return this.session.transcribe(input, {
      detail: this.detail,
      responseFlavor: 'canonical',
      signal,
    } as unknown as TOptions & { readonly responseFlavor: 'canonical' });
  }

  private staleUpdate(): PartialTranscript {
    const state = this.accumulator.getState();
    return {
      kind: 'partial',
      revision: state.revision,
      text: joinTranscriptFragments(state.committedText, state.previewText),
      committedText: state.committedText,
      previewText: state.previewText,
      warnings: [],
      meta: {
        detailLevel: this.detail,
        isFinal: false,
        durationSeconds: this.window.getBufferedDurationSeconds(),
      },
    };
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
    this.assertNotDisposed();
    if (this.isFinalized) {
      throw new Error('Streaming transcriber is finalized. Call reset() before pushing new audio.');
    }
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error('Streaming transcriber is disposed.');
    }
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const run = this.operationTail.then(operation, operation);
    this.operationTail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }
}
