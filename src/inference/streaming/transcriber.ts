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

function normalizeWordKey(word: string): string {
  return word.toLowerCase().replace(/^[^\w\s]+|[^\w\s]+$/g, '').trim();
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
  private consecutiveSilenceSeconds = 0;
  private stateGeneration = 0;
  private operationAbortController = new AbortController();
  private operationTail: Promise<void> = Promise.resolve();
  private disposed = false;
  private disposePromise?: Promise<void>;

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

    if (this.vad) {
      const event = await this.vad.analyze(normalized);
      if (event.isSpeech) {
        this.heardSpeech = true;
        this.consecutiveSilenceSeconds = 0;
      } else if (this.heardSpeech) {
        this.consecutiveSilenceSeconds += normalized.durationSeconds;
        if (this.consecutiveSilenceSeconds * 1000 >= this.minFinalSilenceMs) {
          this.consecutiveSilenceSeconds = 0;
          return this.finalizeInternal(generation);
        }
      }
    }

    if (generation !== this.stateGeneration) return this.staleUpdate();

    // Eviction-Before-Drop Rule: when audio is about to be evicted from the window,
    // transcribe the current audio and seal words/tokens before the freeze boundary.
    if (this.window.wouldEvict(normalized.durationSeconds)) {
      await this.commitBeforeEviction(generation);
    }

    if (generation !== this.stateGeneration) return this.staleUpdate();

    this.window.push(normalized, chunkStartTime);

    if (!this.emitPartials) {
      return this.accumulator.update(this.blankResult(), 'partial');
    }

    return this.transcribeBuffered('partial', generation);
  }

  async flush(): Promise<PartialTranscript> {
    const generation = this.stateGeneration;
    return this.enqueue(() => {
      this.assertNotFinalized();
      this.consecutiveSilenceSeconds = 0;
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
    this.consecutiveSilenceSeconds = 0;
    this.heardSpeech = false;
    this.isFinalized = false;
  }

  /** Release once; every caller awaits the same cleanup and observes its failure. */
  dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.stateGeneration += 1;
    this.consecutiveSilenceSeconds = 0;
    // Publish before abort dispatch, whose listeners may reenter dispose().
    this.disposePromise = Promise.resolve().then(() => this.disposeInternal());
    this.operationAbortController.abort();
    return this.disposePromise;
  }

  private async disposeInternal(): Promise<void> {
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

  private async commitBeforeEviction(generation: number): Promise<void> {
    const audio = this.window.toPcmAudioBuffer();
    if (audio.numberOfFrames === 0) return;

    const signal = this.operationAbortController.signal;
    const windowBaseSeconds = this.window.getWindowBaseSeconds();
    let rawResult: TranscriptResult;
    try {
      rawResult = await this.canonicalTranscribe(audio, signal);
    } catch (error) {
      if (generation !== this.stateGeneration || signal.aborted) {
        return;
      }
      throw error;
    }

    if (generation !== this.stateGeneration) return;

    const offsetResult = this.applyWindowOffset(rawResult, windowBaseSeconds);
    const evictionDuration = this.window.getEvictionDuration();
    const freezeBoundary = windowBaseSeconds + evictionDuration;

    if (offsetResult.words && offsetResult.words.length > 0) {
      const wordsToCommit = offsetResult.words.filter(
        (w) => w.endTime <= freezeBoundary + 1e-4,
      );
      const committedWords = this.accumulator.getCommittedWords();
      const lastCommittedTime =
        committedWords.length > 0 ? committedWords[committedWords.length - 1]!.endTime : -1;
      const newWordsToCommit = wordsToCommit.filter(
        (w) => w.startTime >= lastCommittedTime - 0.15,
      );
      const textToCommit = newWordsToCommit.map((w) => w.text).join(' ');
      const segmentsToCommit = (offsetResult.segments ?? []).filter(
        (s) => s.endTime <= freezeBoundary + 1e-4,
      );
      const tokensToCommit = (offsetResult.tokens ?? []).filter(
        (t) => (t.endTime ?? t.startTime ?? 0) <= freezeBoundary + 1e-4,
      );
      this.accumulator.commit(textToCommit, newWordsToCommit, segmentsToCommit, tokensToCommit);
    } else if (offsetResult.text.trim().length > 0) {
      const allWords = offsetResult.text.trim().split(/\s+/).filter(Boolean);
      const bufferedSeconds = this.window.getBufferedDurationSeconds();
      const fraction = bufferedSeconds > 0 ? Math.min(1, evictionDuration / bufferedSeconds) : 0;
      const commitCount = Math.min(
        allWords.length,
        Math.max(1, Math.floor(allWords.length * fraction)),
      );
      const wordsToCommit = allWords.slice(0, commitCount);
      const deduplicatedCandidate = this.deduplicateAgainstCommitted({
        ...this.blankResult(),
        text: wordsToCommit.join(' '),
      });
      this.accumulator.commit(deduplicatedCandidate.text);
    }
  }

  private applyWindowOffset(result: TranscriptResult, baseSeconds: number): TranscriptResult {
    if (baseSeconds === 0) return result;
    const words = result.words?.map((word) => ({
      ...word,
      startTime: word.startTime + baseSeconds,
      endTime: word.endTime + baseSeconds,
    }));
    const segments = result.segments?.map((segment) => ({
      ...segment,
      startTime: segment.startTime + baseSeconds,
      endTime: segment.endTime + baseSeconds,
    }));
    const tokens = result.tokens?.map((token) => ({
      ...token,
      startTime: token.startTime !== undefined ? token.startTime + baseSeconds : undefined,
      endTime: token.endTime !== undefined ? token.endTime + baseSeconds : undefined,
    }));
    return {
      ...result,
      words,
      segments,
      tokens,
    };
  }

  private deduplicateAgainstCommitted(result: TranscriptResult): TranscriptResult {
    const committedWords = this.accumulator.getCommittedWords();
    const committedText = this.accumulator.getCommittedText();

    if (committedWords.length > 0 && result.words && result.words.length > 0) {
      const lastCommitted = committedWords[committedWords.length - 1]!;
      const lastEndTime = lastCommitted.endTime;
      let firstKeptIndex = 0;

      for (let i = 0; i < result.words.length; i++) {
        const w = result.words[i]!;
        if (w.endTime <= lastEndTime + 0.15) {
          firstKeptIndex = i + 1;
          continue;
        }
        if (w.startTime < lastEndTime - 0.15) {
          const match = committedWords.some(
            (cw) =>
              Math.abs(cw.endTime - w.endTime) < 0.15 &&
              normalizeWordKey(cw.text) === normalizeWordKey(w.text),
          );
          if (match) {
            firstKeptIndex = i + 1;
            continue;
          }
        }
        break;
      }

      const keptWords = result.words.slice(firstKeptIndex);
      const keptText = keptWords.map((w) => w.text).join(' ');
      const keptSegments = result.segments?.filter(
        (s) => s.endTime > lastEndTime + 0.15,
      );
      const keptTokens = result.tokens?.filter(
        (t) => (t.endTime ?? t.startTime ?? 0) > lastEndTime + 0.15,
      );

      return {
        ...result,
        text: keptText,
        words: keptWords.length > 0 ? keptWords : undefined,
        segments: keptSegments && keptSegments.length > 0 ? keptSegments : undefined,
        tokens: keptTokens && keptTokens.length > 0 ? keptTokens : undefined,
      };
    }

    if (committedText.trim().length > 0 && result.text.trim().length > 0) {
      const committedWordList = committedText.trim().split(/\s+/);
      const activeWordList = result.text.trim().split(/\s+/);
      let maxOverlap = 0;
      const maxPossibleOverlap = Math.min(committedWordList.length, activeWordList.length);

      for (let k = 1; k <= maxPossibleOverlap; k++) {
        let match = true;
        for (let j = 0; j < k; j++) {
          const c = normalizeWordKey(committedWordList[committedWordList.length - k + j]!);
          const a = normalizeWordKey(activeWordList[j]!);
          if (c !== a) {
            match = false;
            break;
          }
        }
        if (match) {
          maxOverlap = k;
        }
      }

      const keptWordList = activeWordList.slice(maxOverlap);
      return {
        ...result,
        text: keptWordList.join(' '),
      };
    }

    return result;
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

    const offsetResult = this.applyWindowOffset(canonical, this.window.getWindowBaseSeconds());
    const deduplicatedResult = this.deduplicateAgainstCommitted(offsetResult);

    return this.accumulator.update(deduplicatedResult, kind);
  }

  private async finalizeInternal(generation: number): Promise<PartialTranscript> {
    this.assertNotFinalized();
    this.consecutiveSilenceSeconds = 0;
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
    const words = this.accumulator.getCommittedWords();
    const segments = this.accumulator.getCommittedSegments();
    const tokens = this.accumulator.getCommittedTokens();
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
      words: words.length > 0 ? words.map((w, index) => ({ ...w, index })) : undefined,
      segments: segments.length > 0 ? segments.map((s, index) => ({ ...s, index })) : undefined,
      tokens: tokens.length > 0 ? tokens.map((t, index) => ({ ...t, index })) : undefined,
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

