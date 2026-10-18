import { normalizePcmInput } from '../../audio/audio.js';
import type { AudioInputLike } from '../../types/audio.js';
import type {
  PartialTranscript,
  TranscriptDetailLevel,
  TranscriptResult,
  TranscriptSegment,
  TranscriptToken,
  TranscriptWarning,
  TranscriptWord,
} from '../../types/transcript.js';
import type {
  StreamingSessionOptions,
  StreamingTranscriber,
  StreamingTranscriberState,
} from '../../types/streaming.js';
import type {
  NemotronRnntNativeTranscript,
  NemotronRnntTranscriptionOptions,
} from './types.js';
import type { SpeechSession } from '../../types/index.js';

const TARGET_SAMPLE_RATE = 16000;
const DEFAULT_CHUNK_SAMPLES = 10400; // 65 mel frames * 160 samples (650ms at 16kHz)
const MAX_CUMULATIVE_STREAM_SAMPLES = TARGET_SAMPLE_RATE * 60 * 5;

export interface NemotronStreamingTranscriberOptions extends StreamingSessionOptions {
  readonly modelId?: string;
  readonly modelFamily?: string;
  readonly sampleRate?: number;
  readonly stepSamples?: number;
  readonly onPartial?: (partial: PartialTranscript) => void;
  readonly promptId?: number;
}

function joinTranscriptFragments(committed: string, preview: string): string {
  const c = committed.trim();
  const p = preview.trim();
  if (c && p) {
    return `${c} ${p}`;
  }
  return c || p;
}

function shiftSegments(
  segments: readonly TranscriptSegment[] | undefined,
  offsetSeconds: number,
  baseIndex = 0,
): TranscriptSegment[] | undefined {
  if (!segments) return undefined;
  return segments.map((seg, idx) => ({
    ...seg,
    index: baseIndex + (seg.index ?? idx),
    startTime: Number(((seg.startTime ?? 0) + offsetSeconds).toFixed(3)),
    endTime: Number(((seg.endTime ?? 0) + offsetSeconds).toFixed(3)),
  }));
}

function shiftWords(
  words: readonly TranscriptWord[] | undefined,
  offsetSeconds: number,
  baseIndex = 0,
): TranscriptWord[] | undefined {
  if (!words) return undefined;
  return words.map((w, idx) => ({
    ...w,
    index: baseIndex + (w.index ?? idx),
    startTime: Number(((w.startTime ?? 0) + offsetSeconds).toFixed(3)),
    endTime: Number(((w.endTime ?? 0) + offsetSeconds).toFixed(3)),
  }));
}

function shiftTokens(
  tokens: readonly TranscriptToken[] | undefined,
  offsetSeconds: number,
  baseIndex = 0,
): TranscriptToken[] | undefined {
  if (!tokens) return undefined;
  return tokens.map((tok, idx) => ({
    ...tok,
    index: baseIndex + (tok.index ?? idx),
    startTime:
      typeof tok.startTime === 'number'
        ? Number((tok.startTime + offsetSeconds).toFixed(3))
        : undefined,
    endTime:
      typeof tok.endTime === 'number'
        ? Number((tok.endTime + offsetSeconds).toFixed(3))
        : undefined,
  }));
}

export class NemotronStreamingTranscriber implements StreamingTranscriber {
  private revision = 0;
  private totalPushedDurationSeconds = 0;
  private committedDurationSeconds = 0;
  private committedText = '';
  private previewText = '';
  private committedSegments: TranscriptSegment[] = [];
  private committedWords: TranscriptWord[] = [];
  private committedTokens: TranscriptToken[] = [];
  private accumulatedWarnings: TranscriptWarning[] = [];
  private bufferedChunks: Float32Array[] = [];
  private bufferedSamplesCount = 0;
  private cumulativeChunks: Float32Array[] = [];
  private cumulativeSamplesCount = 0;
  private isFinalized = false;
  private isDisposed = false;
  private generation = 0;
  private operationTail: Promise<void> = Promise.resolve();
  private operationAbortController = new AbortController();
  private readonly partialListeners = new Set<(partial: PartialTranscript) => void>();
  private readonly stepSamples: number;
  private readonly emitPartials: boolean;
  private readonly detailLevel: TranscriptDetailLevel;
  private readonly sampleRate: number;
  private fallbackWarningReported = false;

  constructor(
    private readonly session: SpeechSession<
      NemotronRnntTranscriptionOptions,
      NemotronRnntNativeTranscript
    >,
    private readonly options: NemotronStreamingTranscriberOptions = {},
    private readonly onDispose?: () => Promise<void> | void,
  ) {
    this.sampleRate = options.sampleRate ?? TARGET_SAMPLE_RATE;
    this.stepSamples = options.stepSamples ?? DEFAULT_CHUNK_SAMPLES;
    this.emitPartials = options.emitPartials ?? true;
    this.detailLevel = options.detail ?? 'segments';
    if (options.onPartial) {
      this.partialListeners.add(options.onPartial);
    }
  }

  onPartial(callback: (partial: PartialTranscript) => void): () => void {
    this.partialListeners.add(callback);
    return () => {
      this.partialListeners.delete(callback);
    };
  }

  private emitPartialEvent(partial: PartialTranscript): void {
    for (const listener of this.partialListeners) {
      try {
        listener(partial);
      } catch {
        // Ignore listener failures
      }
    }
  }

  private assertUsable(): void {
    if (this.isDisposed) {
      throw new Error('Nemotron streaming transcriber is disposed.');
    }
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const run = async (): Promise<T> => {
      return await operation();
    };
    const result = this.operationTail.then(run, run);
    this.operationTail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private consumeBufferedSamples(count: number): Float32Array {
    const actual = Math.min(count, this.bufferedSamplesCount);
    const out = new Float32Array(actual);
    let offset = 0;
    while (offset < actual && this.bufferedChunks.length > 0) {
      const first = this.bufferedChunks[0]!;
      const needed = actual - offset;
      if (first.length <= needed) {
        out.set(first, offset);
        offset += first.length;
        this.bufferedChunks.shift();
      } else {
        out.set(first.subarray(0, needed), offset);
        this.bufferedChunks[0] = first.subarray(needed);
        offset += needed;
      }
    }
    this.bufferedSamplesCount -= offset;
    return out;
  }

  private peekBufferedSamples(): Float32Array {
    const out = new Float32Array(this.bufferedSamplesCount);
    let offset = 0;
    for (const chunk of this.bufferedChunks) {
      out.set(chunk, offset);
      offset += chunk.length;
    }
    return out;
  }

  private async transcribeStateful(finalize: boolean): Promise<TranscriptResult | undefined> {
    const statefulSession = this.session as typeof this.session & {
      transcribeStreaming?: (
        audio: Float32Array,
        options: NemotronRnntTranscriptionOptions,
        finalize?: boolean,
      ) => Promise<TranscriptResult>;
    };
    if (!statefulSession.transcribeStreaming) {
      if (!this.fallbackWarningReported) {
        this.fallbackWarningReported = true;
        this.accumulatedWarnings.push({
          code: 'nemotron-rnnt.streaming-fallback-offline-chunks',
          message: 'This session has no stateful streaming API; audio chunks are transcribed independently without persistent model state.',
          recoverable: true,
        });
      }
      return undefined;
    }
    const pcm = new Float32Array(this.cumulativeSamplesCount);
    let offset = 0;
    for (const chunk of this.cumulativeChunks) {
      pcm.set(chunk, offset);
      offset += chunk.length;
    }
    return await statefulSession.transcribeStreaming(pcm, {
      detail: this.detailLevel,
      responseFlavor: 'canonical',
      promptId: this.options.promptId,
      signal: this.operationAbortController.signal,
    }, finalize);
  }

  private replaceCommittedTranscript(canonical: TranscriptResult): void {
    this.committedText = canonical.text?.trim() ?? '';
    this.committedSegments = canonical.segments ? [...canonical.segments] : [];
    this.committedWords = canonical.words ? [...canonical.words] : [];
    this.committedTokens = canonical.tokens ? [...canonical.tokens] : [];
    this.accumulatedWarnings = canonical.warnings ? [...canonical.warnings] : [];
    this.committedDurationSeconds = this.cumulativeSamplesCount / this.sampleRate;
    this.previewText = '';
  }

  private async transcribeAndCommitChunk(
    pcm: Float32Array,
    generation: number,
  ): Promise<void> {
    if (pcm.length === 0 || generation !== this.generation) {
      return;
    }
    const streamed = await this.transcribeStateful(false);
    if (streamed) {
      if (generation === this.generation) this.replaceCommittedTranscript(streamed);
      return;
    }
    const chunkDuration = pcm.length / this.sampleRate;
    const signal = this.operationAbortController.signal;
    try {
      const response = await this.session.transcribe(pcm, {
        detail: this.detailLevel,
        responseFlavor: 'canonical',
        promptId: this.options.promptId,
        signal,
      });

      if (generation !== this.generation) {
        return;
      }

      const canonical = (
        typeof response === 'object' && response !== null && 'canonical' in response
          ? (response as { canonical: TranscriptResult }).canonical
          : response
      ) as TranscriptResult;

      const chunkText = canonical?.text?.trim() ?? '';
      if (chunkText) {
        this.committedText = joinTranscriptFragments(this.committedText, chunkText);
      }

      if (canonical?.warnings) {
        this.accumulatedWarnings.push(...canonical.warnings);
      }

      if (canonical?.segments) {
        const shiftedSegments = shiftSegments(
          canonical.segments,
          this.committedDurationSeconds,
          this.committedSegments.length,
        );
        if (shiftedSegments) {
          this.committedSegments.push(...shiftedSegments);
        }
      }

      if (canonical?.words) {
        const shiftedWords = shiftWords(
          canonical.words,
          this.committedDurationSeconds,
          this.committedWords.length,
        );
        if (shiftedWords) {
          this.committedWords.push(...shiftedWords);
        }
      }

      if (canonical?.tokens) {
        const shiftedTokens = shiftTokens(
          canonical.tokens,
          this.committedDurationSeconds,
          this.committedTokens.length,
        );
        if (shiftedTokens) {
          this.committedTokens.push(...shiftedTokens);
        }
      }

      this.committedDurationSeconds += chunkDuration;
      this.previewText = '';
    } catch (error) {
      if (generation !== this.generation || signal.aborted) {
        return;
      }
      throw error;
    }
  }

  private async transcribePreviewChunk(
    pcm: Float32Array,
    generation: number,
  ): Promise<void> {
    if (pcm.length === 0 || generation !== this.generation) {
      return;
    }
    if ((this.session as typeof this.session & { transcribeStreaming?: unknown }).transcribeStreaming) {
      return;
    }
    const signal = this.operationAbortController.signal;
    try {
      const response = await this.session.transcribe(pcm, {
        detail: this.detailLevel,
        responseFlavor: 'canonical',
        promptId: this.options.promptId,
        signal,
      });

      if (generation !== this.generation) {
        return;
      }

      const canonical = (
        typeof response === 'object' && response !== null && 'canonical' in response
          ? (response as { canonical: TranscriptResult }).canonical
          : response
      ) as TranscriptResult;

      this.previewText = canonical?.text?.trim() ?? '';
    } catch {
      // Ignore preview errors / aborts
    }
  }

  async pushAudio(input: AudioInputLike): Promise<PartialTranscript> {
    const generation = this.generation;
    return this.enqueue(async () => {
      this.assertUsable();
      if (generation !== this.generation) {
        return this.buildPartialTranscript('partial');
      }

      const audio = normalizePcmInput(input).toMono();
      const samples = audio.channels[0] ?? new Float32Array(0);
      if (samples.length === 0) {
        return this.buildPartialTranscript('partial');
      }

      if (this.cumulativeSamplesCount + samples.length > MAX_CUMULATIVE_STREAM_SAMPLES) {
        throw new Error('Nemotron streaming currently retains up to five minutes of cumulative PCM per session.');
      }
      this.bufferedChunks.push(samples.slice());
      this.bufferedSamplesCount += samples.length;
      this.cumulativeChunks.push(samples.slice());
      this.cumulativeSamplesCount += samples.length;
      this.totalPushedDurationSeconds += samples.length / this.sampleRate;

      while (this.bufferedSamplesCount >= this.stepSamples && generation === this.generation) {
        const stepPcm = this.consumeBufferedSamples(this.stepSamples);
        await this.transcribeAndCommitChunk(stepPcm, generation);
      }

      if (this.emitPartials && generation === this.generation) {
        if (this.bufferedSamplesCount >= 160) {
          const previewPcm = this.peekBufferedSamples();
          await this.transcribePreviewChunk(previewPcm, generation);
        } else if (this.bufferedSamplesCount === 0) {
          this.previewText = '';
        }
      }

      this.revision += 1;
      const partial = this.buildPartialTranscript('partial');
      if (partial.text && partial.text.trim().length > 0) {
        this.emitPartialEvent(partial);
      }
      return partial;
    });
  }

  async flush(): Promise<PartialTranscript> {
    const generation = this.generation;
    return this.enqueue(async () => {
      this.assertUsable();
      if (this.bufferedSamplesCount > 0 && generation === this.generation) {
        const remainingPcm = this.consumeBufferedSamples(this.bufferedSamplesCount);
        await this.transcribeAndCommitChunk(remainingPcm, generation);
      }
      this.revision += 1;
      const partial = this.buildPartialTranscript('partial');
      if (partial.text && partial.text.trim().length > 0) {
        this.emitPartialEvent(partial);
      }
      return partial;
    });
  }

  async finalize(): Promise<PartialTranscript> {
    const generation = this.generation;
    return this.enqueue(async () => {
      this.assertUsable();
      if (this.bufferedSamplesCount > 0 && generation === this.generation) {
        const remainingPcm = this.consumeBufferedSamples(this.bufferedSamplesCount);
        await this.transcribeAndCommitChunk(remainingPcm, generation);
      }
      const streamed = await this.transcribeStateful(true);
      if (streamed && generation === this.generation) this.replaceCommittedTranscript(streamed);
      this.previewText = '';
      this.isFinalized = true;
      this.revision += 1;
      const finalResult = this.buildPartialTranscript('final');
      if (finalResult.text && finalResult.text.trim().length > 0) {
        this.emitPartialEvent(finalResult);
      }
      return finalResult;
    });
  }

  async reset(): Promise<void> {
    if (this.isDisposed) {
      throw new Error('Nemotron streaming transcriber is disposed.');
    }
    this.generation += 1;
    this.operationAbortController.abort();
    this.operationAbortController = new AbortController();
    this.bufferedChunks = [];
    this.bufferedSamplesCount = 0;
    this.cumulativeChunks = [];
    this.cumulativeSamplesCount = 0;
    const resettable = this.session as typeof this.session & { resetStreaming?: () => void };
    resettable.resetStreaming?.();
    this.committedText = '';
    this.previewText = '';
    this.committedSegments = [];
    this.committedWords = [];
    this.committedTokens = [];
    this.accumulatedWarnings = [];
    this.fallbackWarningReported = false;
    this.committedDurationSeconds = 0;
    this.totalPushedDurationSeconds = 0;
    this.revision = 0;
    this.isFinalized = false;
  }

  cancel(): void {
    if (this.isDisposed) {
      return;
    }
    this.generation += 1;
    this.operationAbortController.abort();
    this.operationAbortController = new AbortController();
    this.bufferedChunks = [];
    this.bufferedSamplesCount = 0;
    this.cumulativeChunks = [];
    this.cumulativeSamplesCount = 0;
    const resettable = this.session as typeof this.session & { resetStreaming?: () => void };
    resettable.resetStreaming?.();
    this.previewText = '';
  }

  async dispose(): Promise<void> {
    if (this.isDisposed) {
      return;
    }
    this.isDisposed = true;
    this.generation += 1;
    this.operationAbortController.abort();
    this.partialListeners.clear();
    await this.operationTail;
    if (this.onDispose) {
      await this.onDispose();
    }
  }

  getState(): StreamingTranscriberState {
    return {
      revision: this.revision,
      bufferedDurationSeconds: this.totalPushedDurationSeconds,
      committedText: this.committedText,
      previewText: this.previewText,
      isFinalized: this.isFinalized,
    };
  }

  private buildPartialTranscript(kind: 'partial' | 'final'): PartialTranscript {
    const text =
      kind === 'final'
        ? this.committedText
        : joinTranscriptFragments(this.committedText, this.previewText);

    return {
      kind,
      revision: this.revision,
      text,
      committedText: this.committedText,
      previewText: this.previewText,
      warnings: [...this.accumulatedWarnings],
      meta: {
        detailLevel: this.detailLevel,
        isFinal: kind === 'final',
        modelFamily: this.options.modelFamily ?? 'nemotron-rnnt',
        modelId: this.options.modelId ?? 'nemotron-speech',
        sampleRate: this.sampleRate,
        durationSeconds: this.totalPushedDurationSeconds,
      },
      segments: this.committedSegments.length > 0 ? [...this.committedSegments] : undefined,
      words: this.committedWords.length > 0 ? [...this.committedWords] : undefined,
      tokens: this.committedTokens.length > 0 ? [...this.committedTokens] : undefined,
    };
  }
}
