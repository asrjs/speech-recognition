import type { TranscriptResult, TranscriptWord } from '../types/index.js';

export interface AudioRingBufferOptions {
  readonly sampleRate: number;
  readonly durationSeconds: number;
}

export class AudioRingBuffer {
  readonly sampleRate: number;
  readonly maxFrames: number;
  private readonly buffer: Float32Array;
  private currentFrame = 0;

  constructor(options: AudioRingBufferOptions) {
    if (!Number.isFinite(options.sampleRate) || options.sampleRate <= 0) {
      throw new TypeError('AudioRingBuffer requires a positive sampleRate.');
    }
    if (!Number.isFinite(options.durationSeconds) || options.durationSeconds <= 0) {
      throw new TypeError('AudioRingBuffer requires a positive durationSeconds.');
    }

    this.sampleRate = options.sampleRate;
    this.maxFrames = Math.max(1, Math.floor(options.sampleRate * options.durationSeconds));
    this.buffer = new Float32Array(this.maxFrames);
  }

  write(chunk: Float32Array): void {
    let source = chunk;
    if (source.length > this.maxFrames) {
      const skipped = source.length - this.maxFrames;
      source = source.subarray(skipped);
      this.currentFrame += skipped;
    }

    const writePosition = this.currentFrame % this.maxFrames;
    const remainingAtEnd = this.maxFrames - writePosition;
    if (source.length <= remainingAtEnd) {
      this.buffer.set(source, writePosition);
    } else {
      this.buffer.set(source.subarray(0, remainingAtEnd), writePosition);
      this.buffer.set(source.subarray(remainingAtEnd), 0);
    }

    this.currentFrame += source.length;
  }

  read(startFrame: number, endFrame: number): Float32Array {
    const length = this.validateRange(startFrame, endFrame);
    const out = new Float32Array(length);
    this.readInto(startFrame, endFrame, out);
    return out;
  }

  readInto(startFrame: number, endFrame: number, destination: Float32Array): number {
    const length = this.validateRange(startFrame, endFrame);
    if (destination.length < length) {
      throw new RangeError(`Destination buffer too small. Needed ${length}, got ${destination.length}.`);
    }

    const readPosition = startFrame % this.maxFrames;
    const remainingAtEnd = this.maxFrames - readPosition;
    if (length <= remainingAtEnd) {
      destination.set(this.buffer.subarray(readPosition, readPosition + length), 0);
    } else {
      destination.set(this.buffer.subarray(readPosition, this.maxFrames), 0);
      destination.set(this.buffer.subarray(0, length - remainingAtEnd), remainingAtEnd);
    }
    return length;
  }

  getCurrentFrame(): number {
    return this.currentFrame;
  }

  getCurrentTimeSeconds(): number {
    return this.currentFrame / this.sampleRate;
  }

  getBaseFrameOffset(): number {
    return Math.max(0, this.currentFrame - this.maxFrames);
  }

  getFillCount(): number {
    return Math.min(this.currentFrame, this.maxFrames);
  }

  getSize(): number {
    return this.maxFrames;
  }

  reset(): void {
    this.currentFrame = 0;
    this.buffer.fill(0);
  }

  private validateRange(startFrame: number, endFrame: number): number {
    if (!Number.isFinite(startFrame) || startFrame < 0) {
      throw new RangeError('startFrame must be a non-negative finite number.');
    }
    if (!Number.isFinite(endFrame) || endFrame < startFrame) {
      throw new RangeError('endFrame must be greater than or equal to startFrame.');
    }
    const length = endFrame - startFrame;
    if (length === 0) {
      return 0;
    }

    const baseFrame = this.getBaseFrameOffset();
    if (startFrame < baseFrame) {
      throw new RangeError(`Requested frame ${startFrame} has been overwritten. Oldest available: ${baseFrame}.`);
    }
    if (endFrame > this.currentFrame) {
      throw new RangeError(`Requested frame ${endFrame} is in the future. Latest available: ${this.currentFrame}.`);
    }

    return length;
  }
}

export interface StreamingActivityBuffer {
  findSilenceBoundary(searchEndFrame: number, minimumFrame: number, threshold: number): number;
  getSilenceTailDuration(threshold: number): number;
  hasSpeechInRange(startFrame: number, endFrame: number, threshold: number): boolean;
}

export interface StreamingWindow {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly durationSeconds: number;
  readonly isInitial: boolean;
  readonly matureCursorFrame: number;
}

export interface StreamingWindowBuilderOptions {
  readonly sampleRate?: number;
  readonly minDurationSec?: number;
  readonly maxDurationSec?: number;
  readonly minInitialDurationSec?: number;
  readonly maxRetainedBoundaries?: number;
  readonly useActivityBoundaries?: boolean;
  readonly activityThreshold?: number;
  readonly debug?: boolean;
}

export class StreamingWindowBuilder {
  private readonly ringBuffer: Pick<AudioRingBuffer, 'getCurrentFrame' | 'getBaseFrameOffset'>;
  private readonly activityBuffer: StreamingActivityBuffer | null;
  private readonly config: Required<StreamingWindowBuilderOptions>;
  private readonly sentenceEnds: number[] = [];
  private matureCursorFrame = 0;
  private firstBoundaryReceived = false;

  constructor(
    ringBuffer: Pick<AudioRingBuffer, 'getCurrentFrame' | 'getBaseFrameOffset'>,
    activityBuffer: StreamingActivityBuffer | null = null,
    options: StreamingWindowBuilderOptions = {}
  ) {
    this.ringBuffer = ringBuffer;
    this.activityBuffer = activityBuffer;
    this.config = {
      sampleRate: options.sampleRate ?? 16000,
      minDurationSec: options.minDurationSec ?? 3,
      maxDurationSec: options.maxDurationSec ?? 30,
      minInitialDurationSec: options.minInitialDurationSec ?? 1.5,
      maxRetainedBoundaries: options.maxRetainedBoundaries ?? 4,
      useActivityBoundaries: options.useActivityBoundaries ?? true,
      activityThreshold: options.activityThreshold ?? 0.3,
      debug: options.debug ?? false,
    };
  }

  markSentenceEnd(frame: number): void {
    if (!Number.isFinite(frame) || frame < 0) {
      return;
    }
    this.sentenceEnds.push(frame);
    if (this.sentenceEnds.length > this.config.maxRetainedBoundaries) {
      this.sentenceEnds.splice(0, this.sentenceEnds.length - this.config.maxRetainedBoundaries);
    }
    this.firstBoundaryReceived = true;
  }

  advanceMatureCursor(frame: number): void {
    if (!Number.isFinite(frame) || frame <= this.matureCursorFrame) {
      return;
    }
    this.matureCursorFrame = frame;
    this.firstBoundaryReceived = true;
  }

  advanceMatureCursorByTime(seconds: number): void {
    this.advanceMatureCursor(Math.round(seconds * this.config.sampleRate));
  }

  getMatureCursorFrame(): number {
    return this.matureCursorFrame;
  }

  getMatureCursorTimeSeconds(): number {
    return this.matureCursorFrame / this.config.sampleRate;
  }

  buildWindow(): StreamingWindow | null {
    const endFrame = this.ringBuffer.getCurrentFrame();
    const baseFrame = this.ringBuffer.getBaseFrameOffset();
    if (endFrame <= baseFrame) {
      return null;
    }

    const availableFrames = endFrame - baseFrame;
    const minInitialFrames = Math.round(this.config.minInitialDurationSec * this.config.sampleRate);
    if (!this.firstBoundaryReceived) {
      if (availableFrames < minInitialFrames) {
        return null;
      }

      const maxFrames = Math.round(this.config.maxDurationSec * this.config.sampleRate);
      const clippedEnd = Math.min(endFrame, baseFrame + maxFrames);
      return {
        startFrame: baseFrame,
        endFrame: clippedEnd,
        durationSeconds: (clippedEnd - baseFrame) / this.config.sampleRate,
        isInitial: true,
        matureCursorFrame: this.matureCursorFrame,
      };
    }

    let startFrame = this.matureCursorFrame > 0
      ? this.matureCursorFrame
      : this.sentenceEnds.length >= 2
        ? this.sentenceEnds[this.sentenceEnds.length - 2]!
        : this.sentenceEnds.length === 1
          ? this.sentenceEnds[0]!
          : baseFrame;

    if (startFrame < baseFrame) {
      startFrame = baseFrame;
    }
    if (startFrame >= endFrame) {
      return null;
    }

    const minFrames = Math.round(this.config.minDurationSec * this.config.sampleRate);
    let windowFrames = endFrame - startFrame;
    if (windowFrames < minFrames) {
      return null;
    }

    const maxFrames = Math.round(this.config.maxDurationSec * this.config.sampleRate);
    if (windowFrames > maxFrames) {
      startFrame = Math.max(this.matureCursorFrame, endFrame - maxFrames);
      windowFrames = endFrame - startFrame;
    }

    if (this.config.useActivityBoundaries && this.activityBuffer) {
      const searchEnd = Math.min(startFrame + Math.round(this.config.sampleRate * 0.5), endFrame);
      const boundary = this.activityBuffer.findSilenceBoundary(searchEnd, startFrame, this.config.activityThreshold);
      const adjustedFrames = endFrame - boundary;
      if (boundary > startFrame && adjustedFrames >= minFrames) {
        startFrame = boundary;
        windowFrames = adjustedFrames;
      }
    }

    if (this.config.debug) {
      console.debug('[StreamingWindowBuilder] buildWindow', { startFrame, endFrame, windowFrames });
    }

    return {
      startFrame,
      endFrame,
      durationSeconds: windowFrames / this.config.sampleRate,
      isInitial: false,
      matureCursorFrame: this.matureCursorFrame,
    };
  }

  getSilenceTailDurationSeconds(): number {
    if (!this.activityBuffer) {
      return 0;
    }
    return this.activityBuffer.getSilenceTailDuration(this.config.activityThreshold);
  }

  hasSpeechInPendingWindow(): boolean {
    if (!this.activityBuffer) {
      return true;
    }
    const endFrame = this.ringBuffer.getCurrentFrame();
    const startFrame = this.matureCursorFrame > 0 ? this.matureCursorFrame : this.ringBuffer.getBaseFrameOffset();
    if (startFrame >= endFrame) {
      return false;
    }
    return this.activityBuffer.hasSpeechInRange(startFrame, endFrame, this.config.activityThreshold);
  }

  reset(): void {
    this.sentenceEnds.length = 0;
    this.matureCursorFrame = 0;
    this.firstBoundaryReceived = false;
  }
}

export interface MergedTranscriptWord {
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly wordIndex?: number;
}

export interface MergedTranscriptSentence {
  readonly id: string;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly wordCount: number;
  readonly words: readonly MergedTranscriptWord[];
  readonly isCommitted: boolean;
  readonly detectionMethod: 'heuristic';
}

export interface UtteranceTranscriptSnapshot {
  readonly committedText: string;
  readonly previewText: string;
  readonly fullText: string;
  readonly committedSentences: readonly MergedTranscriptSentence[];
  readonly newlyCommittedSentences: readonly MergedTranscriptSentence[];
  readonly pendingSentence: MergedTranscriptSentence | null;
  readonly matureCursorTime: number;
  readonly revision: number;
  readonly sourceText: string;
}

export interface UtteranceTranscriptMergerOptions {
  readonly dedupToleranceSeconds?: number;
  readonly minimumSentenceWords?: number;
  readonly debug?: boolean;
}

interface InternalTranscriptWord {
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
}

const SENTENCE_SPLIT_PATTERN = /[^.!?]+[.!?]+|[^.!?]+$/g;
const SENTENCE_END_PATTERN = /[.!?]$/;

function normalizeSentenceKey(text: string): string {
  return text.replace(/\s+/g, ' ').trim().toLowerCase();
}

function wordsToInternal(words: readonly TranscriptWord[]): InternalTranscriptWord[] {
  return words.map((word) => ({
    text: String(word.text ?? '').trim(),
    startTime: word.startTime,
    endTime: word.endTime,
    confidence: word.confidence,
  })).filter((word) => word.text.length > 0);
}

function textToInternalWords(text: string): InternalTranscriptWord[] {
  return text
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0)
    .map((token) => ({ text: token }));
}

function joinWords(words: readonly InternalTranscriptWord[]): string {
  return words.map((word) => word.text).join(' ').replace(/\s+/g, ' ').trim();
}

function splitSentences(text: string): string[] {
  return (text.trim().match(SENTENCE_SPLIT_PATTERN) ?? [])
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
}

function mapSentenceBoundaries(words: readonly InternalTranscriptWord[], sentences: readonly string[]): number[] {
  const boundaries: number[] = [];
  let wordIndex = 0;
  for (const sentence of sentences) {
    const target = normalizeSentenceKey(sentence).replace(/\s+/g, '');
    let accumulated = '';
    for (let index = wordIndex; index < words.length; index += 1) {
      accumulated += normalizeSentenceKey(words[index]!.text).replace(/\s+/g, '');
      if (accumulated.length >= target.length) {
        wordIndex = index + 1;
        break;
      }
    }
    boundaries.push(wordIndex);
  }
  return boundaries;
}

function createSentence(
  id: string,
  words: readonly InternalTranscriptWord[],
  isCommitted: boolean
): MergedTranscriptSentence | null {
  const text = joinWords(words);
  if (!text) {
    return null;
  }
  return {
    id,
    text,
    startTime: words[0]?.startTime,
    endTime: words[words.length - 1]?.endTime,
    wordCount: words.length,
    words: words.map((word, index) => ({
      text: word.text,
      startTime: word.startTime,
      endTime: word.endTime,
      confidence: word.confidence,
      wordIndex: index,
    })),
    isCommitted,
    detectionMethod: 'heuristic',
  };
}

export class UtteranceTranscriptMerger {
  private readonly dedupToleranceSeconds: number;
  private readonly minimumSentenceWords: number;
  private readonly debug: boolean;
  private committedWords: InternalTranscriptWord[] = [];
  private pendingWords: InternalTranscriptWord[] = [];
  private committedSentences: MergedTranscriptSentence[] = [];
  private finalizedMeta: Array<{ text: string; endTime?: number }> = [];
  private matureCursorTime = 0;
  private revision = 0;
  private sentenceId = 0;
  private sourceText = '';

  constructor(options: UtteranceTranscriptMergerOptions = {}) {
    this.dedupToleranceSeconds = options.dedupToleranceSeconds ?? 0.15;
    this.minimumSentenceWords = Math.max(1, Math.floor(options.minimumSentenceWords ?? 1));
    this.debug = options.debug ?? false;
  }

  process(result: TranscriptResult): UtteranceTranscriptSnapshot {
    const incomingWords = result.words && result.words.length > 0
      ? wordsToInternal(result.words)
      : textToInternalWords(result.text);
    this.sourceText = result.text;
    this.revision += 1;

    if (incomingWords.length === 0) {
      this.pendingWords = [];
      return this.createSnapshot([]);
    }

    const sentences = splitSentences(joinWords(incomingWords));
    const boundaries = mapSentenceBoundaries(incomingWords, sentences);
    const newlyCommitted: MergedTranscriptSentence[] = [];

    if (sentences.length > 1) {
      let previousIndex = 0;
      for (let sentenceIndex = 0; sentenceIndex < sentences.length - 1; sentenceIndex += 1) {
        const endIndex = boundaries[sentenceIndex] ?? previousIndex;
        const sentenceWords = incomingWords.slice(previousIndex, endIndex);
        previousIndex = endIndex;
        if (sentenceWords.length < this.minimumSentenceWords) {
          continue;
        }
        if (this.isDuplicateSentence(joinWords(sentenceWords), sentenceWords.at(-1)?.endTime)) {
          continue;
        }
        this.committedWords.push(...sentenceWords);
        const sentence = createSentence(`sentence_${this.sentenceId++}`, sentenceWords, true);
        if (sentence) {
          this.committedSentences.push(sentence);
          this.finalizedMeta.push({ text: sentence.text, endTime: sentence.endTime });
          newlyCommitted.push(sentence);
          if (Number.isFinite(sentence.endTime) && (sentence.endTime ?? 0) > this.matureCursorTime) {
            this.matureCursorTime = sentence.endTime ?? this.matureCursorTime;
          }
        }
      }
      this.pendingWords = incomingWords.slice(previousIndex);
    } else {
      this.pendingWords = incomingWords;
    }

    if (this.debug) {
      console.debug('[UtteranceTranscriptMerger] process', {
        committedSentences: this.committedSentences.length,
        pendingWords: this.pendingWords.length,
      });
    }

    return this.createSnapshot(newlyCommitted);
  }

  finalizePendingIfComplete(): UtteranceTranscriptSnapshot | null {
    if (this.pendingWords.length === 0) {
      return null;
    }
    const pendingText = joinWords(this.pendingWords);
    if (!SENTENCE_END_PATTERN.test(pendingText.trim())) {
      return null;
    }
    return this.forceFinalizePending();
  }

  forceFinalizePending(): UtteranceTranscriptSnapshot | null {
    if (this.pendingWords.length === 0) {
      return null;
    }
    const pendingText = joinWords(this.pendingWords);
    const pendingEndTime = this.pendingWords.at(-1)?.endTime;
    if (this.isDuplicateSentence(pendingText, pendingEndTime)) {
      this.pendingWords = [];
      return this.createSnapshot([]);
    }

    this.committedWords.push(...this.pendingWords);
    const sentence = createSentence(`sentence_${this.sentenceId++}`, this.pendingWords, true);
    const newlyCommitted = sentence ? [sentence] : [];
    if (sentence) {
      this.committedSentences.push(sentence);
      this.finalizedMeta.push({ text: sentence.text, endTime: sentence.endTime });
      if (Number.isFinite(sentence.endTime) && (sentence.endTime ?? 0) > this.matureCursorTime) {
        this.matureCursorTime = sentence.endTime ?? this.matureCursorTime;
      }
    }
    this.pendingWords = [];
    this.revision += 1;
    return this.createSnapshot(newlyCommitted);
  }

  getCommittedText(): string {
    return joinWords(this.committedWords);
  }

  getPreviewText(): string {
    return joinWords(this.pendingWords);
  }

  getFullText(): string {
    return [this.getCommittedText(), this.getPreviewText()].filter(Boolean).join(' ').trim();
  }

  getCommittedSentences(): readonly MergedTranscriptSentence[] {
    return this.committedSentences;
  }

  getMatureCursorTimeSeconds(): number {
    return this.matureCursorTime;
  }

  reset(): void {
    this.committedWords = [];
    this.pendingWords = [];
    this.committedSentences = [];
    this.finalizedMeta = [];
    this.matureCursorTime = 0;
    this.revision = 0;
    this.sentenceId = 0;
    this.sourceText = '';
  }

  private isDuplicateSentence(text: string, endTime: number | undefined): boolean {
    const normalized = normalizeSentenceKey(text);
    return this.finalizedMeta.some((meta) => {
      if (normalizeSentenceKey(meta.text) !== normalized) {
        return false;
      }
      if (!Number.isFinite(endTime) || !Number.isFinite(meta.endTime)) {
        return true;
      }
      return Math.abs((meta.endTime ?? 0) - (endTime ?? 0)) < this.dedupToleranceSeconds;
    });
  }

  private createSnapshot(newlyCommittedSentences: readonly MergedTranscriptSentence[]): UtteranceTranscriptSnapshot {
    const pendingSentence = createSentence('pending', this.pendingWords, false);
    return {
      committedText: this.getCommittedText(),
      previewText: this.getPreviewText(),
      fullText: this.getFullText(),
      committedSentences: this.committedSentences,
      newlyCommittedSentences,
      pendingSentence,
      matureCursorTime: this.matureCursorTime,
      revision: this.revision,
      sourceText: this.sourceText,
    };
  }
}
