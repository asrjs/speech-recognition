import { normalizePcmInput, PcmAudioBuffer } from '../processors/index.js';
import type {
  AudioInputLike,
  PartialTranscript,
  TranscriptMeta,
  TranscriptResult,
  TranscriptWarning,
  VoiceActivityDetector
} from '../types/index.js';
import {
  AudioRingBuffer,
  StreamingWindowBuilder,
  type StreamingWindow,
  type StreamingWindowBuilderOptions,
  UtteranceTranscriptMerger,
  type UtteranceTranscriptMergerOptions,
  type UtteranceTranscriptSnapshot
} from './realtime.js';
import { VoiceActivityTimeline, type VoiceActivityObservation, type VoiceActivityTimelineSnapshot } from './vad.js';

export interface RealtimeTranscriptionRequest {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly startTimeSeconds: number;
  readonly endTimeSeconds: number;
  readonly durationSeconds: number;
  readonly isInitialWindow: boolean;
  readonly matureCursorFrame: number;
  readonly matureCursorTimeSeconds: number;
  readonly reason: 'push' | 'flush' | 'finalize';
}

export type RealtimeTranscriptionCallback = (
  request: RealtimeTranscriptionRequest
) => Promise<TranscriptResult> | TranscriptResult;

export interface RealtimeControllerPushOptions {
  readonly vadObservation?: VoiceActivityObservation;
}

export interface RealtimeTranscriptionControllerOptions {
  readonly sampleRate: number;
  readonly bufferDurationSeconds?: number;
  readonly vadTimeline?: VoiceActivityTimeline | null;
  readonly vad?: VoiceActivityDetector;
  readonly speechThreshold?: number;
  readonly finalizeSilenceSeconds?: number;
  readonly transcribe: RealtimeTranscriptionCallback;
  readonly window?: StreamingWindowBuilderOptions;
  readonly merger?: UtteranceTranscriptMerger;
  readonly mergerOptions?: UtteranceTranscriptMergerOptions;
}

export interface RealtimeTranscriptionUpdate {
  readonly kind: 'partial' | 'final';
  readonly trigger: 'push' | 'flush' | 'finalize' | 'silence-finalize';
  readonly partial: PartialTranscript;
  readonly snapshot: UtteranceTranscriptSnapshot;
  readonly canonical: TranscriptResult;
  readonly window: StreamingWindow;
  readonly activity: VoiceActivityTimelineSnapshot | null;
}

export interface RealtimeTranscriptionControllerState {
  readonly currentFrame: number;
  readonly currentTimeSeconds: number;
  readonly baseFrame: number;
  readonly trailingSilenceSeconds: number;
  readonly isFinalized: boolean;
  readonly snapshot: UtteranceTranscriptSnapshot;
  readonly lastWindow: StreamingWindow | null;
  readonly activity: VoiceActivityTimelineSnapshot | null;
}

function withTimeOffset(result: TranscriptResult, offsetSeconds: number): TranscriptResult {
  if (!offsetSeconds) {
    return result;
  }

  return {
    ...result,
    segments: result.segments?.map((segment) => ({
      ...segment,
      startTime: segment.startTime + offsetSeconds,
      endTime: segment.endTime + offsetSeconds
    })),
    words: result.words?.map((word) => ({
      ...word,
      startTime: word.startTime + offsetSeconds,
      endTime: word.endTime + offsetSeconds
    })),
    tokens: result.tokens?.map((token) => ({
      ...token,
      startTime: token.startTime !== undefined ? token.startTime + offsetSeconds : undefined,
      endTime: token.endTime !== undefined ? token.endTime + offsetSeconds : undefined
    }))
  };
}

function buildPartialTranscript(
  snapshot: UtteranceTranscriptSnapshot,
  canonical: TranscriptResult,
  kind: 'partial' | 'final'
): PartialTranscript {
  const warnings = canonical.warnings;
  const meta: TranscriptMeta = {
    ...canonical.meta,
    isFinal: kind === 'final',
  };

  return {
    kind,
    revision: snapshot.revision,
    text: kind === 'final' ? snapshot.committedText : snapshot.fullText,
    committedText: snapshot.committedText,
    previewText: snapshot.previewText,
    warnings,
    meta,
    segments: canonical.segments,
    words: canonical.words,
    tokens: canonical.tokens
  };
}

function createEmptyCanonical(detailLevel: TranscriptMeta['detailLevel'], warning?: TranscriptWarning): TranscriptResult {
  return {
    text: '',
    warnings: warning ? [warning] : [],
    meta: {
      detailLevel,
      isFinal: false
    }
  };
}

export class RealtimeTranscriptionController {
  readonly sampleRate: number;
  readonly audio: AudioRingBuffer;
  readonly activity: VoiceActivityTimeline | null;
  readonly windowBuilder: StreamingWindowBuilder;
  readonly merger: UtteranceTranscriptMerger;
  private readonly vad?: VoiceActivityDetector;
  private readonly transcribeCallback: RealtimeTranscriptionCallback;
  private readonly speechThreshold: number;
  private readonly finalizeSilenceFrames: number;
  private lastWindow: StreamingWindow | null = null;
  private lastSnapshot: UtteranceTranscriptSnapshot;
  private isFinalized = false;

  constructor(options: RealtimeTranscriptionControllerOptions) {
    if (!Number.isFinite(options.sampleRate) || options.sampleRate <= 0) {
      throw new TypeError('RealtimeTranscriptionController requires a positive sampleRate.');
    }

    this.sampleRate = options.sampleRate;
    this.audio = new AudioRingBuffer({
      sampleRate: options.sampleRate,
      durationSeconds: options.bufferDurationSeconds ?? 30
    });
    this.activity = options.vadTimeline ?? new VoiceActivityTimeline({
      sampleRate: options.sampleRate,
      maxDurationSeconds: options.bufferDurationSeconds ?? 30,
      speechThreshold: options.speechThreshold
    });
    this.windowBuilder = new StreamingWindowBuilder(this.audio, this.activity, {
      sampleRate: options.sampleRate,
      ...options.window
    });
    this.merger = options.merger ?? new UtteranceTranscriptMerger(options.mergerOptions);
    this.vad = options.vad;
    this.transcribeCallback = options.transcribe;
    this.speechThreshold = options.speechThreshold ?? this.activity?.speechThreshold ?? 0.5;
    this.finalizeSilenceFrames = Math.max(0, Math.round((options.finalizeSilenceSeconds ?? 0.8) * options.sampleRate));
    this.lastSnapshot = this.merger.process(createEmptyCanonical('text'));
  }

  async pushAudio(
    input: AudioInputLike,
    options: RealtimeControllerPushOptions = {}
  ): Promise<RealtimeTranscriptionUpdate | null> {
    this.assertNotFinalized();

    const normalized = (
      input instanceof Float32Array || input instanceof Float64Array
        ? PcmAudioBuffer.fromMono(input, this.sampleRate)
        : normalizePcmInput(input)
    ).toMono();
    if (normalized.sampleRate !== this.sampleRate) {
      throw new RangeError(`RealtimeTranscriptionController expected ${this.sampleRate} Hz audio, received ${normalized.sampleRate} Hz.`);
    }

    const startFrame = this.audio.getCurrentFrame();
    this.audio.write(normalized.channels[0]!);
    const endFrame = this.audio.getCurrentFrame();

    const activity = options.vadObservation
      ? options.vadObservation
      : await this.detectVoiceActivity(input, startFrame, endFrame);
    if (activity && this.activity) {
      this.activity.appendObservation(activity);
    }

    return this.processWindow('push');
  }

  async flush(): Promise<RealtimeTranscriptionUpdate | null> {
    this.assertNotFinalized();
    return this.processWindow('flush');
  }

  async finalize(): Promise<RealtimeTranscriptionUpdate | null> {
    this.assertNotFinalized();
    const update = await this.processWindow('finalize', true);
    this.isFinalized = true;
    return update;
  }

  reset(): void {
    this.audio.reset();
    this.activity?.reset();
    this.windowBuilder.reset();
    this.merger.reset();
    this.lastWindow = null;
    this.lastSnapshot = this.merger.process(createEmptyCanonical('text'));
    this.isFinalized = false;
  }

  getState(): RealtimeTranscriptionControllerState {
    return {
      currentFrame: this.audio.getCurrentFrame(),
      currentTimeSeconds: this.audio.getCurrentTimeSeconds(),
      baseFrame: this.audio.getBaseFrameOffset(),
      trailingSilenceSeconds: this.activity
        ? this.activity.getSilenceTailDuration(this.speechThreshold) / this.sampleRate
        : 0,
      isFinalized: this.isFinalized,
      snapshot: this.lastSnapshot,
      lastWindow: this.lastWindow,
      activity: this.activity?.createSnapshot() ?? null
    };
  }

  private async processWindow(
    reason: 'push' | 'flush' | 'finalize',
    forceFinalizePending = false
  ): Promise<RealtimeTranscriptionUpdate | null> {
    const window = this.windowBuilder.buildWindow();
    if (!window) {
      if (forceFinalizePending) {
        return this.finalizePending('finalize', createEmptyCanonical('text'));
      }
      return null;
    }

    this.lastWindow = window;
    const pcm = this.audio.read(window.startFrame, window.endFrame);
    const request: RealtimeTranscriptionRequest = {
      pcm,
      sampleRate: this.sampleRate,
      startFrame: window.startFrame,
      endFrame: window.endFrame,
      startTimeSeconds: window.startFrame / this.sampleRate,
      endTimeSeconds: window.endFrame / this.sampleRate,
      durationSeconds: window.durationSeconds,
      isInitialWindow: window.isInitial,
      matureCursorFrame: window.matureCursorFrame,
      matureCursorTimeSeconds: window.matureCursorFrame / this.sampleRate,
      reason
    };

    const canonical = withTimeOffset(
      await this.transcribeCallback(request),
      request.startTimeSeconds
    );
    let snapshot = this.merger.process(canonical);
    this.lastSnapshot = snapshot;
    this.windowBuilder.advanceMatureCursorByTime(snapshot.matureCursorTime);

    if (forceFinalizePending) {
      return this.finalizePending('finalize', canonical);
    }

    if (
      this.activity
      && this.finalizeSilenceFrames > 0
      && this.activity.getSilenceTailDuration(this.speechThreshold) >= this.finalizeSilenceFrames
      && snapshot.previewText
    ) {
      return this.finalizePending('silence-finalize', canonical);
    }

    const partial = buildPartialTranscript(snapshot, canonical, 'partial');
    return {
      kind: 'partial',
      trigger: reason,
      partial,
      snapshot,
      canonical,
      window,
      activity: this.activity?.createSnapshot() ?? null
    };
  }

  private finalizePending(
    trigger: 'silence-finalize' | 'finalize',
    canonical: TranscriptResult
  ): RealtimeTranscriptionUpdate | null {
    const finalized = this.merger.forceFinalizePending();
    const snapshot = finalized ?? this.lastSnapshot;
    this.lastSnapshot = snapshot;
    this.windowBuilder.advanceMatureCursorByTime(snapshot.matureCursorTime);

    const window = this.lastWindow ?? {
      startFrame: this.audio.getBaseFrameOffset(),
      endFrame: this.audio.getCurrentFrame(),
      durationSeconds: this.audio.getFillCount() / this.sampleRate,
      isInitial: false,
      matureCursorFrame: Math.round(snapshot.matureCursorTime * this.sampleRate)
    };

    const partial = buildPartialTranscript(snapshot, canonical, 'final');
    return {
      kind: 'final',
      trigger,
      partial,
      snapshot,
      canonical,
      window,
      activity: this.activity?.createSnapshot() ?? null
    };
  }

  private async detectVoiceActivity(
    input: AudioInputLike,
    defaultStartFrame: number,
    defaultEndFrame: number
  ): Promise<VoiceActivityObservation | null> {
    if (!this.vad) {
      return null;
    }

    const event = await this.vad.analyze(input);
    const startFrame = Number.isFinite(event.startTimeSeconds)
      ? Math.max(0, Math.round((event.startTimeSeconds ?? 0) * this.sampleRate))
      : defaultStartFrame;
    const endFrame = Number.isFinite(event.endTimeSeconds)
      ? Math.max(startFrame + 1, Math.round((event.endTimeSeconds ?? 0) * this.sampleRate))
      : defaultEndFrame;

    return {
      startFrame,
      endFrame,
      speechProbability: event.speechProbability,
      isSpeech: event.isSpeech
    };
  }

  private assertNotFinalized(): void {
    if (this.isFinalized) {
      throw new Error('RealtimeTranscriptionController is finalized. Call reset() before pushing new audio.');
    }
  }
}
