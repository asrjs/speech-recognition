import { AudioRingBuffer } from './realtime.js';
import {
  durationMsToAlignedFrameCount,
  resolveStreamingTimelineChunkFrames,
} from './audio-timeline.js';
import {
  DEFAULT_ROUGH_GATE_CONFIG,
  type RoughSpeechGateConfig,
} from './rough-gate-config.js';
import {
  RoughSpeechGate,
  type RoughSpeechChunkSummary,
  type RoughSpeechTimelinePoint,
  type RoughSpeechGateWindowResult,
} from './rough-speech-gate.js';
import {
  scoreSegmentForeground,
  type SegmentForegroundFilterResult,
} from './segment-foreground-filter.js';
import {
  STREAMING_GATE_MODES,
  STREAMING_PROFILE_IDS,
  STREAMING_PRESETS,
  getStreamingPreset,
  isStreamingConfigEqual,
  listStreamingPresets,
  mergeStreamingConfig,
  resolveDefaultMicMode,
  resolveStreamingProfileId,
  type StreamingDetectorConfig,
  type StreamingDetectorConfigOverrides,
  type StreamingGateMode,
  type StreamingDetectorPreset,
  type StreamingProfileId,
} from './streaming-config.js';

export {
  DEFAULT_ROUGH_GATE_CONFIG,
  STREAMING_GATE_MODES,
  STREAMING_PROFILE_IDS,
  STREAMING_PRESETS,
  getStreamingPreset,
  isStreamingConfigEqual,
  listStreamingPresets,
  mergeStreamingConfig,
  resolveDefaultMicMode,
  resolveStreamingProfileId,
};
export type {
  RoughSpeechGateConfig,
  RoughSpeechGateWindowResult,
  StreamingDetectorConfig,
  StreamingDetectorConfigOverrides,
  StreamingGateMode,
  StreamingDetectorPreset,
  StreamingProfileId,
};

export interface StreamingTenVadStatus {
  readonly state: 'idle' | 'initializing' | 'ready' | 'degraded' | 'disabled';
  readonly error: string | null;
  readonly probability: number;
  readonly speaking: boolean;
  readonly threshold: number;
}

export interface StreamingTenVadWindowSummary {
  readonly totalHops: number;
  readonly speechHopCount: number;
  readonly nonSpeechHopCount: number;
  readonly maxConsecutiveSpeech: number;
  readonly maxProbability: number;
  readonly recent: readonly unknown[];
}

export interface StreamingTenVadResultEvent {
  readonly type: 'result';
  readonly payload: unknown;
}

export interface StreamingTenVadLike {
  subscribe(listener: (event: StreamingTenVadResultEvent) => void): () => void;
  init(): Promise<void>;
  reset(): Promise<void>;
  dispose(): Promise<void>;
  updateConfig(config?: Record<string, unknown>): void;
  process(samples: Float32Array, globalSampleOffset: number): boolean;
  getStatus(): StreamingTenVadStatus;
  findFirstSpeechFrame(startFrame: number, endFrame: number): number | null;
  hasRecentSpeech(endFrame: number, windowMs: number, sampleRate: number): boolean;
  hasRecentSilence(endFrame: number, windowMs: number, sampleRate: number): boolean;
  getWindowSummary(
    endFrame: number,
    windowMs: number,
    sampleRate: number,
  ): StreamingTenVadWindowSummary;
}

export interface StreamingDetectorSegment {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly sampleRate: number;
  readonly reason: string;
  readonly metadata: {
    readonly profileId: string;
    readonly rough: RoughSpeechGateWindowResult | null;
    readonly tenVad: StreamingTenVadStatus;
    readonly filter: SegmentForegroundFilterResult | null;
  };
  readPcm(): Float32Array;
}

export interface StreamingSpeechDetectorSnapshot {
  readonly state: 'idle' | 'listening' | 'candidate' | 'speaking';
  readonly sampleRate: number;
  readonly profileId: string;
  readonly config: StreamingDetectorConfig;
  readonly waveform: ReturnType<AudioRingBuffer['getMinMaxPairs']>;
  readonly activeSegment: { readonly startFrame: number; readonly endFrame: number } | null;
  readonly pendingSegmentStartFrame: number | null;
  readonly recentSegments: readonly StreamingDetectorSegment[];
  readonly recentDecisions: ReadonlyArray<{
    readonly id: string;
    readonly at: string;
    readonly message: string;
    readonly meta: Record<string, unknown>;
  }>;
  readonly acceptance: Record<string, unknown> | null;
  readonly gate: {
    readonly requestedMode: StreamingGateMode;
    readonly effectiveMode: StreamingGateMode;
    readonly tenVadReady: boolean;
  };
  readonly rough: Record<string, unknown> & {
    readonly recent?: readonly RoughSpeechChunkSummary[];
    readonly timeline?: readonly RoughSpeechTimelinePoint[];
  };
  readonly foreground: {
    readonly enabled: boolean;
    readonly noiseFloorDbfs: number;
    readonly foregroundMinDb: number;
    readonly onsetMinDb: number;
    readonly longMinDb: number;
    readonly lastResult: Record<string, unknown> | null;
  };
  readonly tenVad: StreamingTenVadStatus;
  readonly warnings: readonly string[];
  readonly error: string | null;
}

export type StreamingSpeechDetectorEvent =
  | { readonly type: 'metrics'; readonly payload: StreamingSpeechDetectorSnapshot }
  | {
      readonly type: 'speech-start';
      readonly payload: { readonly startFrame: number; readonly sampleRate: number };
    }
  | {
      readonly type: 'speech-update';
      readonly payload: {
        readonly startFrame: number;
        readonly endFrame: number;
        readonly sampleRate: number;
      };
    }
  | {
      readonly type: 'speech-end';
      readonly payload: {
        readonly startFrame: number;
        readonly endFrame: number;
        readonly sampleRate: number;
      };
    }
  | { readonly type: 'segment-ready'; readonly payload: StreamingDetectorSegment }
  | { readonly type: 'error'; readonly payload: Error };

interface ActiveSegmentState {
  readonly startFrame: number;
  readonly lastFrame: number;
  readonly source: 'ten-vad' | 'rough-fallback';
}

export interface StreamingSpeechDetectorOptions {
  readonly profileId?: string;
  readonly config?: StreamingDetectorConfigOverrides;
  readonly isRealtimeEouModel?: boolean;
  readonly tenVadFactory?: (
    config: Record<string, unknown>,
    options?: unknown,
  ) => StreamingTenVadLike;
  readonly tenVadOptions?: unknown;
}

function cloneEntries<T extends Record<string, unknown>>(entries: readonly T[]): T[] {
  return entries.map((entry) => ({ ...entry }));
}

export class StreamingSpeechDetector {
  private readonly createTenVad;
  private profileId: string;
  private config: StreamingDetectorConfig;
  private listeners = new Set<(event: StreamingSpeechDetectorEvent) => void>();
  private sampleRate: number;
  private ringBuffer: AudioRingBuffer;
  private roughGate: RoughSpeechGate;
  private tenVad: StreamingTenVadLike | null;
  private state: 'idle' | 'listening' | 'candidate' | 'speaking' = 'idle';
  private activeSegment: ActiveSegmentState | null = null;
  private pendingSegmentStartFrame: number | null = null;
  private recentSegments: StreamingDetectorSegment[] = [];
  private recentDecisions: Array<{
    readonly id: string;
    readonly at: string;
    readonly message: string;
    readonly meta: Record<string, unknown>;
  }> = [];
  private lastMetrics: RoughSpeechGateWindowResult | null = null;
  private lastAcceptanceInfo: Record<string, unknown> | null = null;
  private lastForegroundResult: SegmentForegroundFilterResult | null = null;
  private lastError: Error | null = null;
  private disposed = false;
  private tenVadOptions: unknown;
  private tenVadUnsubscribe: (() => void) | null = null;
  private lastDebugTraceAt = 0;

  constructor(options: StreamingSpeechDetectorOptions = {}) {
    this.createTenVad =
      options.tenVadFactory ??
      (() => {
        throw new Error('StreamingSpeechDetector requires a tenVadFactory when TEN-VAD is enabled.');
      });
    this.profileId =
      options.profileId ?? resolveStreamingProfileId(options.isRealtimeEouModel === true);
    this.config = mergeStreamingConfig(this.profileId, options.config);
    this.sampleRate = this.config.sampleRate;
    this.ringBuffer = new AudioRingBuffer({
      sampleRate: this.sampleRate,
      durationSeconds: this.config.ringBufferDurationMs / 1000,
    });
    this.roughGate = new RoughSpeechGate(this.buildRoughGateConfig());
    this.tenVad = this.config.tenVadEnabled
      ? this.createTenVad(this.buildTenVadConfig(), options.tenVadOptions)
      : null;
    this.tenVadOptions = options.tenVadOptions;
    this.tenVadUnsubscribe = this.subscribeTenVad(this.tenVad);
  }

  private buildRoughGateConfig(): RoughSpeechGateConfig {
    return {
      sampleRate: this.sampleRate,
      analysisWindowMs: this.config.analysisWindowMs,
      energySmoothingWindows: this.config.energySmoothingWindows,
      minSpeechLevelDbfs: this.config.minSpeechLevelDbfs,
      useSnrGate: this.config.useSnrGate,
      snrThreshold: this.config.snrThreshold,
      minSnrThreshold: this.config.minSnrThreshold,
      energyRiseThreshold: this.config.energyRiseThreshold,
      maxOnsetLookbackChunks: this.config.maxOnsetLookbackChunks,
      defaultOnsetLookbackChunks: this.config.defaultOnsetLookbackChunks,
      maxHistoryChunks: Math.max(
        this.config.maxHistoryChunks,
        Math.ceil(this.config.ringBufferDurationMs / Math.max(1, this.config.analysisWindowMs)) + 2,
      ),
      minSpeechDurationMs: this.config.minSpeechDurationMs,
      minSilenceDurationMs: this.config.minSilenceDurationMs,
      initialNoiseFloor: this.config.initialNoiseFloor,
      fastAdaptationRate: this.config.fastAdaptationRate,
      slowAdaptationRate: this.config.slowAdaptationRate,
      minBackgroundDurationSec: this.config.minBackgroundDurationSec,
      levelWindowMs: this.config.levelWindowMs,
    };
  }

  private buildTenVadConfig(): Record<string, unknown> {
    return {
      sampleRate: this.sampleRate,
      hopSize: resolveStreamingTimelineChunkFrames(
        this.sampleRate,
        this.config.chunkDurationMs,
      ),
      threshold: this.config.tenVadThreshold,
      confirmationWindowMs: this.config.tenVadConfirmationWindowMs,
      hangoverMs: this.config.tenVadHangoverMs,
      minSpeechDurationMs: this.config.tenVadMinSpeechDurationMs,
      minSilenceDurationMs: this.config.tenVadMinSilenceDurationMs,
      speechPaddingMs: this.config.tenVadSpeechPaddingMs,
    };
  }

  private subscribeTenVad(tenVad: StreamingTenVadLike | null): (() => void) | null {
    return (
      tenVad?.subscribe((event) => {
        if (event.type === 'result') {
          this.emit({
            type: 'metrics',
            payload: this.getSnapshot(),
          });
        }
      }) ?? null
    );
  }

  private replaceTenVad(tenVad: StreamingTenVadLike | null): void {
    this.tenVadUnsubscribe?.();
    this.tenVadUnsubscribe = this.subscribeTenVad(tenVad);
    this.tenVad = tenVad;
  }

  private isTenVadReady(): boolean {
    return this.tenVad?.getStatus().state === 'ready';
  }

  private resolveEffectiveGateMode(): StreamingGateMode {
    if (this.config.gateMode === STREAMING_GATE_MODES.ROUGH_ONLY) {
      return STREAMING_GATE_MODES.ROUGH_ONLY;
    }
    if (!this.config.tenVadEnabled || !this.isTenVadReady()) {
      return STREAMING_GATE_MODES.ROUGH_ONLY;
    }
    return STREAMING_GATE_MODES.TEN_VAD_ONLY;
  }

  private getCurrentConfigOverrides(): Partial<StreamingDetectorConfig> {
    const preset = getStreamingPreset(this.profileId).config;
    const overrides: Partial<StreamingDetectorConfig> = {};
    const mutableOverrides = overrides as Record<string, unknown>;
    for (const [key, value] of Object.entries(this.config) as Array<
      [keyof StreamingDetectorConfig, StreamingDetectorConfig[keyof StreamingDetectorConfig]]
    >) {
      if (preset[key] !== value) {
        mutableOverrides[key] = value;
      }
    }
    return overrides;
  }

  subscribe(listener: (event: StreamingSpeechDetectorEvent) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(event: StreamingSpeechDetectorEvent): void {
    for (const listener of this.listeners) {
      listener(event);
    }
  }

  private recordDecision(message: string, meta: Record<string, unknown> = {}): void {
    const entry = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      at: new Date().toLocaleTimeString(),
      message,
      meta,
    };
    this.recentDecisions = [...this.recentDecisions.slice(-19), entry];
    console.debug(`[StreamingSpeechDetector] ${message}`, meta);
  }

  private debugTrace(message: string, meta: Record<string, unknown> = {}): void {
    const now = Date.now();
    if (now - this.lastDebugTraceAt < 500) {
      return;
    }
    this.lastDebugTraceAt = now;
    console.debug(`[StreamingSpeechDetector] ${message}`, meta);
  }

  async start({ sampleRate }: { readonly sampleRate?: number } = {}): Promise<void> {
    if (this.disposed) {
      throw new Error('StreamingSpeechDetector has been disposed.');
    }

    if (sampleRate && sampleRate !== this.sampleRate) {
      this.sampleRate = sampleRate;
      this.ringBuffer = new AudioRingBuffer({
        sampleRate: this.sampleRate,
        durationSeconds: this.config.ringBufferDurationMs / 1000,
      });
      this.roughGate = new RoughSpeechGate(this.buildRoughGateConfig());
      this.tenVad?.updateConfig(this.buildTenVadConfig());
    }

    this.state = 'listening';
    this.activeSegment = null;
    this.pendingSegmentStartFrame = null;
    this.lastMetrics = null;
    this.lastAcceptanceInfo = null;
    this.lastForegroundResult = null;
    this.lastError = null;
    this.ringBuffer.reset();
    this.roughGate.reset();

    if (this.tenVad) {
      try {
        await this.tenVad.init();
        await this.tenVad.reset();
        this.recordDecision(
          'TEN-VAD ready',
          this.tenVad.getStatus() as unknown as Record<string, unknown>,
        );
      } catch (error) {
        this.lastError = error instanceof Error ? error : new Error(String(error));
        this.recordDecision('TEN-VAD degraded, falling back to rough gate', {
          error: this.lastError.message,
        });
        this.emit({
          type: 'error',
          payload: this.lastError,
        });
      }
    }

    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  processChunk(
    chunk: Float32Array,
    meta: { readonly startFrame?: number; readonly endFrame?: number } = {},
  ): void {
    if (!chunk.length) {
      return;
    }

    const chunkStartFrame = this.ringBuffer.getCurrentFrame();
    if (meta.startFrame !== undefined && meta.startFrame !== chunkStartFrame) {
      throw new RangeError(
        `processChunk startFrame mismatch. Expected ${chunkStartFrame}, got ${meta.startFrame}.`,
      );
    }

    this.ringBuffer.write(chunk);
    this.tenVad?.process(chunk, chunkStartFrame);

    const rough = this.roughGate.process(chunk);
    this.lastMetrics = rough;

    const nowFrame = this.ringBuffer.getCurrentFrame();
    const effectiveGateMode = this.resolveEffectiveGateMode();
    const tenVadReady = this.isTenVadReady();
    const baseFrame = this.ringBuffer.getBaseFrameOffset();
    const prerollFrames = durationMsToAlignedFrameCount(
      this.config.prerollMs,
      this.sampleRate,
      'ceil',
      this.config.chunkDurationMs,
    );

    if (effectiveGateMode === STREAMING_GATE_MODES.ROUGH_ONLY) {
      this.processWithRoughFallback(rough, nowFrame, chunk.length, prerollFrames, baseFrame);
    } else if (tenVadReady) {
      this.processWithTenVad(nowFrame, prerollFrames, baseFrame);
    }

    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  private processWithTenVad(
    nowFrame: number,
    prerollFrames: number,
    baseFrame: number,
  ): void {
    const confirmWindowMs = Math.max(
      this.config.tenVadConfirmationWindowMs,
      this.config.tenVadMinSpeechDurationMs,
    );
    const silenceWindowMs = Math.max(
      this.config.tenVadConfirmationWindowMs,
      this.config.tenVadMinSilenceDurationMs,
    );
    const startSearchFrames = durationMsToAlignedFrameCount(
      this.config.prerollMs
        + this.config.tenVadSpeechPaddingMs
        + confirmWindowMs
        + this.config.chunkDurationMs * 2,
      this.sampleRate,
      'ceil',
      this.config.chunkDurationMs,
    );
    const searchStartFrame = Math.max(baseFrame, nowFrame - startSearchFrames);
    const tenVadConfirmed =
      this.tenVad?.hasRecentSpeech(nowFrame, confirmWindowMs, this.sampleRate) ?? false;
    const tenVadTailHold =
      this.tenVad?.hasRecentSpeech(nowFrame, this.config.tenVadHangoverMs, this.sampleRate) ?? false;
    const tenVadSilence =
      this.tenVad?.hasRecentSilence(nowFrame, silenceWindowMs, this.sampleRate) ?? false;
    const tenVadStart = this.tenVad?.findFirstSpeechFrame(searchStartFrame, nowFrame) ?? null;
    const segmentStartFrame =
      tenVadStart === null
        ? null
        : Math.max(baseFrame, tenVadStart - prerollFrames);

    if (!this.activeSegment) {
      if (segmentStartFrame !== null && !tenVadConfirmed) {
        this.pendingSegmentStartFrame = segmentStartFrame;
        this.state = 'candidate';
      } else if (segmentStartFrame !== null && tenVadConfirmed) {
        this.activateSegment(segmentStartFrame, nowFrame, 'ten-vad');
        this.recordDecision('TEN-VAD segment accepted', {
          startFrame: segmentStartFrame,
          probability: this.tenVad?.getStatus().probability ?? 0,
        });
      } else {
        this.pendingSegmentStartFrame = null;
        if (this.state !== 'idle') {
          this.state = 'listening';
        }
      }
      return;
    }

    this.activeSegment = {
      ...this.activeSegment,
      lastFrame: nowFrame,
    };

    const segmentFrames = nowFrame - this.activeSegment.startFrame;
    const segmentDurationMs = (segmentFrames / this.sampleRate) * 1000;
    if (segmentDurationMs >= this.config.maxSegmentDurationMs) {
      this.finalizeSegment('max-duration', nowFrame);
      return;
    }

    if (tenVadConfirmed) {
      this.state = 'speaking';
      this.emitSpeechUpdate(nowFrame);
      return;
    }

    if (tenVadTailHold) {
      this.state = 'candidate';
      this.emitSpeechUpdate(nowFrame);
      return;
    }

    if (tenVadSilence) {
      this.recordDecision('Speech finalized by TEN-VAD silence', {
        endFrame: nowFrame,
        probability: this.tenVad?.getStatus().probability ?? 0,
      });
      this.finalizeSegment('ten-vad-silence', nowFrame);
      return;
    }

    this.state = 'candidate';
  }

  private processWithRoughFallback(
    rough: RoughSpeechGateWindowResult,
    nowFrame: number,
    chunkLength: number,
    prerollFrames: number,
    baseFrame: number,
  ): void {
    if (!this.activeSegment && rough.speechStart) {
      const fallbackStart = Math.max(
        baseFrame,
        (rough.onsetFrame ?? nowFrame - chunkLength) - prerollFrames,
      );
      this.activateSegment(fallbackStart, nowFrame, 'rough-fallback');
      this.recordDecision('Rough fallback segment accepted', {
        startFrame: fallbackStart,
        levelDbfs: Number(rough.levelDbfs.toFixed(1)),
        snr: Number(rough.snr.toFixed(2)),
      });
      return;
    }

    if (!this.activeSegment) {
      const isNearSpeechThreshold =
        rough.levelDbfs >= this.config.minSpeechLevelDbfs - 6 ||
        rough.snr >= this.config.minSnrThreshold - 1;
      if (isNearSpeechThreshold && !rough.speechStart && !rough.isSpeech) {
        this.debugTrace('Near-threshold audio rejected before rough fallback onset', {
          levelDbfs: Number(rough.levelDbfs.toFixed(1)),
          snr: Number(rough.snr.toFixed(2)),
        });
      }
      return;
    }

    this.activeSegment = {
      ...this.activeSegment,
      lastFrame: nowFrame,
    };
    const segmentFrames = nowFrame - this.activeSegment.startFrame;
    const segmentDurationMs = (segmentFrames / this.sampleRate) * 1000;
    if (segmentDurationMs >= this.config.maxSegmentDurationMs) {
      this.finalizeSegment('max-duration', nowFrame);
      return;
    }

    if (rough.isSpeech) {
      this.state = 'speaking';
      this.emitSpeechUpdate(nowFrame);
      return;
    }

    if (rough.speechEnd) {
      this.recordDecision('Speech finalized after rough fallback silence', {
        endFrame: nowFrame,
        levelDbfs: Number(rough.levelDbfs.toFixed(1)),
      });
      this.finalizeSegment('silence', nowFrame);
    }
  }

  private activateSegment(
    startFrame: number,
    nowFrame: number,
    source: ActiveSegmentState['source'],
  ): void {
    this.pendingSegmentStartFrame = null;
    this.activeSegment = {
      startFrame,
      lastFrame: nowFrame,
      source,
    };
    this.state = 'speaking';
    this.emit({
      type: 'speech-start',
      payload: {
        startFrame,
        sampleRate: this.sampleRate,
      },
    });
  }

  private emitSpeechUpdate(nowFrame: number): void {
    if (!this.activeSegment) {
      return;
    }
    this.emit({
      type: 'speech-update',
      payload: {
        startFrame: this.activeSegment.startFrame,
        endFrame: nowFrame,
        sampleRate: this.sampleRate,
      },
    });
  }

  finalizeSegment(
    reason: string,
    endFrame = this.ringBuffer.getCurrentFrame(),
  ): StreamingDetectorSegment | null {
    if (!this.activeSegment) {
      return null;
    }

    const startFrame = this.activeSegment.startFrame;
    const pcm = this.ringBuffer.read(startFrame, endFrame);
    const tenVadStatus =
      this.tenVad?.getStatus() ?? {
        state: this.config.tenVadEnabled ? 'idle' : 'disabled',
        error: this.lastError?.message ?? null,
        probability: 0,
        speaking: false,
        threshold: this.config.tenVadThreshold,
      };
    const filterResult = scoreSegmentForeground(
      pcm,
      this.sampleRate,
      this.lastMetrics?.noiseFloorDbfs
        ?? 20 * Math.log10(Math.max(this.config.initialNoiseFloor, 0.000001)),
      this.config,
    );
    this.lastForegroundResult = filterResult;
    this.lastAcceptanceInfo = {
      accepted: filterResult.accepted,
      reason: filterResult.reason,
      segmentReason: reason,
      segmentDurationMs: filterResult.durationMs,
      segmentP90Dbfs: filterResult.segmentP90Dbfs,
      onsetP90Dbfs: filterResult.onsetP90Dbfs,
      foregroundDb: filterResult.foregroundDb,
      onsetDb: filterResult.onsetDb,
      shortSpeech: filterResult.shortSpeech,
      longSpeech: filterResult.longSpeech,
      noiseFloorDbfs: filterResult.noiseFloorDbfs,
      source: this.activeSegment.source,
      gateMode: this.resolveEffectiveGateMode(),
      tenVadState: tenVadStatus.state,
      foregroundFilterEnabled: this.config.foregroundFilterEnabled,
    };

    this.emit({
      type: 'speech-end',
      payload: {
        startFrame,
        endFrame,
        sampleRate: this.sampleRate,
      },
    });

    if (!filterResult.accepted) {
      this.recordDecision('Segment rejected by foreground filter', {
        reason: filterResult.reason,
        durationMs: Number(filterResult.durationMs.toFixed(0)),
        foregroundDb: Number(filterResult.foregroundDb.toFixed(1)),
        onsetDb: Number(filterResult.onsetDb.toFixed(1)),
        noiseFloorDbfs: Number(filterResult.noiseFloorDbfs.toFixed(1)),
      });
      this.activeSegment = null;
      this.pendingSegmentStartFrame = null;
      this.state = 'listening';
      return null;
    }

    const segment: StreamingDetectorSegment = {
      startFrame,
      endFrame,
      sampleRate: this.sampleRate,
      reason,
      metadata: {
        profileId: this.profileId,
        rough: this.lastMetrics,
        tenVad: tenVadStatus,
        filter: filterResult,
      },
      readPcm: () => pcm,
    };

    this.recentSegments = [...this.recentSegments.slice(-11), segment];
    this.recordDecision(`Segment ready: ${reason}`, {
      startFrame: segment.startFrame,
      endFrame: segment.endFrame,
      durationSec: Number(
        ((segment.endFrame - segment.startFrame) / segment.sampleRate).toFixed(2),
      ),
      foregroundDb: Number(filterResult.foregroundDb.toFixed(1)),
    });
    this.emit({
      type: 'segment-ready',
      payload: segment,
    });

    this.activeSegment = null;
    this.pendingSegmentStartFrame = null;
    this.state = 'listening';
    return segment;
  }

  flush(reason = 'manual'): StreamingDetectorSegment | null {
    return this.finalizeSegment(reason);
  }

  async stop({
    flush = true,
  }: {
    readonly flush?: boolean;
  } = {}): Promise<StreamingDetectorSegment | null> {
    let segment: StreamingDetectorSegment | null = null;
    if (flush) {
      segment = this.finalizeSegment('stop');
    } else {
      this.activeSegment = null;
      this.pendingSegmentStartFrame = null;
    }
    this.state = 'idle';
    return segment;
  }

  updateConfig(
    partial: StreamingDetectorConfigOverrides & {
      readonly profileId?: string;
    } = {},
  ): void {
    const nextProfileId = partial.profileId ?? this.profileId;
    const nextOverrides = {
      ...this.getCurrentConfigOverrides(),
      ...partial,
    };
    delete nextOverrides.profileId;

    const previousTenVadEnabled = this.config.tenVadEnabled;
    this.profileId = nextProfileId;
    this.config = mergeStreamingConfig(nextProfileId, nextOverrides);
    this.roughGate.updateConfig(this.buildRoughGateConfig());
    if (!previousTenVadEnabled && this.config.tenVadEnabled) {
      const nextTenVad = this.createTenVad(this.buildTenVadConfig(), this.tenVadOptions);
      this.replaceTenVad(nextTenVad);
    } else if (previousTenVadEnabled && !this.config.tenVadEnabled) {
      const previousTenVad = this.tenVad;
      this.replaceTenVad(null);
      void previousTenVad?.dispose().catch(() => undefined);
    } else {
      this.tenVad?.updateConfig(this.buildTenVadConfig());
    }
    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  getSnapshot(): StreamingSpeechDetectorSnapshot {
    const timelineChunkFrames = resolveStreamingTimelineChunkFrames(
      this.sampleRate,
      this.config.chunkDurationMs,
    );
    const waveformFrameSpan = durationMsToAlignedFrameCount(
      this.config.ringBufferDurationMs,
      this.sampleRate,
      'ceil',
      this.config.chunkDurationMs,
    );
    const waveform = this.ringBuffer.getMinMaxPairs(
      Math.max(1, Math.floor(waveformFrameSpan / Math.max(1, timelineChunkFrames))),
      waveformFrameSpan,
    );
    const waveformPointCount = Math.max(1, Math.floor(waveform.minMax.length / 2));
    const defaultNoiseFloorDbfs = 20 * Math.log10(Math.max(this.config.initialNoiseFloor, 0.000001));

    return {
      state: this.state,
      sampleRate: this.sampleRate,
      profileId: this.profileId,
      config: { ...this.config },
      waveform,
      activeSegment: this.activeSegment
        ? {
            startFrame: this.activeSegment.startFrame,
            endFrame: this.ringBuffer.getCurrentFrame(),
          }
        : null,
      pendingSegmentStartFrame: this.pendingSegmentStartFrame,
      recentSegments: this.recentSegments.map((segment) => ({ ...segment })),
      recentDecisions: cloneEntries(this.recentDecisions),
      acceptance: this.lastAcceptanceInfo,
      gate: {
        requestedMode: this.config.gateMode,
        effectiveMode: this.resolveEffectiveGateMode(),
        tenVadReady: this.isTenVadReady(),
      },
      rough: this.lastMetrics
        ? {
            energy: this.lastMetrics.energy,
            snr: this.lastMetrics.snr,
            noiseFloor: this.lastMetrics.noiseFloor,
            noiseFloorDbfs: this.lastMetrics.noiseFloorDbfs,
            backgroundAverage: this.lastMetrics.backgroundAverage,
            backgroundAverageDbfs: this.lastMetrics.backgroundAverageDbfs,
            confirmedSilenceAverage: this.lastMetrics.confirmedSilenceAverage,
            confirmedSilenceAverageDbfs: this.lastMetrics.confirmedSilenceAverageDbfs,
            rejectedCandidateAverage: this.lastMetrics.rejectedCandidateAverage,
            rejectedCandidateAverageDbfs: this.lastMetrics.rejectedCandidateAverageDbfs,
            threshold: this.lastMetrics.threshold,
            thresholdDbfs: this.lastMetrics.thresholdDbfs,
            levelDbfs: this.lastMetrics.levelDbfs,
            levelWindowRms: this.lastMetrics.levelWindowRms,
            levelWindowDbfs: this.lastMetrics.levelWindowDbfs,
            levelWindowMs: this.lastMetrics.levelWindowMs,
            energyPass: this.lastMetrics.energyPass,
            candidateReason: this.lastMetrics.candidateReason,
            snrThreshold: this.lastMetrics.snrThreshold,
            minSnrThreshold: this.lastMetrics.minSnrThreshold,
            useSnrGate: this.config.useSnrGate,
            snrPass: this.lastMetrics.snrPass,
            isSpeech: this.lastMetrics.isSpeech,
            recent: this.roughGate.getRecentChunks(),
            timeline: this.roughGate.getTimeline(
              waveform.startFrame,
              waveform.endFrame,
              waveformPointCount,
            ),
          }
        : {
            energy: 0,
            snr: 0,
            noiseFloor: this.config.initialNoiseFloor,
            noiseFloorDbfs: defaultNoiseFloorDbfs,
            backgroundAverage: this.config.initialNoiseFloor,
            backgroundAverageDbfs: defaultNoiseFloorDbfs,
            confirmedSilenceAverage: this.config.initialNoiseFloor,
            confirmedSilenceAverageDbfs: defaultNoiseFloorDbfs,
            rejectedCandidateAverage: this.config.initialNoiseFloor,
            rejectedCandidateAverageDbfs: defaultNoiseFloorDbfs,
            threshold: 10 ** (this.config.minSpeechLevelDbfs / 20),
            thresholdDbfs: this.config.minSpeechLevelDbfs,
            levelDbfs: -100,
            levelWindowRms: 0,
            levelWindowDbfs: -100,
            levelWindowMs: this.config.levelWindowMs,
            energyPass: false,
            candidateReason: 'none',
            snrThreshold: this.config.snrThreshold,
            minSnrThreshold: this.config.minSnrThreshold,
            useSnrGate: this.config.useSnrGate,
            snrPass: false,
            isSpeech: false,
            recent: [],
            timeline: [],
          },
      foreground: {
        enabled: this.config.foregroundFilterEnabled,
        noiseFloorDbfs: this.lastMetrics?.noiseFloorDbfs ?? defaultNoiseFloorDbfs,
        foregroundMinDb: this.config.foregroundMinDb,
        onsetMinDb: this.config.foregroundOnsetMinDb,
        longMinDb: this.config.foregroundLongMinDb,
        lastResult: this.lastForegroundResult as unknown as Record<string, unknown> | null,
      },
      tenVad:
        this.tenVad?.getStatus() ?? {
          state: this.config.tenVadEnabled ? 'idle' : 'disabled',
          error: this.lastError?.message ?? null,
          probability: 0,
          speaking: false,
          threshold: this.config.tenVadThreshold,
        },
      warnings: this.buildWarnings(),
      error: this.lastError?.message ?? null,
    };
  }

  private buildWarnings(): string[] {
    const warnings: string[] = [];
    const tenVadStatus = this.tenVad?.getStatus();
    const effectiveGateMode = this.resolveEffectiveGateMode();
    const tenVadUnavailableForRequestedGate =
      tenVadStatus?.state === 'degraded' || tenVadStatus?.state === 'disabled';

    if (tenVadStatus?.state === 'degraded') {
      warnings.push('TEN-VAD is degraded. Speech detection is running on rough fallback only.');
    }

    if (
      this.config.gateMode !== STREAMING_GATE_MODES.ROUGH_ONLY &&
      effectiveGateMode === STREAMING_GATE_MODES.ROUGH_ONLY &&
      tenVadUnavailableForRequestedGate
    ) {
      warnings.push('TEN-VAD-first segmentation is unavailable, so detection is falling back to rough energy gating.');
    }

    if (!this.config.foregroundFilterEnabled) {
      warnings.push('Foreground filtering is disabled. Quiet background speech may be accepted.');
    }

    return warnings;
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.tenVadUnsubscribe?.();
    this.tenVadUnsubscribe = null;
    await this.tenVad?.dispose();
    this.listeners.clear();
  }
}
