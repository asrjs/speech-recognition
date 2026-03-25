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
  type SegmentForegroundFilterConfig,
  type SegmentForegroundFilterResult,
} from './segment-foreground-filter.js';
import {
  PARAKEET_SEGMENTATION_PRESETS,
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
  type StreamingDetectorPreset,
  type StreamingGateMode,
  type StreamingProfileId,
} from './streaming-config.js';
import { amplitudeToDbfs } from './noise-floor.js';

export {
  DEFAULT_ROUGH_GATE_CONFIG,
  PARAKEET_SEGMENTATION_PRESETS,
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
  StreamingDetectorPreset,
  StreamingGateMode,
  StreamingProfileId,
};

export interface StreamingTenVadStatus {
  readonly state: 'idle' | 'initializing' | 'ready' | 'degraded' | 'disabled';
  readonly error: string | null;
  readonly probability: number;
  readonly speaking: boolean;
  readonly threshold: number;
}

export interface StreamingTenVadRecentHop {
  readonly startFrame: number;
  readonly endFrame: number;
  readonly probability: number;
  readonly rawSpeaking?: boolean;
  readonly speaking: boolean;
  readonly createdAt?: number;
}

export interface StreamingTenVadWindowSummary {
  readonly totalHops: number;
  readonly speechHopCount: number;
  readonly nonSpeechHopCount: number;
  readonly maxConsecutiveSpeech: number;
  readonly maxProbability: number;
  readonly recent: readonly StreamingTenVadRecentHop[];
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
    readonly logicalStartFrame: number;
    readonly logicalEndFrame: number;
    readonly source: 'ten-vad' | 'rough-fallback';
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
    readonly noiseFloor: number;
    readonly noiseFloorDbfs: number;
    readonly energyThreshold: number;
    readonly energyThresholdDbfs: number;
    readonly minSpeechDurationMs: number;
    readonly minEnergyPerSecond: number;
    readonly minEnergyIntegral: number;
    readonly adaptiveEnergyThresholdsEnabled: boolean;
    readonly activeResult: Record<string, unknown> | null;
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

interface FinalizedSegmentBounds {
  readonly logicalStartFrame: number;
  readonly logicalEndFrame: number;
  readonly extractedStartFrame: number;
  readonly extractedEndFrame: number;
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

function createDisabledTenVadStatus(
  config: StreamingDetectorConfig,
  error: Error | null,
): StreamingTenVadStatus {
  return {
    state: config.tenVadEnabled ? 'idle' : 'disabled',
    error: error?.message ?? null,
    probability: 0,
    speaking: false,
    threshold: config.tenVadThreshold,
  };
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
  private lastAcceptedLogicalEndFrame: number | null = null;

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
      energyThreshold: this.config.energyThreshold,
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

  private buildForegroundFilterConfig(): SegmentForegroundFilterConfig {
    return {
      foregroundFilterEnabled: this.config.foregroundFilterEnabled,
      minSpeechDurationMs: this.config.minSpeechDurationMs,
      minEnergyPerSecond: this.config.minEnergyPerSecond,
      minEnergyIntegral: this.config.minEnergyIntegral,
      useAdaptiveEnergyThresholds: this.config.useAdaptiveEnergyThresholds,
      adaptiveEnergyIntegralFactor: this.config.adaptiveEnergyIntegralFactor,
      adaptiveEnergyPerSecondFactor: this.config.adaptiveEnergyPerSecondFactor,
      minAdaptiveEnergyIntegral: this.config.minAdaptiveEnergyIntegral,
      minAdaptiveEnergyPerSecond: this.config.minAdaptiveEnergyPerSecond,
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

  private getTenVadStatus(): StreamingTenVadStatus {
    return this.tenVad?.getStatus() ?? createDisabledTenVadStatus(this.config, this.lastError);
  }

  private resolveEffectiveGateMode(): StreamingGateMode {
    return STREAMING_GATE_MODES.ROUGH_ONLY;
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
    this.lastAcceptedLogicalEndFrame = null;
    this.ringBuffer.reset();
    this.roughGate.reset();

    if (this.tenVad) {
      try {
        await this.tenVad.init();
        await this.tenVad.reset();
        this.recordDecision(
          'TEN-VAD ready for diagnostics',
          this.tenVad.getStatus() as unknown as Record<string, unknown>,
        );
      } catch (error) {
        this.lastError = error instanceof Error ? error : new Error(String(error));
        this.recordDecision('TEN-VAD degraded; segmentation remains on rough gate', {
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
    const baseFrame = this.ringBuffer.getBaseFrameOffset();

    if (!this.activeSegment && rough.speechStart) {
      const logicalStartFrame = Math.max(
        baseFrame,
        rough.onsetFrame ?? rough.chunkStartFrame,
      );
      this.activateSegment(logicalStartFrame, nowFrame, 'rough-fallback');
      this.recordDecision('Rough gate speech start', {
        startFrame: logicalStartFrame,
        energy: Number(rough.energy.toFixed(3)),
        noiseFloor: Number(rough.noiseFloor.toFixed(5)),
        snr: Number(rough.snr.toFixed(2)),
      });
    }

    if (!this.activeSegment && !rough.speechStart) {
      const isNearThreshold =
        rough.energy >= this.config.energyThreshold * 0.8
        || rough.snr >= this.config.minSnrThreshold;
      if (isNearThreshold && !rough.isSpeech) {
        this.debugTrace('Near-threshold audio stayed below live start trigger', {
          energy: Number(rough.energy.toFixed(3)),
          threshold: Number(this.config.energyThreshold.toFixed(3)),
          snr: Number(rough.snr.toFixed(2)),
        });
      }
    }

    if (this.activeSegment) {
      this.activeSegment = {
        ...this.activeSegment,
        lastFrame: nowFrame,
      };

      const segmentFrames = nowFrame - this.activeSegment.startFrame;
      const segmentDurationMs = (segmentFrames / this.sampleRate) * 1000;
      if (segmentDurationMs >= this.config.maxSegmentDurationMs) {
        const finalized = this.finalizeSegment('max-duration', nowFrame);
        if (rough.isSpeech) {
          const nextStartFrame = Math.max(baseFrame, rough.chunkStartFrame);
          this.activateSegment(nextStartFrame, nowFrame, 'rough-fallback');
          this.recordDecision('Long speech split into a new rough segment', {
            previousAccepted: finalized?.metadata.filter?.accepted ?? false,
            nextStartFrame,
          });
        }
      } else if (rough.isSpeech) {
        this.state = 'speaking';
        this.emitSpeechUpdate(nowFrame);
      } else if (rough.speechEnd) {
        this.recordDecision('Rough gate finalized speech after sustained silence', {
          endFrame: nowFrame,
          energy: Number(rough.energy.toFixed(3)),
          noiseFloor: Number(rough.noiseFloor.toFixed(5)),
        });
        this.finalizeSegment('silence', nowFrame);
      } else {
        this.state = 'candidate';
      }
    }

    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
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

  private finalizeBounds(
    logicalStartFrame: number,
    logicalEndFrame: number,
  ): FinalizedSegmentBounds {
    const baseFrame = this.ringBuffer.getBaseFrameOffset();
    const currentFrame = this.ringBuffer.getCurrentFrame();
    const prerollFrames = durationMsToAlignedFrameCount(
      this.config.prerollMs,
      this.sampleRate,
      'ceil',
      this.config.chunkDurationMs,
    );
    const overlapFrames = durationMsToAlignedFrameCount(
      this.config.overlapDurationMs,
      this.sampleRate,
      'ceil',
      this.config.chunkDurationMs,
    );
    const hangoverFrames = durationMsToAlignedFrameCount(
      this.config.speechHangoverMs,
      this.sampleRate,
      'ceil',
      this.config.chunkDurationMs,
    );

    let extractedStartFrame = Math.max(baseFrame, logicalStartFrame - prerollFrames);
    if (this.lastAcceptedLogicalEndFrame !== null) {
      const overlappedStartFrame = this.lastAcceptedLogicalEndFrame - overlapFrames;
      extractedStartFrame = Math.max(baseFrame, Math.min(extractedStartFrame, overlappedStartFrame));
    }

    const clampedLogicalStartFrame = Math.max(baseFrame, logicalStartFrame);
    const clampedLogicalEndFrame = Math.max(
      clampedLogicalStartFrame + 1,
      Math.min(currentFrame, logicalEndFrame),
    );
    const extractedEndFrame = Math.max(
      extractedStartFrame + 1,
      Math.min(currentFrame, clampedLogicalEndFrame + hangoverFrames),
    );

    return {
      logicalStartFrame: clampedLogicalStartFrame,
      logicalEndFrame: clampedLogicalEndFrame,
      extractedStartFrame,
      extractedEndFrame,
    };
  }

  finalizeSegment(
    reason: string,
    endFrame = this.ringBuffer.getCurrentFrame(),
  ): StreamingDetectorSegment | null {
    if (!this.activeSegment) {
      return null;
    }

    const activeSegment = this.activeSegment;
    const bounds = this.finalizeBounds(activeSegment.startFrame, endFrame);
    const pcm = this.ringBuffer.read(bounds.extractedStartFrame, bounds.extractedEndFrame);
    const tenVadStatus = this.getTenVadStatus();
    const filterResult = scoreSegmentForeground(
      pcm,
      this.sampleRate,
      this.lastMetrics?.noiseFloor ?? this.config.initialNoiseFloor,
      this.buildForegroundFilterConfig(),
      {
        noiseWindowFrames: this.lastMetrics?.analysisWindowFrames ?? null,
      },
    );
    this.lastForegroundResult = filterResult;
    this.lastAcceptanceInfo = {
      accepted: filterResult.accepted,
      reason: filterResult.reason,
      segmentReason: reason,
      source: activeSegment.source,
      logicalStartFrame: bounds.logicalStartFrame,
      logicalEndFrame: bounds.logicalEndFrame,
      extractedStartFrame: bounds.extractedStartFrame,
      extractedEndFrame: bounds.extractedEndFrame,
      extractedDurationMs: Number(filterResult.durationMs.toFixed(1)),
      noiseFloor: filterResult.noiseFloor,
      noiseFloorDbfs: filterResult.noiseFloorDbfs,
      normalizedPowerAt16k: filterResult.normalizedPowerAt16k,
      normalizedEnergyIntegralAt16k: filterResult.normalizedEnergyIntegralAt16k,
      minEnergyPerSecondThreshold: filterResult.minEnergyPerSecondThreshold,
      minEnergyIntegralThreshold: filterResult.minEnergyIntegralThreshold,
      adaptiveThresholds: filterResult.usedAdaptiveThresholds,
      gateMode: this.resolveEffectiveGateMode(),
      tenVadState: tenVadStatus.state,
    };

    this.emit({
      type: 'speech-end',
      payload: {
        startFrame: bounds.extractedStartFrame,
        endFrame: bounds.extractedEndFrame,
        sampleRate: this.sampleRate,
      },
    });

    this.activeSegment = null;
    this.pendingSegmentStartFrame = null;
    this.state = 'listening';

    if (!filterResult.accepted) {
      this.recordDecision('Segment rejected by final energy gate', {
        reason: filterResult.reason,
        durationMs: Number(filterResult.durationMs.toFixed(0)),
        normalizedPowerAt16k: Number(filterResult.normalizedPowerAt16k.toFixed(2)),
        normalizedEnergyIntegralAt16k: Number(
          filterResult.normalizedEnergyIntegralAt16k.toFixed(2),
        ),
      });
      return null;
    }

    const segment: StreamingDetectorSegment = {
      startFrame: bounds.extractedStartFrame,
      endFrame: bounds.extractedEndFrame,
      sampleRate: this.sampleRate,
      reason,
      metadata: {
        profileId: this.profileId,
        rough: this.lastMetrics,
        tenVad: tenVadStatus,
        filter: filterResult,
        logicalStartFrame: bounds.logicalStartFrame,
        logicalEndFrame: bounds.logicalEndFrame,
        source: activeSegment.source,
      },
      readPcm: () => pcm,
    };

    this.lastAcceptedLogicalEndFrame = bounds.logicalEndFrame;
    this.recentSegments = [...this.recentSegments.slice(-11), segment];
    this.recordDecision(`Segment ready: ${reason}`, {
      startFrame: segment.startFrame,
      endFrame: segment.endFrame,
      logicalEndFrame: bounds.logicalEndFrame,
      durationSec: Number(((segment.endFrame - segment.startFrame) / segment.sampleRate).toFixed(2)),
      normalizedPowerAt16k: Number(filterResult.normalizedPowerAt16k.toFixed(2)),
    });
    this.emit({
      type: 'segment-ready',
      payload: segment,
    });
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
      this.state = 'listening';
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
    const defaultNoiseFloorDbfs = amplitudeToDbfs(this.config.initialNoiseFloor);
    const lastNoiseFloor = this.lastMetrics?.noiseFloor ?? this.config.initialNoiseFloor;

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
            rawEnergy: this.lastMetrics.rawEnergy,
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
            rawEnergy: 0,
            snr: 0,
            noiseFloor: this.config.initialNoiseFloor,
            noiseFloorDbfs: defaultNoiseFloorDbfs,
            backgroundAverage: this.config.initialNoiseFloor,
            backgroundAverageDbfs: defaultNoiseFloorDbfs,
            confirmedSilenceAverage: this.config.initialNoiseFloor,
            confirmedSilenceAverageDbfs: defaultNoiseFloorDbfs,
            rejectedCandidateAverage: this.config.initialNoiseFloor,
            rejectedCandidateAverageDbfs: defaultNoiseFloorDbfs,
            threshold: this.config.energyThreshold,
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
        noiseFloor: lastNoiseFloor,
        noiseFloorDbfs: this.lastMetrics?.noiseFloorDbfs ?? defaultNoiseFloorDbfs,
        energyThreshold: this.config.energyThreshold,
        energyThresholdDbfs: this.config.minSpeechLevelDbfs,
        minSpeechDurationMs: this.config.minSpeechDurationMs,
        minEnergyPerSecond: this.config.minEnergyPerSecond,
        minEnergyIntegral: this.config.minEnergyIntegral,
        adaptiveEnergyThresholdsEnabled: this.config.useAdaptiveEnergyThresholds,
        activeResult: null,
        lastResult: this.lastForegroundResult as unknown as Record<string, unknown> | null,
      },
      tenVad: this.getTenVadStatus(),
      warnings: this.buildWarnings(),
      error: this.lastError?.message ?? null,
    };
  }

  private buildWarnings(): string[] {
    const warnings: string[] = [];
    const tenVadStatus = this.tenVad?.getStatus();

    if (tenVadStatus?.state === 'degraded') {
      warnings.push('TEN-VAD is degraded. Runtime plots may be incomplete.');
    }

    if (this.config.tenVadEnabled) {
      warnings.push('Speech segmentation is running on the rough gate. TEN-VAD is diagnostics-only.');
    }

    if (this.config.gateMode !== STREAMING_GATE_MODES.ROUGH_ONLY) {
      warnings.push('Requested TEN-VAD-first gating is ignored by this Parakeet-style detector port.');
    }

    if (!this.config.foregroundFilterEnabled) {
      warnings.push('Final segment energy rejection is disabled. Quiet background audio may be accepted.');
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
