import { AudioRingBuffer } from './realtime.js';
import {
  DEFAULT_ROUGH_GATE_CONFIG,
  type RoughSpeechGateConfig,
} from './rough-gate-config.js';
import {
  RoughSpeechGate,
  type RoughSpeechGateWindowResult,
} from './rough-speech-gate.js';
import {
  STREAMING_PROFILE_IDS,
  STREAMING_PRESETS,
  getStreamingPreset,
  isStreamingConfigEqual,
  listStreamingPresets,
  mergeStreamingConfig,
  resolveDefaultMicMode,
  resolveStreamingProfileId,
  type StreamingDetectorConfig,
  type StreamingDetectorPreset,
  type StreamingProfileId,
} from './streaming-config.js';

export {
  DEFAULT_ROUGH_GATE_CONFIG,
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
  readonly rough: Record<string, unknown>;
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
  readonly reason: string;
}

export interface StreamingSpeechDetectorOptions {
  readonly profileId?: string;
  readonly config?: Partial<StreamingDetectorConfig> & { readonly energyThreshold?: number };
  readonly isRealtimeEouModel?: boolean;
  readonly tenVadFactory?: (
    config: Record<string, unknown>,
    options?: unknown,
  ) => StreamingTenVadLike;
  readonly tenVadOptions?: unknown;
}

function cloneSegments<T extends Record<string, unknown>>(segments: readonly T[]): T[] {
  return segments.map((segment) => ({ ...segment }));
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
  private lastError: Error | null = null;
  private disposed = false;
  private tenVadOptions: unknown;
  private tenVadUnsubscribe: (() => void) | null = null;

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
      maxHistoryChunks: this.config.maxHistoryChunks,
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
      threshold: this.config.tenVadThreshold,
      confirmationWindowMs: this.config.tenVadConfirmationWindowMs,
      hangoverMs: this.config.tenVadHangoverMs,
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
    }

    this.state = 'listening';
    this.activeSegment = null;
    this.pendingSegmentStartFrame = null;
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
    if (!chunk.length) return;

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
    const isNearSpeechThreshold =
      rough.levelDbfs >= this.config.minSpeechLevelDbfs - 6 ||
      rough.snr >= this.config.minSnrThreshold - 1;

    if (rough.speechStart) {
      const fallbackStart = Math.max(
        this.ringBuffer.getBaseFrameOffset(),
        rough.onsetFrame ?? nowFrame - chunk.length,
      );
      const roughStart = Math.max(
        this.ringBuffer.getBaseFrameOffset(),
        fallbackStart - Math.ceil((this.config.prerollMs / 1000) * this.sampleRate),
      );
      const tenVadSummary = this.tenVad?.getWindowSummary(
        nowFrame,
        this.config.tenVadConfirmationWindowMs,
        this.sampleRate,
      );
      const tenVadStart = this.tenVad?.findFirstSpeechFrame(roughStart, nowFrame) ?? null;
      const startFrame = Math.max(this.ringBuffer.getBaseFrameOffset(), tenVadStart ?? roughStart);
      if (
        this.tenVad &&
        this.tenVad.getStatus().state === 'ready' &&
        tenVadStart === null &&
        !this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        )
      ) {
        this.pendingSegmentStartFrame = startFrame;
        this.state = 'candidate';
        this.recordDecision('Rough onset waiting for TEN-VAD confirmation', {
          startFrame,
          reason: rough.candidateReason,
          energy: Number(rough.energy.toFixed(4)),
          levelDbfs: Number(rough.levelDbfs.toFixed(1)),
          snr: Number(rough.snr.toFixed(2)),
          tenVadState: this.tenVad.getStatus() as unknown as Record<string, unknown>,
          tenVadSummary: tenVadSummary as unknown as Record<string, unknown>,
        });
      } else {
        this.activateSegment(startFrame, nowFrame, rough, tenVadStart, tenVadSummary ?? null);
      }
    }

    if (
      !this.activeSegment &&
      this.pendingSegmentStartFrame !== null &&
      rough.isSpeech &&
      (!this.tenVad ||
        this.tenVad.getStatus().state !== 'ready' ||
        this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ))
    ) {
      this.activateSegment(
        this.pendingSegmentStartFrame,
        nowFrame,
        rough,
        this.pendingSegmentStartFrame,
        this.tenVad?.getWindowSummary(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ) ?? null,
      );
    }

    if (
      this.pendingSegmentStartFrame !== null &&
      !rough.isSpeech &&
      !rough.speechStart &&
      (!this.tenVad ||
        this.tenVad.getStatus().state !== 'ready' ||
        !this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ))
    ) {
      this.recordDecision('Candidate rejected by TEN-VAD / duration guard', {
        startFrame: this.pendingSegmentStartFrame,
        reason: rough.candidateReason,
        energy: Number(rough.energy.toFixed(4)),
        levelDbfs: Number(rough.levelDbfs.toFixed(1)),
        snr: Number(rough.snr.toFixed(2)),
        tenVadSummary:
          this.tenVad?.getWindowSummary(
            nowFrame,
            this.config.tenVadConfirmationWindowMs,
            this.sampleRate,
          ) ?? null,
      });
      this.pendingSegmentStartFrame = null;
    }

    if (this.activeSegment) {
      this.activeSegment = {
        ...this.activeSegment,
        lastFrame: nowFrame,
      };

      const segmentFrames = nowFrame - this.activeSegment.startFrame;
      const segmentDurationMs = (segmentFrames / this.sampleRate) * 1000;
      if (segmentDurationMs >= this.config.maxSegmentDurationMs) {
        this.finalizeSegment('max-duration', nowFrame);
      } else if (rough.isSpeech) {
        this.state = 'speaking';
        this.emit({
          type: 'speech-update',
          payload: {
            startFrame: this.activeSegment.startFrame,
            endFrame: nowFrame,
            sampleRate: this.sampleRate,
          },
        });
      } else if (rough.speechEnd) {
        const tenVadAllowsEnd =
          !this.tenVad ||
          this.tenVad.getStatus().state !== 'ready' ||
          this.tenVad.hasRecentSilence(
            nowFrame,
            this.config.tenVadConfirmationWindowMs,
            this.sampleRate,
          );

        if (tenVadAllowsEnd) {
          this.recordDecision('Speech finalized after silence', {
            endFrame: nowFrame,
            energy: Number(rough.energy.toFixed(4)),
            levelDbfs: Number(rough.levelDbfs.toFixed(1)),
            snr: Number(rough.snr.toFixed(2)),
          });
          this.finalizeSegment('silence', nowFrame);
        } else {
          this.state = 'candidate';
          this.recordDecision('Silence rejected by TEN-VAD hangover', {
            endFrame: nowFrame,
            probability: this.tenVad?.getStatus().probability ?? 0,
          });
        }
      } else if (
        this.tenVad &&
        this.tenVad.getStatus().state === 'ready' &&
        this.tenVad.hasRecentSpeech(nowFrame, this.config.tenVadHangoverMs, this.sampleRate)
      ) {
        this.state = 'candidate';
        this.recordDecision('Speech tail held by TEN-VAD hangover', {
          endFrame: nowFrame,
          probability: this.tenVad.getStatus().probability,
        });
      }
    } else if (isNearSpeechThreshold && !rough.speechStart && !rough.isSpeech) {
      this.recordDecision('Candidate rejected before speech start', {
        reason: rough.candidateReason,
        energy: Number(rough.energy.toFixed(4)),
        levelDbfs: Number(rough.levelDbfs.toFixed(1)),
        snr: Number(rough.snr.toFixed(2)),
        energyThresholdDbfs: this.config.minSpeechLevelDbfs,
        useSnrGate: this.config.useSnrGate,
        snrThreshold: this.config.snrThreshold,
        minSpeechDurationMs: this.config.minSpeechDurationMs,
      });
    }

    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
  }

  private activateSegment(
    startFrame: number,
    nowFrame: number,
    rough: RoughSpeechGateWindowResult,
    tenVadStart: number | null,
    tenVadSummary: StreamingTenVadWindowSummary | null,
  ): void {
    this.pendingSegmentStartFrame = null;
    this.activeSegment = {
      startFrame,
      lastFrame: nowFrame,
      reason: 'speech-start',
    };
    this.state = 'speaking';
    this.lastAcceptanceInfo = {
      reason:
        rough.candidateReason === 'energy-threshold'
          ? tenVadStart !== null
            ? 'Accepted because energy crossed threshold and TEN-VAD confirmed speech'
            : 'Accepted because energy crossed threshold'
          : 'Accepted by rough speech gate',
      energy: rough.energy,
      threshold: rough.threshold,
      thresholdDbfs: rough.thresholdDbfs,
      levelDbfs: rough.levelDbfs,
      levelWindowDbfs: rough.levelWindowDbfs,
      energyPass: rough.energyPass,
      snr: rough.snr,
      snrPass: rough.snrPass,
      useSnrGate: this.config.useSnrGate,
      tenVadConfirmed: tenVadStart !== null,
      tenVadState: this.tenVad?.getStatus().state ?? 'disabled',
    };
    this.recordDecision('Speech accepted', {
      startFrame,
      reason: rough.candidateReason,
      energy: Number(rough.energy.toFixed(4)),
      levelDbfs: Number(rough.levelDbfs.toFixed(1)),
      snr: Number(rough.snr.toFixed(2)),
      tenVadStart,
      tenVadSummary: tenVadSummary as unknown as Record<string, unknown> | null,
    });
    this.emit({
      type: 'speech-start',
      payload: {
        startFrame,
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
    const segment: StreamingDetectorSegment = {
      startFrame,
      endFrame,
      sampleRate: this.sampleRate,
      reason,
      metadata: {
        profileId: this.profileId,
        rough: this.lastMetrics,
        tenVad:
          this.tenVad?.getStatus() ?? {
            state: 'disabled',
            error: null,
            probability: 0,
            speaking: false,
            threshold: this.config.tenVadThreshold,
          },
      },
      readPcm: () => this.ringBuffer.read(startFrame, endFrame),
    };

    this.recentSegments = [...this.recentSegments.slice(-11), segment];
    this.recordDecision(`Segment ready: ${reason}`, {
      startFrame: segment.startFrame,
      endFrame: segment.endFrame,
      durationSec: Number(
        ((segment.endFrame - segment.startFrame) / segment.sampleRate).toFixed(2),
      ),
    });
    this.emit({
      type: 'speech-end',
      payload: {
        startFrame: segment.startFrame,
        endFrame: segment.endFrame,
        sampleRate: segment.sampleRate,
      },
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
    this.emit({
      type: 'metrics',
      payload: this.getSnapshot(),
    });
    return segment;
  }

  updateConfig(
    partial: Partial<StreamingDetectorConfig> & { readonly profileId?: string } = {},
  ): void {
    const nextProfileId = partial.profileId ?? this.profileId;
    const carriedOverrides = this.getCurrentConfigOverrides();
    const nextOverrides = {
      ...carriedOverrides,
      ...partial,
    };
    delete (nextOverrides as { profileId?: string }).profileId;

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
    const waveform = this.ringBuffer.getMinMaxPairs(
      this.config.waveformPointCount,
      Math.ceil((this.config.ringBufferDurationMs / 1000) * this.sampleRate),
    );

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
      recentDecisions: cloneSegments(this.recentDecisions),
      acceptance: this.lastAcceptanceInfo,
      rough: this.lastMetrics
        ? {
            energy: this.lastMetrics.energy,
            snr: this.lastMetrics.snr,
            noiseFloor: this.lastMetrics.noiseFloor,
            noiseFloorDbfs: this.lastMetrics.noiseFloorDbfs,
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
          }
        : {
            energy: 0,
            snr: 0,
            noiseFloor: this.config.initialNoiseFloor,
            noiseFloorDbfs: 20 * Math.log10(Math.max(this.config.initialNoiseFloor, 0.000001)),
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

    if (tenVadStatus?.state === 'degraded') {
      warnings.push('TEN-VAD is degraded. Speech detection is running on rough energy gating only.');
    }

    if (this.state === 'candidate' && tenVadStatus?.state === 'ready' && !tenVadStatus.speaking) {
      warnings.push('TEN-VAD has not confirmed speech yet. Candidate onset is still being held.');
    }

    if (
      this.lastAcceptanceInfo &&
      this.lastAcceptanceInfo.tenVadConfirmed === false &&
      tenVadStatus?.state !== 'degraded'
    ) {
      warnings.push(
        'Latest accepted speech came from the rough gate without TEN-VAD confirmation.',
      );
    }

    if (this.lastMetrics && !this.config.useSnrGate) {
      warnings.push(
        'SNR is shown for diagnostics only. Speech triggering currently uses the dBFS speech-level threshold.',
      );
    }

    return warnings;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.tenVadUnsubscribe?.();
    this.tenVadUnsubscribe = null;
    await this.tenVad?.dispose();
    this.listeners.clear();
  }
}
