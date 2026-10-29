import { AudioRingBuffer } from './realtime.js';
import { subscribeToAbortSignal } from '../io/abort.js';

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
  STREAMING_VAD_BACKENDS,
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
  type StreamingVadBackend,
  type StreamingProfileId,
} from './streaming-config.js';
import type {
  StreamingTenVadStatus,
  StreamingTenVadLike,
  StreamingDetectorSegment,
  StreamingSpeechDetectorSnapshot,
  StreamingSpeechDetectorEvent,
  StreamingSpeechDetectorOptions,
} from './streaming-detector-types.js';
import {
  buildAcceptanceInfo,
  buildDetectorWarnings,
  buildForegroundSnapshot,
  buildRoughSnapshot,
  cloneEntries,
  createDisabledTenVadStatus,
  getStreamingVadBackendName,
} from './streaming-detector-snapshot.js';
import {
  buildForegroundFilterConfig,
  buildRoughGateConfig,
  buildTenVadConfig,
  getConfigOverrides,
  createStreamingVadInitAbortedError,
  isStreamingVadInitAborted,
  mergeConfigOverrides,
  resolveEffectiveGateMode,
} from './streaming-detector-gate-config.js';
import {
  computeFinalizedSegmentBounds,
  isQuietChunk,
  type ActiveSegmentState,
  type FinalizedSegmentBounds,
} from './streaming-segment-bounds.js';

export {
  STREAMING_GATE_MODES,
  DEFAULT_ROUGH_GATE_CONFIG,
  PARAKEET_SEGMENTATION_PRESETS,
  STREAMING_VAD_BACKENDS,
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
  StreamingGateMode,
  RoughSpeechGateConfig,
  RoughSpeechGateWindowResult,
  StreamingDetectorConfig,
  StreamingDetectorConfigOverrides,
  StreamingDetectorPreset,
  StreamingVadBackend,
  StreamingProfileId,
};
export type * from './streaming-detector-types.js';

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
  private lastFinalizedLogicalEndFrame = 0;
  private continuationAfterMaxDuration = false;
  private readonly initSignal?: { readonly aborted: boolean } | null;
  private startAbort: AbortController | null = null;
  private startCancelled = false;

  private getVadBackendName(): string {
    return getStreamingVadBackendName(this.config);
  }

  constructor(options: StreamingSpeechDetectorOptions = {}) {
    this.createTenVad =
      options.tenVadFactory ??
      (() => {
        throw new Error(`StreamingSpeechDetector requires a tenVadFactory when ${this.getVadBackendName()} is enabled.`);
      });
    this.profileId =
      options.profileId ?? resolveStreamingProfileId(options.isRealtimeEouModel === true);
    this.config = mergeStreamingConfig(this.profileId, options.config);
    if ((options.config as unknown as Record<string, unknown> | undefined)?.vadMode !== undefined) {
      (this.config as unknown as Record<string, unknown>).vadMode = (options.config as unknown as Record<string, unknown>).vadMode;
    }
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
    this.initSignal = options.signal;
    this.tenVadUnsubscribe = this.subscribeTenVad(this.tenVad);
  }

  private buildRoughGateConfig(): RoughSpeechGateConfig {
    return buildRoughGateConfig(this.config, this.sampleRate);
  }

  private buildTenVadConfig(): Record<string, unknown> {
    return buildTenVadConfig(this.config, this.sampleRate);
  }

  private buildForegroundFilterConfig(): SegmentForegroundFilterConfig {
    return buildForegroundFilterConfig(this.config, this.continuationAfterMaxDuration);
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

  private get tenVadState(): StreamingTenVadStatus['state'] {
    return this.getTenVadStatus().state;
  }

  private resolveEffectiveGateMode(): StreamingGateMode {
    return resolveEffectiveGateMode(this.config, this.tenVad !== null, this.tenVadState);
  }

  private getCurrentConfigOverrides(): Partial<StreamingDetectorConfig> {
    return getConfigOverrides(this.profileId, this.config);
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

  async start({
    sampleRate,
    signal,
  }: {
    readonly sampleRate?: number;
    readonly signal?: { readonly aborted: boolean } | null;
  } = {}): Promise<void> {
    if (this.disposed) {
      throw new Error('StreamingSpeechDetector has been disposed.');
    }

    const callerSignal = signal ?? this.initSignal;
    this.startCancelled = false;
    const startAbort = new AbortController();
    this.startAbort = startAbort;
    const onCallerAbort = () => startAbort.abort();
    const cleanupCallerAbort = subscribeToAbortSignal(callerSignal, onCallerAbort);
    if (callerSignal?.aborted) {
      startAbort.abort();
    }

    try {
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
      this.lastFinalizedLogicalEndFrame = 0;
      this.continuationAfterMaxDuration = false;
      this.ringBuffer.reset();
      this.roughGate.reset();

      if (this.tenVad) {
        try {
          await this.tenVad.init(startAbort.signal);
          if (this.startCancelled || this.disposed) {
            return;
          }
          if (startAbort.signal.aborted || callerSignal?.aborted) {
            this.state = 'idle';
            await this.tenVad.dispose().catch(() => undefined);
            throw createStreamingVadInitAbortedError();
          }
          await this.tenVad.reset();
          const backendName = this.getVadBackendName();
          this.recordDecision(
            `${backendName} ready for diagnostics`,
            this.tenVad.getStatus() as unknown as Record<string, unknown>,
          );
        } catch (error) {
          if (this.startCancelled || this.disposed) {
            return;
          }
          if (isStreamingVadInitAborted(error, startAbort.signal) || callerSignal?.aborted) {
            this.state = 'idle';
            await this.tenVad.dispose().catch(() => undefined);
            throw createStreamingVadInitAbortedError();
          }
          this.lastError = error instanceof Error ? error : new Error(String(error));
          const backendName = this.getVadBackendName();
          this.recordDecision(`${backendName} degraded; segmentation remains on rough gate`, {
            error: this.lastError.message,
          });
          this.emit({
            type: 'error',
            payload: this.lastError,
          });
        }
      }

      if (this.startCancelled || this.disposed) {
        return;
      }

      this.emit({
        type: 'metrics',
        payload: this.getSnapshot(),
      });
    } finally {
      cleanupCallerAbort();
      if (this.startAbort === startAbort) {
        this.startAbort = null;
      }
    }
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

    const effectiveGateMode = this.resolveEffectiveGateMode();
    const isHybrid = effectiveGateMode === 'hybrid';
    const isVadOnly = effectiveGateMode === STREAMING_GATE_MODES.VAD_ONLY;

    // Retained history must not replay a finalized (even rejected) utterance.
    const vadOnsetFrame = !this.activeSegment && (isHybrid || isVadOnly)
      ? this.tenVad?.findFirstSpeechFrame(
          Math.max(baseFrame, this.lastFinalizedLogicalEndFrame), nowFrame,
        ) ?? null
      : null;
    const vadOnsetSpeaking =
      (isHybrid || isVadOnly) &&
      vadOnsetFrame !== null &&
      this.tenVad !== null &&
      (this.tenVad.getStatus().speaking ||
        this.tenVad.hasRecentSpeech(
          nowFrame,
          this.config.tenVadConfirmationWindowMs,
          this.sampleRate,
        ));

    if (!this.activeSegment) {
      if (isVadOnly) {
        if (vadOnsetSpeaking && vadOnsetFrame !== null) {
          const firstSpeechFrame = vadOnsetFrame;
          this.activateSegment(firstSpeechFrame, nowFrame, 'ten-vad');
          this.recordDecision('Neural VAD speech start', {
            startFrame: firstSpeechFrame,
            gateMode: effectiveGateMode,
          });
        }
      } else if (rough.speechStart) {
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
      } else if (isHybrid && vadOnsetSpeaking && vadOnsetFrame !== null) {
        const firstSpeechFrame = vadOnsetFrame;
        this.activateSegment(firstSpeechFrame, nowFrame, 'ten-vad');
        this.recordDecision('Hybrid neural VAD speech start', {
          startFrame: firstSpeechFrame,
          gateMode: effectiveGateMode,
        });
      }
    }

    if (!this.activeSegment && !rough.speechStart && !vadOnsetSpeaking) {
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
      // Prefer a quiet chunk shortly before the hard limit. Decide prospectively:
      // already-delivered streaming audio must never be cut retroactively.
      const softWindowMs = Math.min(160, this.config.maxSegmentDurationMs * 0.1);
      let quietNearLimit = false;
      if (segmentDurationMs >= this.config.maxSegmentDurationMs - softWindowMs &&
          segmentDurationMs >= this.config.minSpeechDurationMs) {
        quietNearLimit = isQuietChunk(
          chunk,
          Math.max(rough.noiseFloor * 2, this.config.energyThreshold * 0.5),
        );
      }
      if (segmentDurationMs >= this.config.maxSegmentDurationMs || quietNearLimit) {
        if (quietNearLimit && segmentDurationMs < this.config.maxSegmentDurationMs) {
          this.recordDecision('Long speech split at a quiet chunk before the hard limit', {
            endFrame: nowFrame,
            segmentDurationMs,
          });
        }
        const finalized = this.finalizeSegment('max-duration', nowFrame);
        const speechStillActive =
          quietNearLimit || (effectiveGateMode === STREAMING_GATE_MODES.ROUGH_ONLY
            ? rough.isSpeech
            : rough.isSpeech ||
              (this.tenVad?.hasRecentSpeech(
                nowFrame,
                this.config.tenVadHangoverMs,
                this.sampleRate,
              ) ?? false));

        if (speechStillActive) {
          this.continuationAfterMaxDuration = true;
          const nextStartFrame = Math.max(baseFrame, rough.chunkStartFrame);
          this.activateSegment(
            nextStartFrame,
            nowFrame,
            isVadOnly ? 'ten-vad' : 'rough-fallback',
          );
          this.recordDecision('Long speech split into a new rough segment', {
            previousAccepted: finalized?.metadata.filter?.accepted ?? false,
            nextStartFrame,
          });
        }
      } else if (isVadOnly) {
        const vadSpeaking =
          this.tenVad?.getStatus().speaking ||
          (this.tenVad?.hasRecentSpeech(
            nowFrame,
            this.config.tenVadHangoverMs,
            this.sampleRate,
          ) ?? false);
        if (vadSpeaking) {
          this.state = 'speaking';
          this.emitSpeechUpdate(nowFrame);
        } else {
          const vadSilence =
            this.tenVad?.hasRecentSilence(
              nowFrame,
              this.config.minSilenceDurationMs,
              this.sampleRate,
            ) ?? true;
          if (vadSilence) {
            this.recordDecision('Neural VAD finalized speech after sustained silence', {
              endFrame: nowFrame,
            });
            this.finalizeSegment('silence', nowFrame);
          } else {
            this.state = 'candidate';
          }
        }
      } else if (isHybrid) {
        const vadSpeaking =
          this.tenVad?.getStatus().speaking ||
          (this.tenVad?.hasRecentSpeech(
            nowFrame,
            this.config.tenVadHangoverMs,
            this.sampleRate,
          ) ?? false);
        const isSpeaking = rough.isSpeech || vadSpeaking;

        if (isSpeaking) {
          this.state = 'speaking';
          this.emitSpeechUpdate(nowFrame);
        } else {
          const vadSilence =
            this.tenVad?.hasRecentSilence(
              nowFrame,
              this.config.minSilenceDurationMs,
              this.sampleRate,
            ) ?? true;
          if ((rough.speechEnd || !rough.isSpeech) && vadSilence) {
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
    return computeFinalizedSegmentBounds(
      {
        baseFrame: this.ringBuffer.getBaseFrameOffset(),
        currentFrame: this.ringBuffer.getCurrentFrame(),
        config: this.config,
        sampleRate: this.sampleRate,
        continuationAfterMaxDuration: this.continuationAfterMaxDuration,
        lastAcceptedLogicalEndFrame: this.lastAcceptedLogicalEndFrame,
      },
      logicalStartFrame,
      logicalEndFrame,
    );
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
    this.lastAcceptanceInfo = buildAcceptanceInfo(
      filterResult,
      reason,
      activeSegment.source,
      bounds,
      this.resolveEffectiveGateMode(),
      tenVadStatus.state,
    );

    this.emit({
      type: 'speech-end',
      payload: {
        startFrame: bounds.extractedStartFrame,
        endFrame: bounds.extractedEndFrame,
        sampleRate: this.sampleRate,
      },
    });

    this.activeSegment = null;
    this.lastFinalizedLogicalEndFrame = bounds.logicalEndFrame;
    this.pendingSegmentStartFrame = null;
    this.state = 'listening';

    if (reason !== 'max-duration') {
      this.continuationAfterMaxDuration = false;
    }

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
    this.startCancelled = true;
    this.startAbort?.abort();
    let segment: StreamingDetectorSegment | null = null;
    if (flush) {
      segment = this.finalizeSegment('stop');
    } else {
      this.activeSegment = null;
      this.pendingSegmentStartFrame = null;
      this.state = 'listening';
    }
    this.continuationAfterMaxDuration = false;
    this.state = 'idle';
    return segment;
  }

  updateConfig(
    partial: StreamingDetectorConfigOverrides & {
      readonly profileId?: string;
    } = {},
  ): void {
    const nextProfileId = partial.profileId ?? this.profileId;
    const nextOverrides = mergeConfigOverrides(this.getCurrentConfigOverrides(), partial);

    const previousTenVadEnabled = this.config.tenVadEnabled;
    const previousVadBackend = this.config.vadBackend;
    const previousVadMode = (this.config as unknown as Record<string, unknown>).vadMode;
    this.profileId = nextProfileId;
    this.config = mergeStreamingConfig(nextProfileId, nextOverrides);
    const nextVadMode = (partial as unknown as Record<string, unknown> | undefined)?.vadMode ?? previousVadMode;
    if (nextVadMode !== undefined) {
      (this.config as unknown as Record<string, unknown>).vadMode = nextVadMode;
    }
    this.roughGate.updateConfig(this.buildRoughGateConfig());

    if (!previousTenVadEnabled && this.config.tenVadEnabled) {
      const nextTenVad = this.createTenVad(this.buildTenVadConfig(), this.tenVadOptions);
      this.replaceTenVad(nextTenVad);
    } else if (
      previousTenVadEnabled
      && this.config.tenVadEnabled
      && previousVadBackend !== this.config.vadBackend
    ) {
      const previousTenVad = this.tenVad;
      const nextTenVad = this.createTenVad(this.buildTenVadConfig(), this.tenVadOptions);
      this.replaceTenVad(nextTenVad);
      void previousTenVad?.dispose().catch(() => undefined);
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
        vadBackend: this.config.vadBackend,
        tenVadReady: this.isTenVadReady(),
      },
      rough: buildRoughSnapshot(
        this.lastMetrics,
        this.config,
        this.roughGate,
        waveform,
        waveformPointCount,
      ),
      foreground: buildForegroundSnapshot(
        this.config,
        this.lastMetrics,
        this.lastForegroundResult,
      ),
      tenVad: this.getTenVadStatus(),
      warnings: this.buildWarnings(),
      error: this.lastError?.message ?? null,
    };
  }

  private buildWarnings(): string[] {
    return buildDetectorWarnings(
      this.config,
      this.tenVad?.getStatus(),
      this.resolveEffectiveGateMode(),
    );
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.startCancelled = true;
    this.startAbort?.abort();
    this.tenVadUnsubscribe?.();
    this.tenVadUnsubscribe = null;
    await this.tenVad?.dispose();
    this.listeners.clear();
  }
}
