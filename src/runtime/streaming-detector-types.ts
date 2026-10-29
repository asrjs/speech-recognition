import type { AudioRingBuffer } from './realtime.js';
import type {
  RoughSpeechChunkSummary,
  RoughSpeechGateWindowResult,
  RoughSpeechTimelinePoint,
} from './rough-speech-gate.js';
import type { SegmentForegroundFilterResult } from './segment-foreground-filter.js';
import type {
  StreamingDetectorConfig,
  StreamingDetectorConfigOverrides,
  StreamingGateMode,
  StreamingVadBackend,
} from './streaming-config.js';

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
  init(signal?: { readonly aborted: boolean } | null): Promise<void>;
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
    readonly vadBackend: StreamingVadBackend;
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

export interface StreamingSpeechDetectorOptions {
  readonly profileId?: string;
  readonly config?: StreamingDetectorConfigOverrides;
  readonly isRealtimeEouModel?: boolean;
  readonly tenVadFactory?: (
    config: Record<string, unknown>,
    options?: unknown,
  ) => StreamingTenVadLike;
  readonly tenVadOptions?: unknown;
  readonly signal?: { readonly aborted: boolean } | null;
}
