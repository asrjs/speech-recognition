import type {
  BaseTranscriptionOptions,
  ModelInferenceLimits,
  SegmentationStrategy,
  WindowMergeStrategy,
} from '../types/index.js';

export interface ResolvedWindowPolicy {
  readonly windowing: 'auto' | 'disabled' | 'force';
  readonly sampleRate: number;
  readonly maxInputDurationSec?: number;
  readonly windowDurationSec: number;
  readonly minWindowDurationSec?: number;
  readonly maxWindowDurationSec?: number;
  readonly autoWindowThresholdSec: number;
  readonly overlapSec: number;
  readonly strideSec?: number;
  readonly segmentationStrategy: SegmentationStrategy;
  readonly mergeStrategy: WindowMergeStrategy;
  readonly preferSentenceBoundaryWindowing: boolean;
  readonly preferVadSegmentWindowing: boolean;
}

export interface ResolveWindowPolicyOptions extends BaseTranscriptionOptions {
  readonly inference?: ModelInferenceLimits;
}

const DEFAULT_LIMITS: ModelInferenceLimits = {
  sampleRate: 16000,
  maxInputDurationSec: 60,
  recommendedWindowDurationSec: 30,
  minWindowDurationSec: 5,
  maxWindowDurationSec: 60,
  autoWindowThresholdSec: 60,
  defaultOverlapSec: 5,
  supportsWordTimestamps: false,
  supportsSegmentTimestamps: false,
  defaultSegmentationStrategy: 'none',
  defaultMergeStrategy: 'concat',
};

function finitePositive(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value) && value > 0;
}

function clamp(value: number, min: number | undefined, max: number | undefined): number {
  let next = value;
  if (finitePositive(min)) {
    next = Math.max(min, next);
  }
  if (finitePositive(max)) {
    next = Math.min(max, next);
  }
  return next;
}

export function resolveWindowPolicy(options: ResolveWindowPolicyOptions = {}): ResolvedWindowPolicy {
  const limits = options.inference ?? DEFAULT_LIMITS;
  const unsafe = options.unsafeAllowOverMaxWindow === true;
  const minWindowDurationSec = limits.minWindowDurationSec;
  const maxWindowDurationSec = options.maxInputDurationSeconds ?? limits.maxWindowDurationSec;
  const requestedWindow = options.windowDurationSeconds ?? options.chunkLengthSeconds;
  const baseWindowDurationSec =
    requestedWindow ?? limits.recommendedWindowDurationSec ?? limits.maxInputDurationSec ?? 30;
  const windowDurationSec = unsafe
    ? Math.max(minWindowDurationSec ?? 0.001, baseWindowDurationSec)
    : clamp(baseWindowDurationSec, minWindowDurationSec, maxWindowDurationSec);
  const overlapSec = Math.max(
    0,
    Math.min(
      windowDurationSec - 0.001,
      options.overlapSeconds ?? limits.defaultOverlapSec ?? options.strideLengthSeconds ?? 0,
    ),
  );

  return {
    windowing: options.windowing ?? 'auto',
    sampleRate: limits.sampleRate,
    maxInputDurationSec: options.maxInputDurationSeconds ?? limits.maxInputDurationSec,
    windowDurationSec,
    minWindowDurationSec,
    maxWindowDurationSec,
    autoWindowThresholdSec:
      limits.autoWindowThresholdSec ?? limits.maxInputDurationSec ?? windowDurationSec,
    overlapSec,
    strideSec: options.strideLengthSeconds ?? limits.defaultStrideSec,
    segmentationStrategy: options.segmentationStrategy ?? limits.defaultSegmentationStrategy,
    mergeStrategy: options.mergeStrategy ?? limits.defaultMergeStrategy,
    preferSentenceBoundaryWindowing: limits.preferSentenceBoundaryWindowing ?? false,
    preferVadSegmentWindowing: limits.preferVadSegmentWindowing ?? false,
  };
}

export function createDefaultModelInferenceLimits(input: {
  readonly family?: string;
  readonly modelId?: string;
}): ModelInferenceLimits {
  const family = String(input.family ?? '').toLowerCase();
  const modelId = String(input.modelId ?? '').toLowerCase();
  if (family.includes('whisper') || modelId.includes('whisper')) {
    return {
      sampleRate: 16000,
      maxInputDurationSec: 30,
      recommendedWindowDurationSec: 30,
      minWindowDurationSec: 5,
      maxWindowDurationSec: 30,
      autoWindowThresholdSec: 30,
      defaultStrideSec: 5,
      supportsWordTimestamps: true,
      supportsTokenTimestamps: true,
      supportsSegmentTimestamps: true,
      supportsConfidence: true,
      defaultSegmentationStrategy: 'whisper-token',
      defaultMergeStrategy: 'whisper-stride',
    };
  }
  if (family.includes('nemo-tdt') || modelId.includes('tdt') || modelId.includes('parakeet')) {
    return {
      sampleRate: 16000,
      maxInputDurationSec: 180,
      recommendedWindowDurationSec: 90,
      minWindowDurationSec: 20,
      maxWindowDurationSec: 180,
      autoWindowThresholdSec: 180,
      defaultOverlapSec: 10,
      preferSentenceBoundaryWindowing: true,
      supportsWordTimestamps: true,
      supportsTokenTimestamps: true,
      supportsSegmentTimestamps: true,
      supportsConfidence: true,
      defaultSegmentationStrategy: 'word-punctuation',
      defaultMergeStrategy: 'word-dedupe',
    };
  }
  return DEFAULT_LIMITS;
}

export function shouldUseWindowing(audioDurationSec: number, policy: ResolvedWindowPolicy): boolean {
  if (policy.windowing === 'disabled') {
    return false;
  }
  if (policy.windowing === 'force') {
    return audioDurationSec > policy.windowDurationSec;
  }
  return audioDurationSec > policy.autoWindowThresholdSec;
}
