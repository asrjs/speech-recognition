import type {
  FireRedAedConfig,
  FireRedStreamVadConfig,
  FireRedVadConfig,
  NormalizedFireRedAedConfig,
  NormalizedFireRedStreamVadConfig,
  NormalizedFireRedVadConfig,
} from '../types.js';
import { ConfigAliasConflictError } from './errors.js';

function resolveAliasedValue<T>(
  config: Record<string, unknown>,
  snakeKey: string,
  camelKey: string,
  fallback: T,
): T {
  const snake = config[snakeKey] as T | undefined;
  const camel = config[camelKey] as T | undefined;
  if (snake !== undefined && camel !== undefined && snake !== camel) {
    throw new ConfigAliasConflictError(
      `Conflicting values for "${snakeKey}" and "${camelKey}". Provide only one or set both to the same value.`,
    );
  }
  return (snake ?? camel ?? fallback) as T;
}

function toBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

function toNumber(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback;
  }
  return value;
}

function normalizeRuntimeShared(config: Record<string, unknown>): Record<string, unknown> {
  const clone: Record<string, unknown> = { ...config };
  const useGpu = resolveAliasedValue(clone, 'use_gpu', 'useGpu', false);
  clone.use_gpu = toBoolean(useGpu, false);
  return clone;
}

export function normalizeVadConfig(input: FireRedVadConfig = {}): NormalizedFireRedVadConfig {
  const config = normalizeRuntimeShared(input as Record<string, unknown>);
  const speechThreshold = toNumber(
    resolveAliasedValue(config, 'speech_threshold', 'speechThreshold', 0.4),
    0.4,
  );
  if (speechThreshold < 0 || speechThreshold > 1) {
    throw new RangeError('speech_threshold must be in [0, 1].');
  }

  const minSpeechFrame = toNumber(
    resolveAliasedValue(config, 'min_speech_frame', 'minSpeechFrame', 20),
    20,
  );
  if (minSpeechFrame <= 0) {
    throw new RangeError('min_speech_frame must be positive.');
  }

  return {
    ...input,
    use_gpu: toBoolean(config.use_gpu, false),
    smooth_window_size: toNumber(
      resolveAliasedValue(config, 'smooth_window_size', 'smoothWindowSize', 5),
      5,
    ),
    speech_threshold: speechThreshold,
    min_speech_frame: minSpeechFrame,
    max_speech_frame: toNumber(
      resolveAliasedValue(config, 'max_speech_frame', 'maxSpeechFrame', 2000),
      2000,
    ),
    min_silence_frame: toNumber(
      resolveAliasedValue(config, 'min_silence_frame', 'minSilenceFrame', 20),
      20,
    ),
    merge_silence_frame: toNumber(
      resolveAliasedValue(config, 'merge_silence_frame', 'mergeSilenceFrame', 0),
      0,
    ),
    extend_speech_frame: toNumber(
      resolveAliasedValue(config, 'extend_speech_frame', 'extendSpeechFrame', 0),
      0,
    ),
    chunk_max_frame: toNumber(
      resolveAliasedValue(config, 'chunk_max_frame', 'chunkMaxFrame', 30000),
      30000,
    ),
  };
}

export function normalizeStreamVadConfig(
  input: FireRedStreamVadConfig = {},
): NormalizedFireRedStreamVadConfig {
  const config = normalizeRuntimeShared(input as Record<string, unknown>);
  const speechThreshold = toNumber(
    resolveAliasedValue(config, 'speech_threshold', 'speechThreshold', 0.4),
    0.4,
  );
  if (speechThreshold < 0 || speechThreshold > 1) {
    throw new RangeError('speech_threshold must be in [0, 1].');
  }
  const minSpeechFrame = toNumber(
    resolveAliasedValue(config, 'min_speech_frame', 'minSpeechFrame', 8),
    8,
  );
  if (minSpeechFrame <= 0) {
    throw new RangeError('min_speech_frame must be positive.');
  }
  return {
    ...input,
    use_gpu: toBoolean(config.use_gpu, false),
    smooth_window_size: toNumber(
      resolveAliasedValue(config, 'smooth_window_size', 'smoothWindowSize', 5),
      5,
    ),
    speech_threshold: speechThreshold,
    pad_start_frame: toNumber(
      resolveAliasedValue(config, 'pad_start_frame', 'padStartFrame', 5),
      5,
    ),
    min_speech_frame: minSpeechFrame,
    max_speech_frame: toNumber(
      resolveAliasedValue(config, 'max_speech_frame', 'maxSpeechFrame', 2000),
      2000,
    ),
    min_silence_frame: toNumber(
      resolveAliasedValue(config, 'min_silence_frame', 'minSilenceFrame', 20),
      20,
    ),
    chunk_max_frame: toNumber(
      resolveAliasedValue(config, 'chunk_max_frame', 'chunkMaxFrame', 30000),
      30000,
    ),
  };
}

export function normalizeAedConfig(input: FireRedAedConfig = {}): NormalizedFireRedAedConfig {
  const config = normalizeRuntimeShared(input as Record<string, unknown>);
  return {
    ...input,
    use_gpu: toBoolean(config.use_gpu, false),
    smooth_window_size: toNumber(
      resolveAliasedValue(config, 'smooth_window_size', 'smoothWindowSize', 5),
      5,
    ),
    speech_threshold: toNumber(
      resolveAliasedValue(config, 'speech_threshold', 'speechThreshold', 0.4),
      0.4,
    ),
    singing_threshold: toNumber(
      resolveAliasedValue(config, 'singing_threshold', 'singingThreshold', 0.5),
      0.5,
    ),
    music_threshold: toNumber(
      resolveAliasedValue(config, 'music_threshold', 'musicThreshold', 0.5),
      0.5,
    ),
    min_event_frame: toNumber(
      resolveAliasedValue(config, 'min_event_frame', 'minEventFrame', 20),
      20,
    ),
    max_event_frame: toNumber(
      resolveAliasedValue(config, 'max_event_frame', 'maxEventFrame', 2000),
      2000,
    ),
    min_silence_frame: toNumber(
      resolveAliasedValue(config, 'min_silence_frame', 'minSilenceFrame', 20),
      20,
    ),
    merge_silence_frame: toNumber(
      resolveAliasedValue(config, 'merge_silence_frame', 'mergeSilenceFrame', 0),
      0,
    ),
    extend_speech_frame: toNumber(
      resolveAliasedValue(config, 'extend_speech_frame', 'extendSpeechFrame', 0),
      0,
    ),
    chunk_max_frame: toNumber(
      resolveAliasedValue(config, 'chunk_max_frame', 'chunkMaxFrame', 30000),
      30000,
    ),
  };
}
