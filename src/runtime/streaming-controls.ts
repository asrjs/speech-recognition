import { STREAMING_TIMELINE_CHUNK_MS } from './audio-timeline.js';
import type { StreamingDetectorConfig } from './streaming-config.js';

/**
 * Consumer-facing metadata for a tunable realtime detector setting.
 * UI packages can use this to render sliders, tooltips, docs, or validation.
 */
export interface StreamingControlDefinition {
  readonly field: keyof StreamingDetectorConfig;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly description: string;
  readonly guide: string;
  readonly chunkAligned?: boolean;
  readonly stepChunkMultiplier?: number;
  readonly maxFromConfigField?: keyof StreamingDetectorConfig;
}

function isMillisecondsField(field: keyof StreamingDetectorConfig): boolean {
  return String(field).endsWith('Ms');
}

function isChunkField(field: keyof StreamingDetectorConfig): boolean {
  return String(field).endsWith('Chunks');
}

/**
 * Shared tuning-control metadata for realtime detector settings.
 * The core library remains headless, but it serves canonical ranges,
 * step sizes, and descriptions so framework bindings can stay consistent.
 */
export const STREAMING_CONTROL_DEFINITIONS: readonly StreamingControlDefinition[] = [
  {
    field: 'analysisWindowMs',
    label: 'analysis',
    min: STREAMING_TIMELINE_CHUNK_MS,
    max: 160,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Rough-gate analysis window. Larger windows are steadier but slower to react.',
    guide: 'Lower for faster onset and release. Raise for noisier microphones.',
    chunkAligned: true,
  },
  {
    field: 'energySmoothingDurationMs',
    label: 'energy_smoothing',
    min: 80,
    max: 1600,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Release and hold smoothing for the rough energy gate.',
    guide: 'Lower to reduce sticky tails. Raise to suppress jitter.',
    chunkAligned: true,
  },
  {
    field: 'prerollMs',
    label: 'preroll',
    min: 80,
    max: 1200,
    step: 20,
    description: 'Audio kept before accepted onset so initial consonants are not cut.',
    guide: 'Raise if starts are clipped. Too high adds extra leading silence.',
    chunkAligned: true,
  },
  {
    field: 'minSilenceDurationMs',
    label: 'silence',
    min: 0,
    max: 2000,
    step: 20,
    description: 'How long rough silence must last before the segment can end.',
    guide: 'Lower for shorter tails. Raise to avoid choppy segmentation.',
    chunkAligned: true,
  },
  {
    field: 'minSpeechDurationMs',
    label: 'min_speech',
    min: 80,
    max: 640,
    step: 40,
    description: 'Minimum rough-gate speech duration before onset is accepted.',
    guide: 'Lower for short words. Raise to reject clicks and bumps.',
    chunkAligned: true,
  },
  {
    field: 'minSpeechLevelDbfs',
    label: 'speech_dbfs',
    min: -72,
    max: -12,
    step: 1,
    description: 'RMS energy threshold for the rough gate, in dBFS.',
    guide: 'More negative is more sensitive. Less negative rejects quieter speech and noise.',
  },
  {
    field: 'levelWindowMs',
    label: 'level_window',
    min: 200,
    max: 2000,
    step: STREAMING_TIMELINE_CHUNK_MS * 4,
    description: 'Longer RMS level window used for diagnostics and noise-relative metrics.',
    guide: 'Raise for steadier meters. Lower for faster visual response.',
    chunkAligned: true,
    stepChunkMultiplier: 4,
    maxFromConfigField: 'ringBufferDurationMs',
  },
  {
    field: 'energyRiseThreshold',
    label: 'energy_rise',
    min: 0.01,
    max: 0.4,
    step: 0.01,
    description: 'Extra onset cue from sudden energy growth.',
    guide: 'Raise to ignore small fluctuations. Lower to catch softer onsets.',
  },
  {
    field: 'maxOnsetLookbackChunks',
    label: 'lookback',
    min: 1,
    max: 12,
    step: 1,
    description: 'Maximum onset rewind in chunk units.',
    guide: 'Each chunk is the configured frame size. Raise if starts are still clipped.',
  },
  {
    field: 'tenVadThreshold',
    label: 'ten_threshold',
    min: 0.05,
    max: 0.95,
    step: 0.01,
    description: 'TEN-VAD probability threshold for speech versus non-speech.',
    guide: 'Lower is more permissive. Higher is stricter.',
  },
  {
    field: 'tenVadConfirmationWindowMs',
    label: 'ten_confirm',
    min: 64,
    max: 800,
    step: 16,
    description: 'Window used to confirm recent TEN-VAD speech or silence.',
    guide: 'Lower for faster turns. Raise for more stable confirmation.',
    chunkAligned: true,
  },
  {
    field: 'tenVadHangoverMs',
    label: 'ten_hangover',
    min: 64,
    max: 1200,
    step: 16,
    description: 'How long recent TEN-VAD speech can keep a tail alive.',
    guide: 'Lower to shorten tails. Raise to keep phrase endings intact.',
    chunkAligned: true,
  },
  {
    field: 'tenVadMinSpeechDurationMs',
    label: 'ten_min_speech',
    min: 48,
    max: 640,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Minimum continuous TEN-VAD speech needed before speech is confirmed.',
    guide: 'Lower for quicker starts. Raise to reject burst noise.',
    chunkAligned: true,
  },
  {
    field: 'tenVadMinSilenceDurationMs',
    label: 'ten_min_silence',
    min: 32,
    max: 640,
    step: STREAMING_TIMELINE_CHUNK_MS,
    description: 'Minimum continuous TEN-VAD silence needed before speech can end.',
    guide: 'Lower for quicker release. Raise to avoid cutting internal pauses.',
    chunkAligned: true,
  },
  {
    field: 'maxSegmentDurationMs',
    label: 'max_segment',
    min: 400,
    max: 12000,
    step: 100,
    description: 'Hard cap for one live segment before forced finalization.',
    guide: 'Raise for long utterances. Lower for faster transcript turnover.',
  },
] as const;

export interface StreamingControlConstraints {
  readonly min: number;
  readonly max: number;
  readonly step: number;
}

/**
 * Return the canonical list of realtime tuning controls exposed by the library.
 *
 * @example
 * ```ts
 * import { listStreamingControls } from '@asrjs/speech-recognition/realtime';
 *
 * const controls = listStreamingControls();
 * const sliderFields = controls.map((control) => control.field);
 * ```
 */
export function listStreamingControls(): readonly StreamingControlDefinition[] {
  return STREAMING_CONTROL_DEFINITIONS;
}

/**
 * Look up one tuning control definition by config field.
 *
 * @example
 * ```ts
 * import { getStreamingControlDefinition } from '@asrjs/speech-recognition/realtime';
 *
 * const silenceControl = getStreamingControlDefinition('minSilenceDurationMs');
 * if (silenceControl) {
 *   console.log(silenceControl.label); // "silence"
 * }
 * ```
 */
export function getStreamingControlDefinition(
  field: keyof StreamingDetectorConfig,
): StreamingControlDefinition | undefined {
  return STREAMING_CONTROL_DEFINITIONS.find((definition) => definition.field === field);
}

/**
 * Resolve the effective min/max/step for a control against a concrete runtime config.
 * This accounts for dynamic constraints such as chunk-aligned step sizes and
 * max values derived from other config fields.
 *
 * @example
 * ```ts
 * import {
 *   getStreamingControlDefinition,
 *   resolveStreamingControlConstraints,
 * } from '@asrjs/speech-recognition/realtime';
 *
 * const definition = getStreamingControlDefinition('levelWindowMs');
 * const constraints = resolveStreamingControlConstraints(definition!, {
 *   chunkDurationMs: 32,
 *   ringBufferDurationMs: 12000,
 * });
 *
 * console.log(constraints);
 * // { min: 200, max: 12000, step: 128 }
 * ```
 */
export function resolveStreamingControlConstraints(
  definition: StreamingControlDefinition,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): StreamingControlConstraints {
  const chunkDurationMs = Math.max(
    1,
    Math.round(resolvedConfig?.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS),
  );
  const step = definition.chunkAligned
    ? Math.max(1, chunkDurationMs * Math.max(1, definition.stepChunkMultiplier ?? 1))
    : definition.step;
  const maxFromConfig =
    definition.maxFromConfigField &&
    typeof resolvedConfig?.[definition.maxFromConfigField] === 'number' &&
    Number.isFinite(resolvedConfig[definition.maxFromConfigField])
      ? Number(resolvedConfig[definition.maxFromConfigField])
      : null;
  const max = Math.max(definition.min, maxFromConfig ?? definition.max);
  return {
    min: definition.min,
    max,
    step,
  };
}

/**
 * Resolve the effective step for a tuning control against a concrete runtime config.
 * Time-based controls follow the active chunk size so framework bindings stay aligned.
 *
 * @example
 * ```ts
 * import {
 *   getStreamingControlDefinition,
 *   resolveStreamingControlStep,
 * } from '@asrjs/speech-recognition/realtime';
 *
 * const definition = getStreamingControlDefinition('tenVadMinSpeechDurationMs');
 * const step = resolveStreamingControlStep(definition!, { chunkDurationMs: 16 });
 *
 * console.log(step); // 16
 * ```
 */
export function resolveStreamingControlStep(
  definition: StreamingControlDefinition,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): number {
  return resolveStreamingControlConstraints(definition, resolvedConfig).step;
}

/**
 * Clamp an arbitrary input value to the effective runtime range of a control.
 *
 * @example
 * ```ts
 * import {
 *   clampStreamingControlValue,
 *   getStreamingControlDefinition,
 * } from '@asrjs/speech-recognition/realtime';
 *
 * const definition = getStreamingControlDefinition('tenVadThreshold');
 * const clamped = clampStreamingControlValue(definition!, 1.2);
 *
 * console.log(clamped); // 0.95
 * ```
 */
export function clampStreamingControlValue(
  definition: StreamingControlDefinition,
  value: number,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): number {
  if (!Number.isFinite(value)) {
    return resolveStreamingControlConstraints(definition, resolvedConfig).min;
  }
  const { min, max } = resolveStreamingControlConstraints(definition, resolvedConfig);
  return Math.min(max, Math.max(min, value));
}

/**
 * Snap an arbitrary input value to the effective runtime step of a control after clamping.
 * This is useful for framework bindings that accept free-form text input or custom sliders.
 *
 * @example
 * ```ts
 * import {
 *   getStreamingControlDefinition,
 *   normalizeStreamingControlValue,
 * } from '@asrjs/speech-recognition/realtime';
 *
 * const definition = getStreamingControlDefinition('minSilenceDurationMs');
 * const normalized = normalizeStreamingControlValue(definition!, 23, {
 *   chunkDurationMs: 16,
 * });
 *
 * console.log(normalized); // 16
 * ```
 */
export function normalizeStreamingControlValue(
  definition: StreamingControlDefinition,
  value: number,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): number {
  const { min, step } = resolveStreamingControlConstraints(definition, resolvedConfig);
  const clamped = clampStreamingControlValue(definition, value, resolvedConfig);
  if (!Number.isFinite(step) || step <= 0) {
    return clamped;
  }
  const snapped = min + Math.round((clamped - min) / step) * step;
  return clampStreamingControlValue(definition, snapped, resolvedConfig);
}

/**
 * Format a control value using the canonical unit semantics of the setting.
 *
 * @example
 * ```ts
 * import {
 *   formatStreamingControlValue,
 *   getStreamingControlDefinition,
 * } from '@asrjs/speech-recognition/realtime';
 *
 * const definition = getStreamingControlDefinition('minSpeechLevelDbfs');
 * const label = formatStreamingControlValue(definition!, -47);
 *
 * console.log(label); // "-47.0 dBFS"
 * ```
 */
export function formatStreamingControlValue(
  definition: StreamingControlDefinition,
  value: number | null | undefined,
): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '--';
  }
  if (definition.field === 'minSpeechLevelDbfs') {
    return `${value.toFixed(1)} dBFS`;
  }
  if (
    definition.field === 'tenVadThreshold' ||
    definition.field === 'energyRiseThreshold'
  ) {
    return value.toFixed(2);
  }
  if (isMillisecondsField(definition.field)) {
    return `${value.toFixed(0)} ms`;
  }
  if (isChunkField(definition.field)) {
    return `${value.toFixed(0)} chunks`;
  }
  return value.toFixed(2);
}

/**
 * Format a consumer-facing hint describing range, resolution, and chunk alignment.
 *
 * @example
 * ```ts
 * import {
 *   formatStreamingControlHint,
 *   getStreamingControlDefinition,
 * } from '@asrjs/speech-recognition/realtime';
 *
 * const definition = getStreamingControlDefinition('levelWindowMs');
 * const hint = formatStreamingControlHint(definition!, {
 *   chunkDurationMs: 16,
 *   ringBufferDurationMs: 12000,
 * });
 *
 * console.log(hint); // "200..12000 ms · step 64 ms · chunk-aligned 16 ms"
 * ```
 */
export function formatStreamingControlHint(
  definition: StreamingControlDefinition,
  resolvedConfig?: Partial<StreamingDetectorConfig> | null,
): string {
  const { min, max, step } = resolveStreamingControlConstraints(definition, resolvedConfig);
  const chunkDurationMs = Math.round(
    resolvedConfig?.chunkDurationMs ?? STREAMING_TIMELINE_CHUNK_MS,
  );
  const chunkNote = definition.chunkAligned
    ? `chunk-aligned ${chunkDurationMs} ms`
    : null;

  const range =
    definition.field === 'minSpeechLevelDbfs'
      ? `${min}..${max} dBFS`
      : definition.field === 'tenVadThreshold' || definition.field === 'energyRiseThreshold'
        ? `${min}..${max}`
        : isMillisecondsField(definition.field)
          ? `${min}..${max} ms`
          : isChunkField(definition.field)
            ? `${min}..${max} chunks`
            : `${min}..${max}`;

  const stepLabel =
    definition.field === 'minSpeechLevelDbfs'
      ? `step ${step} dB`
      : isMillisecondsField(definition.field)
        ? `step ${step} ms`
        : isChunkField(definition.field)
          ? `step ${step} chunk${step === 1 ? '' : 's'}`
          : `step ${step}`;

  return chunkNote ? `${range} · ${stepLabel} · ${chunkNote}` : `${range} · ${stepLabel}`;
}
