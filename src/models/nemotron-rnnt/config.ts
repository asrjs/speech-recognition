import type { ModelClassification } from '../../types/index.js';
import { describeModelClassification } from '../nemo-common/index.js';
import type { NemotronRnntEncoderWindowing, NemotronRnntModelConfig } from './types.js';

export const NEMOTRON_BUNDLE_CONFIG_FORMAT = 'asrjs.nemotron-rnnt';
export const NEMOTRON_BUNDLE_CONFIG_SCHEMA_VERSION = 1;

export interface NemotronRnntEncoderWindow {
  readonly startFrame: number;
  readonly inputFrames: number;
  readonly advanceFrames: number;
  readonly first: boolean;
}

export const DEFAULT_NEMOTRON_RNNT_CLASSIFICATION: ModelClassification = {
  ecosystem: 'nemo',
  processor: 'nemo-mel',
  encoder: 'fastconformer',
  decoder: 'rnnt',
  topology: 'rnnt',
  family: 'nemotron',
  task: 'asr',
};

const DEFAULT_NEMOTRON_RNNT_PROMPT_IDS = {
  auto: 101,
  en: 0,
  tr: 18,
};

const DEFAULT_NEMOTRON_RNNT_ENCODER_CACHE = {
  channelLayers: 24,
  channelFrames: 56,
  channelDim: 1024,
  timeLayers: 24,
  timeFrames: 8,
  timeDim: 1024,
};

const BASE_NEMOTRON_RNNT_CONFIG: NemotronRnntModelConfig = {
  ecosystem: 'nemo',
  architecture: 'nemotron-rnnt',
  encoderArchitecture: 'fastconformer',
  decoderArchitecture: 'rnnt',
  sampleRate: 16000,
  frameShiftSeconds: 0.01,
  subsamplingFactor: 8,
  melBins: 128,
  preprocessorValidLengthMode: 'onnx',
  preprocessorNormalization: 'none',
  predictionHiddenSize: 640,
  predictionLayers: 2,
  chunkFrames: 65,
  encoderOutputFramesPerChunk: 7,
  encoderCache: DEFAULT_NEMOTRON_RNNT_ENCODER_CACHE,
  promptIds: DEFAULT_NEMOTRON_RNNT_PROMPT_IDS,
  defaultPromptId: DEFAULT_NEMOTRON_RNNT_PROMPT_IDS.auto,
  jointWindowFrames: 16,
  maxDecodeSteps: 200,
  maxOutputTokens: 200,
  blankTokenId: 13087,
  vocabularySize: 13088,
  languages: ['en', 'tr', 'auto'],
  tokenizer: {
    kind: 'bpe',
    blankTokenId: 13087,
    unkTokenId: 13087,
  },
};

export function parseNemotronRnntConfig(
  _modelId: string,
  override: Partial<NemotronRnntModelConfig> = {},
): NemotronRnntModelConfig {
  const config: NemotronRnntModelConfig = {
    ...BASE_NEMOTRON_RNNT_CONFIG,
    ...override,
    encoderCache: {
      ...BASE_NEMOTRON_RNNT_CONFIG.encoderCache,
      ...(override.encoderCache ?? {}),
    },
    promptIds: {
      ...BASE_NEMOTRON_RNNT_CONFIG.promptIds,
      ...(override.promptIds ?? {}),
    },
    tokenizer: {
      ...BASE_NEMOTRON_RNNT_CONFIG.tokenizer,
      ...(override.tokenizer ?? {}),
    },
    encoderWindowing:
      override.encoderWindowing !== undefined
        ? validateEncoderWindowing(
            override.encoderWindowing,
            override.chunkFrames ?? BASE_NEMOTRON_RNNT_CONFIG.chunkFrames,
          )
        : undefined,
  };
  return config;
}

/**
 * Read the small, versioned metadata contract shipped with compatible ONNX
 * bundles. Unrelated config.json files are ignored so legacy bundles remain
 * on their established static-window path.
 */
export function parseNemotronRnntBundleConfig(
  value: unknown,
): Pick<NemotronRnntModelConfig, 'encoderWindowing'> | undefined {
  if (!isRecord(value) || value.format !== NEMOTRON_BUNDLE_CONFIG_FORMAT) {
    return undefined;
  }
  if (value.schemaVersion !== NEMOTRON_BUNDLE_CONFIG_SCHEMA_VERSION) {
    throw new Error(`Unsupported Nemotron bundle schema version: ${String(value.schemaVersion)}.`);
  }
  if (value.family !== 'nemotron-rnnt') {
    throw new Error('Nemotron bundle config family must be "nemotron-rnnt".');
  }
  if (value.encoderWindowing === undefined) {
    return { encoderWindowing: undefined };
  }
  if (!isRecord(value.encoderWindowing)) {
    throw new Error('Nemotron encoderWindowing metadata must be an object.');
  }

  const encoderWindowing = validateEncoderWindowing(
    value.encoderWindowing,
    BASE_NEMOTRON_RNNT_CONFIG.chunkFrames,
  );
  return { encoderWindowing };
}

export function planNemotronRnntEncoderWindows(
  stableFrameStart: number,
  stableFrameEnd: number,
  windowing: NemotronRnntEncoderWindowing,
): NemotronRnntEncoderWindow[] {
  validateEncoderWindowing(windowing, windowing.shiftFrames + windowing.leftContextFrames);
  if (
    !isNonNegativeInteger(stableFrameStart) ||
    !isNonNegativeInteger(stableFrameEnd) ||
    stableFrameEnd < stableFrameStart
  ) {
    throw new Error('Nemotron stable frame range must be non-negative and ordered.');
  }
  if (stableFrameStart === stableFrameEnd) return [];
  if (
    stableFrameStart !== 0 &&
    (stableFrameStart < windowing.firstChunkFrames ||
      (stableFrameStart - windowing.firstChunkFrames) % windowing.shiftFrames !== 0)
  ) {
    throw new Error('Nemotron stable frame start is not aligned to the declared window shift.');
  }

  const windows: NemotronRnntEncoderWindow[] = [];
  let cursor = stableFrameStart;
  if (cursor === 0) {
    const advanceFrames = Math.min(windowing.firstChunkFrames, stableFrameEnd);
    windows.push({
      startFrame: 0,
      inputFrames: advanceFrames,
      advanceFrames,
      first: true,
    });
    cursor = windowing.firstChunkFrames;
  }

  while (cursor < stableFrameEnd) {
    const advanceFrames = Math.min(windowing.shiftFrames, stableFrameEnd - cursor);
    windows.push({
      startFrame: cursor - windowing.leftContextFrames,
      inputFrames: windowing.leftContextFrames + advanceFrames,
      advanceFrames,
      first: false,
    });
    cursor += advanceFrames;
  }
  return windows;
}

export function getNemotronRnntStableFrameEnd(
  availableFrames: number,
  windowing: NemotronRnntEncoderWindowing,
): number {
  validateEncoderWindowing(windowing, windowing.shiftFrames + windowing.leftContextFrames);
  if (!isNonNegativeInteger(availableFrames)) {
    throw new Error('Nemotron available frame count must be a non-negative integer.');
  }
  if (availableFrames < windowing.firstChunkFrames) return 0;
  return (
    windowing.firstChunkFrames +
    Math.floor((availableFrames - windowing.firstChunkFrames) / windowing.shiftFrames) *
      windowing.shiftFrames
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function validateEncoderWindowing(
  value: unknown,
  chunkFrames: number,
): NemotronRnntEncoderWindowing {
  if (!isRecord(value)) {
    throw new Error('Nemotron encoderWindowing metadata must be an object.');
  }
  const { firstChunkFrames, shiftFrames, leftContextFrames } = value;
  if (
    !isPositiveInteger(firstChunkFrames) ||
    !isPositiveInteger(shiftFrames) ||
    !isNonNegativeInteger(leftContextFrames)
  ) {
    throw new Error(
      'Nemotron encoderWindowing values must use positive integer firstChunkFrames and shiftFrames, and a non-negative integer leftContextFrames.',
    );
  }
  if (leftContextFrames > firstChunkFrames) {
    throw new Error('Nemotron leftContextFrames cannot exceed firstChunkFrames.');
  }
  if (firstChunkFrames > chunkFrames || shiftFrames + leftContextFrames !== chunkFrames) {
    throw new Error(
      `Nemotron encoderWindowing geometry must fit chunkFrames (${chunkFrames}): firstChunkFrames must not exceed it and shiftFrames + leftContextFrames must equal it.`,
    );
  }
  return { firstChunkFrames, shiftFrames, leftContextFrames };
}

export function describeNemotronRnntModel(
  modelId: string,
  classification: ModelClassification,
  config: NemotronRnntModelConfig,
): string {
  const label = describeModelClassification(classification);
  return `Nemotron 3.5 ASR Streaming model for ${modelId} (${label}, ${config.melBins} mel bins, vocab ${config.vocabularySize ?? '?'}).`;
}
