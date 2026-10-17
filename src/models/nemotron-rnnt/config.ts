import type { ModelClassification } from '../../types/index.js';
import { describeModelClassification } from '../nemo-common/index.js';
import type {
  NemotronRnntModelConfig,
  NemotronRnntStreamingWindowMs,
} from './types.js';

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

export const DEFAULT_NEMOTRON_RNNT_STREAMING_WINDOW_MS: NemotronRnntStreamingWindowMs = 560;

export interface NemotronRnntStreamingWindow {
  readonly durationMs: NemotronRnntStreamingWindowMs;
  readonly chunkFrames: number;
  readonly encoderOutputFrames: number;
}

/**
 * The official Nemotron streaming operating points. `chunkFrames` is the
 * mel-frame input size for the exported encoder; `encoderOutputFrames` is the
 * resulting 80 ms encoder-frame count.
 */
export const NEMOTRON_RNNT_STREAMING_WINDOWS: Readonly<
  Record<NemotronRnntStreamingWindowMs, NemotronRnntStreamingWindow>
> = {
  80: { durationMs: 80, chunkFrames: 17, encoderOutputFrames: 1 },
  160: { durationMs: 160, chunkFrames: 25, encoderOutputFrames: 2 },
  320: { durationMs: 320, chunkFrames: 41, encoderOutputFrames: 4 },
  560: { durationMs: 560, chunkFrames: 65, encoderOutputFrames: 7 },
  1120: { durationMs: 1120, chunkFrames: 121, encoderOutputFrames: 14 },
};

export function resolveNemotronRnntStreamingWindow(
  durationMs: number | undefined,
): NemotronRnntStreamingWindow | undefined {
  if (durationMs === undefined) {
    return undefined;
  }
  const window = Number.isInteger(durationMs)
    ? NEMOTRON_RNNT_STREAMING_WINDOWS[
        durationMs as NemotronRnntStreamingWindowMs
      ]
    : undefined;
  if (!window) {
    throw new Error(
      `Unsupported Nemotron streaming window ${durationMs} ms. ` +
        'Supported windows: 80, 160, 320, 560, 1120 ms.',
    );
  }
  return window;
}

const BASE_NEMOTRON_RNNT_CONFIG: NemotronRnntModelConfig = {
  ecosystem: 'nemo',
  architecture: 'nemotron-rnnt',
  encoderArchitecture: 'fastconformer',
  decoderArchitecture: 'rnnt',
  sampleRate: 16000,
  frameShiftSeconds: 0.01,
  subsamplingFactor: 8,
  melBins: 128,
  preprocessorValidLengthMode: 'centered',
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
  return {
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
  };
}

export function describeNemotronRnntModel(
  modelId: string,
  classification: ModelClassification,
  config: NemotronRnntModelConfig,
): string {
  const label = describeModelClassification(classification);
  return `Nemotron 3.5 ASR Streaming model for ${modelId} (${label}, ${config.melBins} mel bins, vocab ${config.vocabularySize ?? '?'}).`;
}
