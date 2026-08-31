import type { ModelClassification } from '../../types/index.js';
import { describeModelClassification } from '../nemo-common/index.js';
import type { NemotronRnntModelConfig } from './types.js';

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
  preprocessorValidLengthMode: 'centered',
  preprocessorNormalization: 'none',
  predictionHiddenSize: 640,
  predictionLayers: 2,
  chunkFrames: 65,
  encoderOutputFramesPerChunk: 7,
  encoderCache: DEFAULT_NEMOTRON_RNNT_ENCODER_CACHE,
  promptIds: DEFAULT_NEMOTRON_RNNT_PROMPT_IDS,
  defaultPromptId: DEFAULT_NEMOTRON_RNNT_PROMPT_IDS.auto,
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
