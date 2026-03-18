import type { ModelClassification } from '../../types/index.js';
import { describeModelClassification } from '../nemo-common/index.js';
import type { NemoRnntModelConfig } from './types.js';

export const DEFAULT_NEMO_RNNT_CLASSIFICATION: ModelClassification = {
  ecosystem: 'nemo',
  processor: 'nemo-mel',
  encoder: 'fastconformer',
  decoder: 'rnnt',
  topology: 'rnnt',
  task: 'asr',
};

const BASE_NEMO_RNNT_CONFIG: NemoRnntModelConfig = {
  ecosystem: 'nemo',
  architecture: 'nemo-rnnt',
  encoderArchitecture: 'fastconformer',
  decoderArchitecture: 'rnnt',
  sampleRate: 16000,
  frameShiftSeconds: 0.01,
  subsamplingFactor: 8,
  melBins: 128,
  preprocessorValidLengthMode: 'centered',
  preprocessorNormalization: 'none',
  predictionHiddenSize: 640,
  predictionLayers: 1,
  maxSymbolsPerStep: 10,
  vocabularySize: 1026,
  languages: ['en'],
  tokenizer: {
    kind: 'sentencepiece',
    blankTokenId: 1026,
  },
};

export function parseNemoRnntConfig(
  _modelId: string,
  override: Partial<NemoRnntModelConfig> = {},
): NemoRnntModelConfig {
  return {
    ...BASE_NEMO_RNNT_CONFIG,
    ...override,
    tokenizer: {
      ...BASE_NEMO_RNNT_CONFIG.tokenizer,
      ...override.tokenizer,
    },
  };
}

export function describeNemoRnntModel(
  modelId: string,
  classification: ModelClassification,
  config: NemoRnntModelConfig,
): string {
  const label = describeModelClassification(classification);
  return `NeMo RNNT model for ${modelId} (${label}, ${config.melBins} mel bins).`;
}
