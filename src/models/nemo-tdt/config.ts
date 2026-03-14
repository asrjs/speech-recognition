import type { ModelClassification } from '../../types/index.js';
import { describeModelClassification } from '../nemo-common/index.js';
import type { NemoTdtModelConfig } from './types.js';

export const DEFAULT_NEMO_TDT_CLASSIFICATION: ModelClassification = {
  ecosystem: 'nemo',
  processor: 'nemo-mel',
  encoder: 'fastconformer',
  decoder: 'tdt',
  topology: 'tdt',
  task: 'asr'
};

const BASE_NEMO_TDT_CONFIG: NemoTdtModelConfig = {
  ecosystem: 'nemo',
  architecture: 'nemo-tdt',
  encoderArchitecture: 'fastconformer',
  decoderArchitecture: 'tdt',
  sampleRate: 16000,
  frameShiftSeconds: 0.01,
  subsamplingFactor: 8,
  melBins: 128,
  predictionHiddenSize: 640,
  predictionLayers: 2,
  maxSymbolsPerStep: 10,
  vocabularySize: 1025,
  languages: ['en'],
  tokenizer: {
    kind: 'sentencepiece',
    blankTokenId: 0
  }
};

export function parseNemoTdtConfig(
  _modelId: string,
  override: Partial<NemoTdtModelConfig> = {}
): NemoTdtModelConfig {
  return {
    ...BASE_NEMO_TDT_CONFIG,
    ...override,
    tokenizer: {
      ...BASE_NEMO_TDT_CONFIG.tokenizer,
      ...override.tokenizer
    }
  };
}

export function describeNemoTdtModel(
  modelId: string,
  classification: ModelClassification,
  config: NemoTdtModelConfig
): string {
  const label = describeModelClassification(classification);
  return `NeMo TDT model for ${modelId} (${label}, ${config.melBins} mel bins).`;
}
