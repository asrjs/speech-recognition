import type { ModelClassification } from '../../types/index.js';
import type { LasrCtcModelConfig } from './types.js';

export const DEFAULT_LASR_CTC_CLASSIFICATION: ModelClassification = {
  ecosystem: 'lasr',
  processor: 'kaldi-mel',
  encoder: 'conformer',
  decoder: 'ctc',
  topology: 'ctc',
  task: 'asr',
};

const BASE_LASR_CTC_CONFIG: LasrCtcModelConfig = {
  ecosystem: 'lasr',
  architecture: 'lasr-ctc',
  processorArchitecture: 'kaldi-mel',
  encoderArchitecture: 'conformer',
  decoderArchitecture: 'ctc',
  sampleRate: 16000,
  rawStride: 160,
  nMels: 128,
  featureHopSeconds: 0.01,
  vocabularySize: 32,
  languages: ['en'],
  tokenizer: {
    kind: 'sentencepiece',
    blankTokenId: 31,
  },
};

export function parseLasrCtcConfig(
  _modelId: string,
  override: Partial<LasrCtcModelConfig> = {},
): LasrCtcModelConfig {
  return {
    ...BASE_LASR_CTC_CONFIG,
    ...override,
    tokenizer: {
      ...BASE_LASR_CTC_CONFIG.tokenizer,
      ...override.tokenizer,
    },
  };
}

export function describeLasrCtcModel(
  modelId: string,
  classification: ModelClassification,
  config: LasrCtcModelConfig,
): string {
  return `LASR CTC family for ${modelId} (${classification.processor ?? config.processorArchitecture} -> ${classification.encoder ?? config.encoderArchitecture} -> ${classification.decoder ?? config.decoderArchitecture}).`;
}
