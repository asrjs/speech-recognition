import type { ModelClassification } from '../../types/index.js';
import type { HfCtcModelConfig } from './types.js';

export const DEFAULT_HF_CTC_CLASSIFICATION: ModelClassification = {
  ecosystem: 'hf',
  processor: 'wav2vec2-conv',
  encoder: 'wav2vec2-conformer',
  decoder: 'ctc',
  topology: 'ctc',
  task: 'asr'
};

const BASE_HF_CTC_CONFIG: HfCtcModelConfig = {
  ecosystem: 'hf',
  architecture: 'hf-ctc-common',
  processorArchitecture: 'wav2vec2-conv',
  encoderArchitecture: 'wav2vec2-conformer',
  decoderArchitecture: 'ctc',
  sampleRate: 16000,
  rawStride: 320,
  vocabularySize: 32,
  languages: ['en'],
  tokenizer: {
    kind: 'wordpiece',
    blankTokenId: 31
  }
};

export function parseHfCtcConfig(
  _modelId: string,
  override: Partial<HfCtcModelConfig> = {}
): HfCtcModelConfig {
  return {
    ...BASE_HF_CTC_CONFIG,
    ...override,
    tokenizer: {
      ...BASE_HF_CTC_CONFIG.tokenizer,
      ...override.tokenizer
    }
  };
}

export function describeHfCtcModel(
  modelId: string,
  classification: ModelClassification,
  config: HfCtcModelConfig
): string {
  return `HF CTC scaffold model for ${modelId} (${classification.processor ?? 'raw-conv'} -> ${classification.encoder ?? config.encoderArchitecture} -> CTC).`;
}
