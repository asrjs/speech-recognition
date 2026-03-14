import type { ModelClassification } from '../../types/index.js';
import type { WhisperSeq2SeqModelConfig } from './types.js';

export const DEFAULT_WHISPER_CLASSIFICATION: ModelClassification = {
  ecosystem: 'openai',
  processor: 'whisper-mel',
  encoder: 'whisper-transformer',
  decoder: 'transformer-decoder',
  topology: 'aed',
  task: 'multitask-asr-translation'
};

const BASE_WHISPER_CONFIG: WhisperSeq2SeqModelConfig = {
  ecosystem: 'openai',
  architecture: 'whisper-seq2seq',
  processorArchitecture: 'whisper-mel',
  encoderArchitecture: 'whisper-transformer',
  decoderArchitecture: 'transformer-decoder',
  sampleRate: 16000,
  melBins: 80,
  maxSourcePositions: 1500,
  maxTargetPositions: 448,
  vocabularySize: 51865,
  languages: ['auto'],
  tokenizer: {
    kind: 'tiktoken',
    bosTokenId: 50257,
    eosTokenId: 50256,
    padTokenId: 50256
  }
};

export function parseWhisperSeq2SeqConfig(
  _modelId: string,
  override: Partial<WhisperSeq2SeqModelConfig> = {}
): WhisperSeq2SeqModelConfig {
  return {
    ...BASE_WHISPER_CONFIG,
    ...override,
    tokenizer: {
      ...BASE_WHISPER_CONFIG.tokenizer,
      ...override.tokenizer
    }
  };
}

export function describeWhisperSeq2SeqModel(
  modelId: string,
  classification: ModelClassification,
  config: WhisperSeq2SeqModelConfig
): string {
  return `Whisper seq2seq scaffold model for ${modelId} (${classification.processor ?? config.processorArchitecture} -> ${classification.encoder ?? config.encoderArchitecture} -> ${classification.topology ?? 'aed'}).`;
}
