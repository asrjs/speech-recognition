import type { ModelClassification } from '../../types/index.js';
import { describeModelClassification } from '../nemo-common/index.js';
import type { NemoAedModelConfig, NemoAedPromptSettings } from './types.js';

export const DEFAULT_NEMO_AED_CLASSIFICATION: ModelClassification = {
  ecosystem: 'nemo',
  processor: 'nemo-mel',
  encoder: 'fastconformer',
  decoder: 'transformer-decoder',
  topology: 'aed',
  task: 'multitask-asr-translation',
};

export const DEFAULT_NEMO_AED_PROMPT: NemoAedPromptSettings = {
  sourceLanguage: 'en',
  targetLanguage: 'en',
  decoderContext: '',
  emotion: '<|emo:undefined|>',
  punctuate: true,
  inverseTextNormalization: false,
  timestamps: false,
  diarize: false,
};

/**
 * Shared baseline config for Canary-style NeMo AED exports.
 *
 * These values come from the current Canary 180M Flash reference export and
 * checkpoint config:
 * - 16 kHz mono input
 * - 128-bin NeMo mel frontend
 * - FastConformer encoder
 * - Transformer decoder
 * - aggregate tokenizer with prompt format `canary2`
 */
const BASE_NEMO_AED_CONFIG: NemoAedModelConfig = {
  ecosystem: 'nemo',
  architecture: 'nemo-aed',
  encoderArchitecture: 'fastconformer',
  decoderArchitecture: 'transformer-decoder',
  sampleRate: 16000,
  frameShiftSeconds: 0.01,
  subsamplingFactor: 8,
  melBins: 128,
  preprocessorValidLengthMode: 'centered',
  preprocessorNormalization: 'per_feature',
  vocabularySize: 5248,
  languages: ['en', 'de', 'es', 'fr'],
  encoderHiddenSize: 512,
  decoderHiddenSize: 1024,
  encoderOutputSize: 1024,
  maxTargetPositions: 1024,
  promptFormat: 'canary2',
  promptDefaults: [DEFAULT_NEMO_AED_PROMPT],
  tokenizer: {
    kind: 'aggregate',
    bosTokenId: 4,
    eosTokenId: 3,
    padTokenId: 2,
    vocabSize: 5248,
  },
};

export function parseNemoAedConfig(
  _modelId: string,
  override: Partial<NemoAedModelConfig> = {},
): NemoAedModelConfig {
  return {
    ...BASE_NEMO_AED_CONFIG,
    ...override,
    promptDefaults: override.promptDefaults ?? BASE_NEMO_AED_CONFIG.promptDefaults,
    tokenizer: {
      ...BASE_NEMO_AED_CONFIG.tokenizer,
      ...override.tokenizer,
    },
  };
}

export function describeNemoAedModel(
  modelId: string,
  classification: ModelClassification,
  config: NemoAedModelConfig,
): string {
  const label = describeModelClassification(classification);
  return `NeMo AED model for ${modelId} (${label}, ${config.melBins} mel bins, prompt format ${config.promptFormat}).`;
}
