import type { ModelClassification } from '../../types/index.js';
import type { Qwen3AsrModelConfig } from './types.js';

export const DEFAULT_QWEN3_ASR_CLASSIFICATION: ModelClassification = {
  ecosystem: 'qwen',
  processor: 'qwen3-asr-mel',
  encoder: 'qwen3-asr-audio',
  decoder: 'qwen3',
  topology: 'speech-llm',
  task: 'multilingual-asr',
};

const SUPPORTED_LANGUAGES = [
  'Chinese',
  'English',
  'Cantonese',
  'French',
  'German',
  'Italian',
  'Spanish',
  'Portuguese',
  'Russian',
  'Arabic',
  'Japanese',
  'Korean',
  'Hindi',
  'Turkish',
  'Vietnamese',
  'Indonesian',
  'Thai',
  'Malay',
  'Dutch',
  'Swedish',
  'Danish',
  'Polish',
  'Finnish',
  'Czech',
  'Filipino',
  'Persian',
  'Greek',
  'Hungarian',
  'Macedonian',
  'Romanian',
] as const;

export const DEFAULT_QWEN3_ASR_CONFIG: Qwen3AsrModelConfig = {
  ecosystem: 'qwen',
  architecture: 'qwen3-asr',
  processorArchitecture: 'qwen3-asr-mel',
  encoderArchitecture: 'qwen3-asr-audio-encoder',
  decoderArchitecture: 'qwen3-asr-qwen3-decoder',
  sampleRate: 16000,
  melBins: 128,
  hopLength: 160,
  nFft: 400,
  minInputSamples: 8000,
  maxInputDurationSec: 30,
  languages: SUPPORTED_LANGUAGES,
  tokenizer: {
    kind: 'bpe',
    vocabSize: 151936,
    eosTokenId: 151643,
    padTokenId: 151645,
  },
  graph: {
    numLayers: 28,
    numKvHeads: 8,
    headDim: 128,
    hiddenSize: 1024,
    vocabularySize: 151936,
    audioWindowFrames: 800,
    audioTokensPerWindow: 104,
    audioFramesMultiple: 800,
    batchSize: 1,
    pastSeedLength: 1,
    pastSeedValue: 0,
    pastSeedAttentionMask: -65504,
    eosTokenIds: [151643, 151645],
    padTokenId: 151645,
    audioPadTokenId: 151676,
    audioStartTokenId: 151669,
    audioEndTokenId: 151670,
    imStartTokenId: 151644,
    imEndTokenId: 151645,
    logitsOutputLocation: 'cpu',
    cacheOutputLocation: 'gpu-buffer',
  },
};

export function parseQwen3AsrConfig(
  _modelId: string,
  override: Partial<Qwen3AsrModelConfig> = {},
): Qwen3AsrModelConfig {
  return {
    ...DEFAULT_QWEN3_ASR_CONFIG,
    ...override,
    tokenizer: {
      ...DEFAULT_QWEN3_ASR_CONFIG.tokenizer,
      ...override.tokenizer,
    },
    graph: {
      ...DEFAULT_QWEN3_ASR_CONFIG.graph,
      ...override.graph,
      eosTokenIds: override.graph?.eosTokenIds ?? DEFAULT_QWEN3_ASR_CONFIG.graph.eosTokenIds,
    },
  };
}

export function describeQwen3AsrModel(
  modelId: string,
  classification: ModelClassification,
  config: Qwen3AsrModelConfig,
): string {
  return `Qwen3-ASR speech-LLM for ${modelId} (${classification.task}, ${config.graph.hiddenSize}-wide decoder, ${config.melBins}-bin mel frontend).`;
}

export function normalizeQwenLanguage(language: string | undefined): string | undefined {
  if (!language || language.trim().length === 0) return undefined;
  const normalized = language.trim().toLowerCase();
  const aliases: Record<string, string> = {
    zh: 'Chinese',
    'zh-cn': 'Chinese',
    en: 'English',
    fr: 'French',
    de: 'German',
    it: 'Italian',
    es: 'Spanish',
    pt: 'Portuguese',
    ru: 'Russian',
    ar: 'Arabic',
    ja: 'Japanese',
    ko: 'Korean',
    hi: 'Hindi',
    tr: 'Turkish',
    vi: 'Vietnamese',
    id: 'Indonesian',
    th: 'Thai',
    nl: 'Dutch',
    sv: 'Swedish',
    da: 'Danish',
    pl: 'Polish',
    fi: 'Finnish',
    cs: 'Czech',
    fil: 'Filipino',
    fa: 'Persian',
    el: 'Greek',
    hu: 'Hungarian',
    mk: 'Macedonian',
    ro: 'Romanian',
  };
  return aliases[normalized] ??
    DEFAULT_QWEN3_ASR_CONFIG.languages.find((candidate) => candidate.toLowerCase() === normalized) ??
    language.trim();
}
