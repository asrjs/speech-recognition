import type { ModelClassification } from '../../types/index.js';
import type { Wav2Vec2ModelConfig } from './types.js';

export const DEFAULT_WAV2VEC2_CLASSIFICATION: ModelClassification = {
  ecosystem: 'meta',
  processor: 'wav2vec2-conv',
  encoder: 'wav2vec2-conformer',
  decoder: 'ctc',
  topology: 'ctc',
  task: 'asr',
};

export const DEFAULT_WAV2VEC2_CONFIG: Wav2Vec2ModelConfig = {
  ecosystem: 'meta',
  architecture: 'wav2vec2',
  processorArchitecture: 'wav2vec2-conv',
  encoderArchitecture: 'wav2vec2-conformer',
  decoderArchitecture: 'ctc',
  sampleRate: 16000,
  outputStride: 320,
  numFeatExtractLayers: 7,
  convDim: 512,
  convKernel: [10, 3, 3, 3, 3, 2, 2],
  convStride: [5, 2, 2, 2, 2, 2, 2],
  hiddenSize: 768,
  numHiddenLayers: 12,
  numAttentionHeads: 12,
  vocabularySize: 32,
  ctcBlankId: 0,
  languages: ['en'],
  tokenizer: { kind: 'char' },
  doStableLayerNorm: false,
  layerNormEps: 1e-5,
  featExtractActivation: 'gelu',
  convBias: false,
  featExtractNorm: 'group',
};

function asNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asNumberArray(value: unknown, fallback: readonly number[]): readonly number[] {
  if (!Array.isArray(value)) {
    return fallback;
  }
  const result: number[] = [];
  for (const item of value) {
    if (typeof item === 'number' && Number.isFinite(item)) {
      result.push(item);
    } else {
      return fallback;
    }
  }
  return result.length > 0 ? result : fallback;
}

function asString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.length > 0 ? value : fallback;
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback;
}

export function parseWav2Vec2Config(raw: Record<string, unknown>): Wav2Vec2ModelConfig {
  const d = DEFAULT_WAV2VEC2_CONFIG;

  return {
    ...d,
    numHiddenLayers: asNumber(raw['num_hidden_layers'], d.numHiddenLayers),
    hiddenSize: asNumber(raw['hidden_size'], d.hiddenSize),
    numAttentionHeads: asNumber(raw['num_attention_heads'], d.numAttentionHeads),
    convDim: asNumber(raw['conv_dim'], d.convDim),
    convKernel: asNumberArray(raw['conv_kernel'], d.convKernel),
    convStride: asNumberArray(raw['conv_stride'], d.convStride),
    convBias: asBoolean(raw['conv_bias'], d.convBias),
    vocabularySize: asNumber(raw['vocab_size'], d.vocabularySize),
    ctcBlankId: asNumber(raw['pad_token_id'], d.ctcBlankId),
    doStableLayerNorm: asBoolean(raw['do_stable_layer_norm'], d.doStableLayerNorm),
    layerNormEps: asNumber(raw['layer_norm_eps'], d.layerNormEps),
    featExtractActivation: asString(raw['feat_extract_activation'], d.featExtractActivation),
    featExtractNorm:
      raw['feat_extract_norm'] === 'layer' || raw['feat_extract_norm'] === 'group'
        ? raw['feat_extract_norm']
        : d.featExtractNorm,
    numFeatExtractLayers: asNumber(raw['num_feat_extract_layers'], d.numFeatExtractLayers),
    outputStride: asNumber(raw['output_stride'], d.outputStride),
    sampleRate: asNumber(raw['sampling_rate'], d.sampleRate),
    tokenizer: {
      kind: 'char',
      blankTokenId: asNumber(raw['pad_token_id'], d.ctcBlankId),
    },
  };
}

export function describeWav2Vec2Model(config: Wav2Vec2ModelConfig): string {
  const totalStride = config.convStride.reduce((acc, s) => acc * s, 1);
  const framesPerSec = config.sampleRate / totalStride;

  return (
    `Wav2Vec2 (${config.processorArchitecture} → ${config.encoderArchitecture} → ${config.decoderArchitecture}): ` +
    `${config.numHiddenLayers} layers, ${config.hiddenSize} hidden, ${config.numAttentionHeads} heads, ` +
    `vocab ${config.vocabularySize}, ${framesPerSec.toFixed(1)} frames/sec, ` +
    `sample rate ${config.sampleRate} Hz.`
  );
}
