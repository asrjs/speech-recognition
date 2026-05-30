import type { ModelClassification } from '../../types/index.js';
import type { Wav2Vec2ArtifactSource, Wav2Vec2ModelConfig } from '../../models/wav2vec2/index.js';

export interface Wav2Vec2PresetManifest {
  readonly preset: 'wav2vec2';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<Wav2Vec2ModelConfig>;
  /**
   * Optional until the asrjs-owned ONNX export is published. Consumers can pass
   * a direct source for local smoke tests without relying on a hub repo.
   */
  readonly source?: Wav2Vec2ArtifactSource;
}

export const WAV2VEC2_PRESET_MANIFESTS: readonly Wav2Vec2PresetManifest[] = [
  {
    preset: 'wav2vec2',
    modelId: 'facebook/wav2vec2-base-960h',
    aliases: ['wav2vec2', 'wav2vec2-base-960h', 'base-960h'],
    description: 'Facebook Wav2Vec2 Base 960h CTC preset over raw 16 kHz waveform input.',
    classification: {
      ecosystem: 'meta',
      processor: 'wav2vec2-conv',
      encoder: 'wav2vec2-conformer',
      decoder: 'ctc',
      topology: 'ctc',
      family: 'wav2vec2',
      task: 'asr',
    },
    config: {
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
      tokenizer: {
        kind: 'char',
        blankTokenId: 0,
      },
      doStableLayerNorm: false,
      layerNormEps: 1e-5,
      featExtractActivation: 'gelu',
      convBias: false,
      featExtractNorm: 'group',
    },
    source: {
      kind: 'huggingface',
      repoId: 'ysdede/wav2vec2-base-960h-onnx',
      modelFilename: 'wav2vec2-base-960h.onnx',
      modelDataFilename: 'wav2vec2-base-960h.onnx.data',
      tokenizerFilename: 'vocab.json',
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function resolveWav2Vec2PresetManifest(
  modelId: string,
): Wav2Vec2PresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);

  return WAV2VEC2_PRESET_MANIFESTS.find((manifest) => {
    if (normalizePresetId(manifest.modelId) === normalizedModelId) {
      return true;
    }

    return (manifest.aliases ?? []).some((alias) => normalizePresetId(alias) === normalizedModelId);
  });
}
