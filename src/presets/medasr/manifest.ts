import type { ModelClassification } from '../../types/index.js';
import type { LasrCtcArtifactSource, LasrCtcModelConfig } from '../../models/lasr-ctc/index.js';

export interface MedAsrPresetManifest {
  readonly preset: 'medasr';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<LasrCtcModelConfig>;
  readonly source?: LasrCtcArtifactSource;
}

export const MEDASR_PRESET_MANIFESTS: readonly MedAsrPresetManifest[] = [
  {
    preset: 'medasr',
    modelId: 'google/medasr',
    aliases: ['medasr', 'google-medasr'],
    description: 'Google MedASR preset over the shared LASR CTC implementation boundary.',
    classification: {
      ecosystem: 'lasr',
      processor: 'kaldi-mel',
      encoder: 'conformer',
      decoder: 'ctc',
      topology: 'ctc',
      family: 'medasr',
      task: 'asr',
    },
    config: {
      languages: ['en'],
      vocabularySize: 32,
      nMels: 128,
      featureHopSeconds: 0.01,
      tokenizer: {
        kind: 'sentencepiece',
        blankTokenId: 31,
      },
    },
    source: {
      kind: 'huggingface',
      repoId: 'ysdede/medasr-onnx',
      tokenizerFilename: 'tokens.txt',
      modelFilename: 'model.onnx',
      modelDataFilename: 'model.onnx.data',
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function resolveMedAsrPresetManifest(modelId: string): MedAsrPresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);

  return MEDASR_PRESET_MANIFESTS.find((manifest) => {
    if (normalizePresetId(manifest.modelId) === normalizedModelId) {
      return true;
    }

    return (manifest.aliases ?? []).some((alias) => normalizePresetId(alias) === normalizedModelId);
  });
}
