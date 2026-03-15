import type { ModelClassification } from '../../types/index.js';
import type { HfCtcModelConfig } from '../../models/hf-ctc-common/index.js';

export interface MedAsrPresetManifest {
  readonly preset: 'medasr';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<HfCtcModelConfig>;
}

export const MEDASR_PRESET_MANIFESTS: readonly MedAsrPresetManifest[] = [
  {
    preset: 'medasr',
    modelId: 'google/medasr',
    aliases: ['medasr', 'google-medasr'],
    description: 'Google MedASR preset over the shared HF CTC implementation boundary.',
    classification: {
      ecosystem: 'hf',
      processor: 'wav2vec2-conv',
      encoder: 'wav2vec2-conformer',
      decoder: 'ctc',
      topology: 'ctc',
      family: 'medasr',
      task: 'asr',
    },
    config: {
      languages: ['en'],
      vocabularySize: 32,
      tokenizer: {
        kind: 'wordpiece',
        blankTokenId: 31,
      },
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
