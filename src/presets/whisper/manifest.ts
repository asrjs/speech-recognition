import type { ModelClassification } from '../../types/index.js';
import type { WhisperSeq2SeqModelConfig } from '../../models/whisper-seq2seq/index.js';

export interface WhisperPresetManifest {
  readonly preset: 'whisper';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<WhisperSeq2SeqModelConfig>;
}

export const WHISPER_PRESET_MANIFESTS: readonly WhisperPresetManifest[] = [
  {
    preset: 'whisper',
    modelId: 'openai/whisper-base',
    aliases: ['whisper-base', 'whisper-base-multilingual'],
    description: 'Whisper base preset over the shared Whisper seq2seq implementation boundary.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 1500,
      maxTargetPositions: 448,
      languages: ['auto'],
    },
  },
  {
    preset: 'whisper',
    modelId: 'openai/whisper-large-v3',
    aliases: ['whisper-large-v3', 'whisper-large'],
    description: 'Whisper large-v3 preset over the shared Whisper seq2seq implementation boundary.',
    classification: {
      ecosystem: 'openai',
      processor: 'whisper-mel',
      encoder: 'whisper-transformer',
      decoder: 'transformer-decoder',
      topology: 'aed',
      family: 'whisper',
      task: 'multitask-asr-translation',
    },
    config: {
      maxSourcePositions: 1500,
      maxTargetPositions: 448,
      languages: ['auto'],
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function resolveWhisperPresetManifest(modelId: string): WhisperPresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);

  return WHISPER_PRESET_MANIFESTS.find((manifest) => {
    if (normalizePresetId(manifest.modelId) === normalizedModelId) {
      return true;
    }

    return (manifest.aliases ?? []).some((alias) => normalizePresetId(alias) === normalizedModelId);
  });
}
