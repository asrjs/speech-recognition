import type { ModelClassification } from '../../types/index.js';
import type {
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
} from '../../models/whisper-seq2seq/index.js';

export interface WhisperPresetManifest {
  readonly preset: 'whisper';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<WhisperSeq2SeqModelConfig>;
  readonly source?: WhisperArtifactSource;
}

export const WHISPER_PRESET_MANIFESTS: readonly WhisperPresetManifest[] = [
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-tiny',
    aliases: ['whisper-tiny', 'openai/whisper-tiny'],
    description:
      'Whisper Tiny multilingual preset. ~39M params. Fastest, lowest memory. Good for smoke tests and low-resource devices.',
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
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 80,
      vocabularySize: 51865,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-tiny_timestamped',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-base',
    aliases: ['whisper-base', 'openai/whisper-base', 'whisper-base-multilingual'],
    description:
      'Whisper Base multilingual preset. ~74M params. Default baseline for browser Whisper. Good balance of quality and speed.',
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
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 80,
      vocabularySize: 51865,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-base_timestamped',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-small',
    aliases: ['whisper-small', 'openai/whisper-small'],
    description:
      'Whisper Small multilingual preset. ~244M params. Better quality than base, heavier. Use when accuracy matters more than speed.',
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
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 80,
      vocabularySize: 51865,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-small_timestamped',
    },
  },
  {
    preset: 'whisper',
    modelId: 'onnx-community/whisper-large-v3-turbo',
    aliases: ['whisper-large-v3-turbo', 'openai/whisper-large-v3-turbo'],
    description:
      'Whisper Large-v3 Turbo multilingual preset. ~809M params. Experimental desktop/WebGPU only. Do not use on mobile or low-memory devices.',
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
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      melBins: 128,
      vocabularySize: 51866,
      languages: ['auto'],
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-large-v3-turbo_timestamped',
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

export function resolveWhisperArtifactSource(
  modelId: string,
): WhisperArtifactSource | undefined {
  return resolveWhisperPresetManifest(modelId)?.source;
}
