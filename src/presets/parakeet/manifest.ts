import type { ModelClassification } from '../../types/index.js';
import type { NemoTdtArtifactSource, NemoTdtModelConfig } from '../../models/nemo-tdt/index.js';

export interface ParakeetPresetManifest {
  readonly preset: 'parakeet';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<NemoTdtModelConfig>;
  readonly source?: NemoTdtArtifactSource;
  readonly quirks?: {
    readonly preprocessorBackend?: 'js' | 'onnx';
  };
}

export const PARAKEET_PRESET_MANIFESTS: readonly ParakeetPresetManifest[] = [
  {
    preset: 'parakeet',
    modelId: 'parakeet-tdt-0.6b-v2',
    aliases: ['nvidia/parakeet-tdt-0.6b-v2', 'ysdede/parakeet-tdt-0.6b-v2-onnx'],
    description: 'Parakeet preset over the NeMo TDT implementation boundary.',
    classification: {
      ecosystem: 'nemo',
      processor: 'nemo-mel',
      encoder: 'fastconformer',
      decoder: 'tdt',
      topology: 'tdt',
      family: 'parakeet',
      task: 'asr',
    },
    config: {
      vocabularySize: 1025,
      languages: ['en'],
      melBins: 128,
    },
    source: {
      kind: 'huggingface',
      repoId: 'ysdede/parakeet-tdt-0.6b-v2-onnx',
      preprocessorName: 'nemo128',
      preprocessorBackend: 'js',
    },
    quirks: {
      preprocessorBackend: 'js',
    },
  },
  {
    preset: 'parakeet',
    modelId: 'parakeet-tdt-0.6b-v3',
    aliases: [
      'nvidia/parakeet-tdt-0.6b-v3',
      'ysdede/parakeet-tdt-0.6b-v3-onnx',
      'parakeet-tdt-multilingual',
    ],
    description: 'Multilingual Parakeet preset over the NeMo TDT implementation boundary.',
    classification: {
      ecosystem: 'nemo',
      processor: 'nemo-mel',
      encoder: 'fastconformer',
      decoder: 'tdt',
      topology: 'tdt',
      family: 'parakeet',
      task: 'asr',
    },
    config: {
      vocabularySize: 8193,
      languages: ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'uk', 'ja', 'ko', 'zh'],
      melBins: 128,
    },
    source: {
      kind: 'huggingface',
      repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
      preprocessorName: 'nemo128',
      preprocessorBackend: 'js',
    },
    quirks: {
      preprocessorBackend: 'js',
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function listParakeetPresetManifests(): readonly ParakeetPresetManifest[] {
  return PARAKEET_PRESET_MANIFESTS;
}

export function resolveParakeetPresetManifest(modelId: string): ParakeetPresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);

  return PARAKEET_PRESET_MANIFESTS.find((manifest) => {
    if (normalizePresetId(manifest.modelId) === normalizedModelId) {
      return true;
    }

    return (manifest.aliases ?? []).some((alias) => normalizePresetId(alias) === normalizedModelId);
  });
}

export function resolveParakeetArtifactSource(modelId: string): NemoTdtArtifactSource | undefined {
  return resolveParakeetPresetManifest(modelId)?.source;
}
