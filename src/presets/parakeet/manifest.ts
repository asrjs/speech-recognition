import type { ModelClassification } from '../../types/index.js';
import type { NemoTdtArtifactSource, NemoTdtModelConfig } from '../../models/nemo-tdt/index.js';
import type { NemoRnntArtifactSource, NemoRnntModelConfig } from '../../models/nemo-rnnt/index.js';
import { getModelConfig } from './catalog.js';

export interface ParakeetPresetManifest {
  readonly preset: 'parakeet';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly config: Partial<NemoTdtModelConfig> | Partial<NemoRnntModelConfig>;
  readonly source?: NemoTdtArtifactSource | NemoRnntArtifactSource;
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
      // v3 ONNX encoder output and decoder input are [batch, 1024, time].
      encoderOutputLayout: 'bdt',
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
  {
    preset: 'parakeet',
    modelId: 'parakeet-realtime-eou-120m-v1',
    aliases: [
      'nvidia/parakeet_realtime_eou_120m-v1',
      'nvidia/parakeet-realtime-eou-120m-v1',
      'ysdede/parakeet-realtime-eou-120m-v1-onnx',
    ],
    description: 'Realtime Parakeet preset over the NeMo RNNT implementation boundary.',
    classification: {
      ecosystem: 'nemo',
      processor: 'nemo-mel',
      encoder: 'fastconformer',
      decoder: 'rnnt',
      topology: 'rnnt',
      family: 'parakeet',
      task: 'asr',
    },
    config: {
      vocabularySize: 1026,
      languages: ['en'],
      melBins: 128,
      predictionHiddenSize: 640,
      predictionLayers: 1,
      tokenizer: {
        kind: 'sentencepiece',
        blankTokenId: 1026,
      },
    },
    source: {
      kind: 'huggingface',
      repoId: 'ysdede/parakeet-realtime-eou-120m-v1-onnx',
      preprocessorName: 'nemo128',
      preprocessorBackend: 'js',
    },
    quirks: {
      preprocessorBackend: 'js',
    },
  },
  {
    preset: 'parakeet',
    modelId: 'oruk/orukeet',
    aliases: ['oruk/orukeet-v0.1.0', 'orukeet'],
    description:
      'Community multilingual fine-tune of Parakeet TDT v3 over the same NeMo implementation boundary.',
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
      languages: [
        'bg',
        'cs',
        'da',
        'de',
        'el',
        'en',
        'es',
        'et',
        'fi',
        'fr',
        'hr',
        'hu',
        'it',
        'lt',
        'lv',
        'mt',
        'nl',
        'pl',
        'pt',
        'ro',
        'ru',
        'sk',
        'sl',
        'sv',
        'uk',
      ],
      melBins: 128,
      tokenizer: {
        kind: 'sentencepiece',
        blankTokenId: 8192,
      },
      // Audited own Orukeet export: encoder output and decoder input are [B, D, T].
      encoderOutputLayout: 'bdt',
    },
    // No Hub source: at revision b59c13a7 the upstream repo publishes graphs
    // under onnx/combined-v0.1.0-int8/ and a tokenizer.json, not the root
    // encoder-model/decoder_joint-model/vocab.txt layout this preset resolves.
    // Advertising one would 404, so Orukeet is local-folder only until a
    // matching validated export source exists.
    quirks: {
      preprocessorBackend: 'js',
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

function withCatalogDefaults(
  manifest: ParakeetPresetManifest | undefined,
): ParakeetPresetManifest | undefined {
  const config = manifest ? getModelConfig(manifest.modelId) : null;
  if (!manifest || !config || manifest.source?.kind !== 'huggingface') {
    return manifest;
  }

  return {
    ...manifest,
    source: {
      ...manifest.source,
      revision: manifest.source.revision ?? config.defaultRevision,
      cacheKeyFallbackRevisions: config.cacheKeyFallbackRevisions,
    },
  };
}

export function listParakeetPresetManifests(): readonly ParakeetPresetManifest[] {
  return PARAKEET_PRESET_MANIFESTS.map((manifest) => withCatalogDefaults(manifest)!);
}

export function resolveParakeetPresetManifest(modelId: string): ParakeetPresetManifest | undefined {
  const normalizedModelId = normalizePresetId(modelId);

  return withCatalogDefaults(
    PARAKEET_PRESET_MANIFESTS.find((manifest) => {
      if (normalizePresetId(manifest.modelId) === normalizedModelId) {
        return true;
      }

      return (manifest.aliases ?? []).some(
        (alias) => normalizePresetId(alias) === normalizedModelId,
      );
    }),
  );
}

export function resolveParakeetArtifactSource(
  modelId: string,
): NemoTdtArtifactSource | NemoRnntArtifactSource | undefined {
  return resolveParakeetPresetManifest(modelId)?.source;
}
