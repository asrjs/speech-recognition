import type { NemotronRnntArtifactSource, NemotronRnntModelOptions } from '../../models/nemotron-rnnt/index.js';
import type { FamilyModelLoadRequest, ModelClassification, SpeechPresetFactory } from '../../types/index.js';

export const DEFAULT_NEMOTRON_PRESET_MODEL = 'nemotron-3.5-asr-streaming-0.6b';

export interface NemotronPresetManifest {
  readonly preset: 'nemotron';
  readonly modelId: string;
  readonly aliases?: readonly string[];
  readonly description: string;
  readonly classification: ModelClassification;
  readonly source?: NemotronRnntArtifactSource;
}

export const NEMOTRON_PRESET_MANIFESTS: readonly NemotronPresetManifest[] = [
  {
    preset: 'nemotron',
    modelId: DEFAULT_NEMOTRON_PRESET_MODEL,
    aliases: [
      'nvidia/nemotron-3.5-asr-streaming-0.6b',
      'onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4',
      'onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx',
      'codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx',
      'pantinor/nemotron-3.5-asr-streaming-0.6b-onnx',
    ],
    description:
      'Nemotron 3.5 ASR Streaming 0.6B over the cache-aware encoder + predictor + joint layout.',
    classification: {
      ecosystem: 'nemo',
      processor: 'nemo-mel',
      encoder: 'fastconformer',
      decoder: 'rnnt',
      topology: 'rnnt',
      family: 'nemotron',
      task: 'asr',
    },
    source: {
      kind: 'huggingface',
      repoId: 'onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4',
      preprocessorBackend: 'js',
    },
  },
];

function normalizePresetId(modelId: string): string {
  return modelId.trim().toLowerCase();
}

export function listNemotronPresetManifests(): readonly NemotronPresetManifest[] {
  return NEMOTRON_PRESET_MANIFESTS;
}

export function resolveNemotronPresetManifest(
  modelId: string,
): NemotronPresetManifest | undefined {
  const normalized = normalizePresetId(modelId);
  return NEMOTRON_PRESET_MANIFESTS.find(
    (manifest) =>
      normalizePresetId(manifest.modelId) === normalized ||
      manifest.aliases?.some((alias) => normalizePresetId(alias) === normalized),
  );
}

export function resolveNemotronArtifactSource(
  modelId: string,
): NemotronRnntArtifactSource | undefined {
  return resolveNemotronPresetManifest(modelId)?.source;
}

export interface CreateNemotronPresetFactoryOptions {
  readonly useManifestSource?: boolean;
}

export function createNemotronPresetFactory(
  options: CreateNemotronPresetFactoryOptions = {},
): SpeechPresetFactory<NemotronRnntModelOptions, NemotronRnntModelOptions> {
  return {
    preset: 'nemotron',
    supports(modelId?: string): boolean {
      return modelId ? resolveNemotronPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(request): Promise<
      FamilyModelLoadRequest<NemotronRnntModelOptions>
    > {
      const modelId = request.modelId ?? DEFAULT_NEMOTRON_PRESET_MODEL;
      const manifest = resolveNemotronPresetManifest(modelId);
      const manifestSource = options.useManifestSource
        ? resolveNemotronArtifactSource(modelId)
        : undefined;
      const resolvedOptions: NemotronRnntModelOptions = {
        ...request.options,
        config: {
          ...(request.options?.config ?? {}),
        },
        source: request.options?.source ?? manifestSource,
      };
      return {
        family: 'nemotron-rnnt',
        modelId,
        classification: {
          family: 'nemotron',
          ...(manifest?.classification ?? {}),
          ...request.classification,
        },
        resolvedPreset: 'nemotron',
        options: resolvedOptions,
      };
    },
  };
}

export const createNemotronPresetFamily = createNemotronPresetFactory;
