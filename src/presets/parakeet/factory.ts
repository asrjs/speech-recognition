import type { ModelClassification, SpeechModelFactory } from '../../types/index.js';
import {
  createNemoTdtModelFamily,
  describeNemoTdtModel,
  parseNemoTdtConfig,
  type CreateNemoTdtModelFamilyOptions,
  type NemoTdtArtifactSource,
  type NemoTdtModelOptions,
  type NemoTdtNativeTranscript,
  type NemoTdtTranscriptionOptions
} from '../../models/nemo-tdt/index.js';
import {
  resolveParakeetArtifactSource,
  resolveParakeetPresetManifest
} from './manifest.js';

export interface CreateParakeetPresetFactoryOptions {
  readonly dependencies?: CreateNemoTdtModelFamilyOptions['dependencies'];
  readonly useManifestSource?: boolean;
}

function parakeetClassificationFallback(classification: Partial<ModelClassification> = {}): boolean {
  return classification.family === 'parakeet'
    || (
      classification.ecosystem === 'nemo'
      && classification.decoder === 'tdt'
      && classification.family === 'parakeet'
    );
}

export function createParakeetPresetFactory(
  options: CreateParakeetPresetFactoryOptions = {}
): SpeechModelFactory<NemoTdtModelOptions, NemoTdtTranscriptionOptions, NemoTdtNativeTranscript> {
  const baseFactory = createNemoTdtModelFamily({
    family: 'parakeet',
    classification: {
      family: 'parakeet'
    },
    dependencies: options.dependencies,
    supportsModel(modelId, classification) {
      return resolveParakeetPresetManifest(modelId) !== undefined || parakeetClassificationFallback(classification);
    },
    resolveConfig(modelId, request) {
      const manifest = resolveParakeetPresetManifest(modelId);
      return parseNemoTdtConfig(modelId, {
        ...manifest?.config,
        ...request.options?.config
      });
    },
    describeModel(modelId, classification, config) {
      const manifest = resolveParakeetPresetManifest(modelId);
      return manifest?.description ?? describeNemoTdtModel(modelId, classification, config);
    }
  });

  return {
    ...baseFactory,
    async createModel(request, context) {
      const manifestSource: NemoTdtArtifactSource | undefined = options.useManifestSource
        ? resolveParakeetArtifactSource(request.modelId)
        : undefined;
      const nextOptions: NemoTdtModelOptions | undefined = request.options
        ? {
            ...request.options,
            source: request.options.source ?? manifestSource
          }
        : (manifestSource ? { source: manifestSource } : undefined);

      return baseFactory.createModel(
        {
          ...request,
          options: nextOptions
        },
        context
      );
    }
  };
}

export const createParakeetPresetFamily = createParakeetPresetFactory;
