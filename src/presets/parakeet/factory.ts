import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateNemoTdtModelFamilyOptions,
  NemoTdtModelOptions,
} from '../../models/nemo-tdt/index.js';
import { DEFAULT_MODEL } from './catalog.js';
import { resolveParakeetArtifactSource, resolveParakeetPresetManifest } from './manifest.js';

export interface CreateParakeetPresetFactoryOptions {
  readonly dependencies?: CreateNemoTdtModelFamilyOptions['dependencies'];
  readonly useManifestSource?: boolean;
}

export function createParakeetPresetFactory(
  options: CreateParakeetPresetFactoryOptions = {},
): SpeechPresetFactory<NemoTdtModelOptions, NemoTdtModelOptions> {
  return {
    preset: 'parakeet',
    supports(modelId?: string): boolean {
      return modelId ? resolveParakeetPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<NemoTdtModelOptions>> {
      const modelId = request.modelId ?? DEFAULT_MODEL;
      const manifest = resolveParakeetPresetManifest(modelId);
      const manifestSource = options.useManifestSource
        ? resolveParakeetArtifactSource(modelId)
        : undefined;

      return {
        family: 'nemo-tdt',
        modelId,
        classification: {
          family: 'parakeet',
          ...request.classification,
        },
        resolvedPreset: 'parakeet',
        options: {
          ...request.options,
          config: {
            ...manifest?.config,
            ...request.options?.config,
          },
          source: request.options?.source ?? manifestSource,
        },
      };
    },
  };
}

export const createParakeetPresetFamily = createParakeetPresetFactory;
