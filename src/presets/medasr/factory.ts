import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateLasrCtcModelFamilyOptions,
  LasrCtcModelOptions,
} from '../../models/lasr-ctc/index.js';
import { resolveMedAsrPresetManifest } from './manifest.js';

export interface CreateMedAsrPresetFactoryOptions {
  readonly dependencies?: CreateLasrCtcModelFamilyOptions['dependencies'];
}

export function createMedAsrPresetFactory(
  options: CreateMedAsrPresetFactoryOptions = {},
): SpeechPresetFactory<LasrCtcModelOptions, LasrCtcModelOptions> {
  void options.dependencies;

  return {
    preset: 'medasr',
    supports(modelId?: string): boolean {
      return modelId ? resolveMedAsrPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<LasrCtcModelOptions>> {
      const modelId = request.modelId ?? 'google/medasr';
      const manifest = resolveMedAsrPresetManifest(modelId);

      return {
        family: 'lasr-ctc',
        modelId,
        classification: {
          family: 'medasr',
          ...request.classification,
        },
        resolvedPreset: 'medasr',
        options: {
          ...request.options,
          config: {
            ...manifest?.config,
            ...request.options?.config,
          },
          source: request.options?.source ?? manifest?.source,
        },
      };
    },
  };
}
