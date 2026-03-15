import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateHfCtcModelFamilyOptions,
  HfCtcModelOptions,
} from '../../models/hf-ctc-common/index.js';
import { resolveMedAsrPresetManifest } from './manifest.js';

export interface CreateMedAsrPresetFactoryOptions {
  readonly dependencies?: CreateHfCtcModelFamilyOptions['dependencies'];
}

export function createMedAsrPresetFactory(
  _options: CreateMedAsrPresetFactoryOptions = {},
): SpeechPresetFactory<HfCtcModelOptions, HfCtcModelOptions> {
  return {
    preset: 'medasr',
    supports(modelId?: string): boolean {
      return modelId ? resolveMedAsrPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<HfCtcModelOptions>> {
      const modelId = request.modelId ?? 'google/medasr';
      const manifest = resolveMedAsrPresetManifest(modelId);

      return {
        family: 'hf-ctc',
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
        },
      };
    },
  };
}
