import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateNemoTdtModelFamilyOptions,
  NemoTdtModelOptions,
} from '../../models/nemo-tdt/index.js';
import type { NemoRnntModelOptions } from '../../models/nemo-rnnt/index.js';
import { DEFAULT_MODEL } from './catalog.js';
import { resolveParakeetArtifactSource, resolveParakeetPresetManifest } from './manifest.js';

export interface CreateParakeetPresetFactoryOptions {
  readonly dependencies?: CreateNemoTdtModelFamilyOptions['dependencies'];
  readonly useManifestSource?: boolean;
}

export function createParakeetPresetFactory(
  options: CreateParakeetPresetFactoryOptions = {},
): SpeechPresetFactory<
  NemoTdtModelOptions | NemoRnntModelOptions,
  NemoTdtModelOptions | NemoRnntModelOptions
> {
  return {
    preset: 'parakeet',
    supports(modelId?: string): boolean {
      return modelId ? resolveParakeetPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<NemoTdtModelOptions | NemoRnntModelOptions>> {
      const modelId = request.modelId ?? DEFAULT_MODEL;
      const manifest = resolveParakeetPresetManifest(modelId);
      const manifestSource = options.useManifestSource
        ? resolveParakeetArtifactSource(modelId)
        : undefined;
      const family = manifest?.classification.topology === 'rnnt' ? 'nemo-rnnt' : 'nemo-tdt';
      const resolvedOptions = {
        ...request.options,
        config: {
          ...(manifest?.config ?? {}),
          ...(request.options?.config ?? {}),
        },
        source: request.options?.source ?? manifestSource,
      } as NemoTdtModelOptions | NemoRnntModelOptions;

      return {
        family,
        modelId,
        classification: {
          family: 'parakeet',
          ...request.classification,
        },
        resolvedPreset: 'parakeet',
        options: resolvedOptions,
      };
    },
  };
}

export const createParakeetPresetFamily = createParakeetPresetFactory;
