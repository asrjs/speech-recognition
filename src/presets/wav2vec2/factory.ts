import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateWav2Vec2ModelFamilyOptions,
  Wav2Vec2ModelOptions,
} from '../../models/wav2vec2/index.js';
import { resolveWav2Vec2PresetManifest } from './manifest.js';

export interface CreateWav2Vec2PresetFactoryOptions {
  readonly dependencies?: CreateWav2Vec2ModelFamilyOptions['dependencies'];
  /**
   * Keep false until a published asrjs Wav2Vec2 ONNX repo exists. Local smoke
   * tests should pass a direct source explicitly.
   */
  readonly useManifestSource?: boolean;
}

export function createWav2Vec2PresetFactory(
  options: CreateWav2Vec2PresetFactoryOptions = {},
): SpeechPresetFactory<Wav2Vec2ModelOptions, Wav2Vec2ModelOptions> {
  void options.dependencies;
  const useManifestSource = options.useManifestSource ?? false;

  return {
    preset: 'wav2vec2',
    supports(modelId?: string): boolean {
      return modelId ? resolveWav2Vec2PresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<Wav2Vec2ModelOptions>> {
      const modelId = request.modelId ?? 'facebook/wav2vec2-base-960h';
      const manifest = resolveWav2Vec2PresetManifest(modelId);

      if (!manifest) {
        throw new Error(`Unknown Wav2Vec2 preset model "${modelId}".`);
      }

      return {
        family: 'wav2vec2',
        modelId: manifest.modelId,
        classification: {
          ...manifest.classification,
          ...request.classification,
        },
        resolvedPreset: 'wav2vec2',
        options: {
          ...request.options,
          config: {
            ...manifest.config,
            ...request.options?.config,
          },
          source: request.options?.source ?? (useManifestSource ? manifest.source : undefined),
        },
      };
    },
  };
}
