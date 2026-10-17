import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type { ParakeetCtcModelOptions } from '../../models/parakeet-ctc/types.js';
import { DEFAULT_PARAKEET_CTC_CONFIG } from '../../models/parakeet-ctc/model.js';

export interface CreateParakeetCtcPresetFactoryOptions {
  readonly defaultModelId?: string;
}

/**
 * Resolves `parakeet-ctc` requests onto the local ParakeetForCTC family.
 *
 * The descriptor is local-folder only, so a request without an explicit source
 * fails closed rather than inventing a Hub repository that publishes no ONNX.
 */
export function createParakeetCtcPresetFactory(
  options: CreateParakeetCtcPresetFactoryOptions = {},
): SpeechPresetFactory<ParakeetCtcModelOptions, ParakeetCtcModelOptions> {
  return {
    preset: 'parakeet-ctc',
    supports(modelId?: string): boolean {
      if (!modelId) return false;
      return modelId.includes('parakeet-ctc') || modelId.includes('ctc-110m');
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<ParakeetCtcModelOptions>> {
      if (!request.options?.source) {
        throw new Error(
          'parakeet-ctc has no validated Hub export; supply a local artifact source.',
        );
      }

      return {
        family: 'parakeet-ctc',
        modelId: request.modelId ?? options.defaultModelId ?? 'parakeet-ctc-110m-en',
        classification: { family: 'parakeet-ctc', ...request.classification },
        resolvedPreset: 'parakeet-ctc',
        options: {
          config: {
            ...DEFAULT_PARAKEET_CTC_CONFIG,
            ...request.options?.config,
          },
          ...request.options,
        } as ParakeetCtcModelOptions,
      };
    },
  };
}
