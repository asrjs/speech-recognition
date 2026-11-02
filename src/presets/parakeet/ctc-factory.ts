import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type { ParakeetCtcModelOptions } from '../../models/parakeet-ctc/types.js';
import { DEFAULT_PARAKEET_CTC_CONFIG } from '../../models/parakeet-ctc/model.js';
import { PARAKEET_CTC_PUBLISHED_BUNDLE, publishedBundleFileUrl } from '../published-bundles.js';

export interface CreateParakeetCtcPresetFactoryOptions {
  readonly defaultModelId?: string;
}

/** Resolves ParakeetForCTC requests with the pinned export or explicit local artifacts. */
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
      const source = request.options?.source ?? {
        kind: 'direct' as const,
        artifacts: {
          encoderUrl: publishedBundleFileUrl(PARAKEET_CTC_PUBLISHED_BUNDLE, 'model'),
          tokenizerUrl: publishedBundleFileUrl(PARAKEET_CTC_PUBLISHED_BUNDLE, 'tokenizer'),
        },
      };

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
          source,
        } as ParakeetCtcModelOptions,
      };
    },
  };
}
