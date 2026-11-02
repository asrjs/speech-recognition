import type { SpeechPresetFactory } from '../types/index.js';
import type { OmnilingualCtcLoadOptions } from '../models/omnilingual-ctc/factory.js';
import { OMNILINGUAL_CTC_PUBLISHED_BUNDLE, publishedBundleFileUrl } from './published-bundles.js';

/** Resolves Omnilingual CTC requests with the pinned export or explicit local artifacts. */
export function createOmnilingualCtcPresetFactory(): SpeechPresetFactory<
  OmnilingualCtcLoadOptions,
  OmnilingualCtcLoadOptions
> {
  return {
    preset: 'omnilingual-ctc',
    supports(modelId?: string): boolean {
      if (!modelId) return false;
      return modelId.toLowerCase().includes('omnilingual');
    },
    async resolveModelRequest(request) {
      const options = request.options?.modelUrl ? request.options : {
        ...request.options,
        modelUrl: publishedBundleFileUrl(OMNILINGUAL_CTC_PUBLISHED_BUNDLE, 'model'),
        tokensUrl: publishedBundleFileUrl(OMNILINGUAL_CTC_PUBLISHED_BUNDLE, 'tokenizer'),
      };
      return {
        family: 'omnilingual-ctc',
        modelId: request.modelId ?? 'omnilingual-ctc-300m',
        classification: { family: 'omnilingual-ctc', ...request.classification },
        resolvedPreset: 'omnilingual-ctc',
        options,
      };
    },
  };
}
