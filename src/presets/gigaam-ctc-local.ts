import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { GigaAmModelOptions } from '../models/gigaam-ctc/types.js';
import { GIGAAM_CTC_PUBLISHED_BUNDLE } from './published-bundles.js';

export function createGigaAmCtcPresetFactory(): SpeechPresetFactory<GigaAmModelOptions, GigaAmModelOptions> {
  return {
    preset: 'gigaam-ctc',
    supports(modelId?: string) { return modelId === 'gigaam-multilingual-ctc'; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<GigaAmModelOptions>> {
      const source = request.options?.source ?? {
        kind: 'huggingface' as const,
        repoId: GIGAAM_CTC_PUBLISHED_BUNDLE.repoId,
        revision: GIGAAM_CTC_PUBLISHED_BUNDLE.revision,
      };
      return { family: 'gigaam-ctc', modelId: request.modelId ?? 'gigaam-multilingual-ctc',
        classification: { family: 'gigaam-ctc', ...request.classification },
        resolvedPreset: 'gigaam-ctc', options: { ...request.options, source } };
    },
  };
}
