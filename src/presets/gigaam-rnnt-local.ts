import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { GigaAmRnntModelOptions } from '../models/gigaam-rnnt/types.js';
import { GIGAAM_RNNT_PUBLISHED_BUNDLE } from './published-bundles.js';

export function createGigaAmRnntPresetFactory(): SpeechPresetFactory<GigaAmRnntModelOptions, GigaAmRnntModelOptions> {
  return {
    preset: 'gigaam-rnnt',
    supports(modelId?: string) { return modelId === 'gigaam-v3-e2e-rnnt'; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<GigaAmRnntModelOptions>> {
      const source = request.options?.source ?? {
        kind: 'huggingface' as const,
        repoId: GIGAAM_RNNT_PUBLISHED_BUNDLE.repoId,
        revision: GIGAAM_RNNT_PUBLISHED_BUNDLE.revision,
      };
      return { family: 'gigaam-rnnt', modelId: request.modelId ?? 'gigaam-v3-e2e-rnnt',
        classification: { family: 'gigaam-rnnt', ...request.classification },
        resolvedPreset: 'gigaam-rnnt', options: { ...request.options, source } };
    },
  };
}
