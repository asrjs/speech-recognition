import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { XAsrModelOptions } from '../models/x-asr/types.js';
import { XASR_PUBLISHED_BUNDLE } from './published-bundles.js';

const MODEL_ID = 'x-asr-zh-en-160ms';

export function createXAsrPresetFactory(): SpeechPresetFactory<XAsrModelOptions, XAsrModelOptions> {
  return {
    preset: 'x-asr',
    supports(modelId?: string) { return modelId === MODEL_ID; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<XAsrModelOptions>> {
      const source = request.options?.source ?? {
        kind: 'huggingface' as const,
        repoId: XASR_PUBLISHED_BUNDLE.repoId,
        revision: XASR_PUBLISHED_BUNDLE.revision,
      };
      return {
        family: 'x-asr',
        modelId: request.modelId ?? MODEL_ID,
        classification: { family: 'x-asr', ...request.classification },
        resolvedPreset: 'x-asr',
        options: {
          ...request.options,
          source,
          config: { ...request.options?.config, chunkMs: 160 },
        },
      };
    },
  };
}
