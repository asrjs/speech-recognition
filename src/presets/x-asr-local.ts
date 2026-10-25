import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { XAsrModelOptions } from '../models/x-asr/types.js';

const MODEL_ID = 'x-asr-zh-en-160ms';

export function createXAsrPresetFactory(): SpeechPresetFactory<XAsrModelOptions, XAsrModelOptions> {
  return {
    preset: 'x-asr',
    supports(modelId?: string) { return modelId === MODEL_ID; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<XAsrModelOptions>> {
      if (request.options?.source?.kind !== 'direct') {
        throw new Error('X-ASR 160 ms requires local official ONNX artifacts via a direct artifact source.');
      }
      return {
        family: 'x-asr',
        modelId: request.modelId ?? MODEL_ID,
        classification: { family: 'x-asr', ...request.classification },
        resolvedPreset: 'x-asr',
        options: {
          ...request.options,
          config: { ...request.options.config, chunkMs: 160 },
        },
      };
    },
  };
}
