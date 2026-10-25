import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { GigaAmRnntModelOptions } from '../models/gigaam-rnnt/types.js';

export function createGigaAmRnntPresetFactory(): SpeechPresetFactory<GigaAmRnntModelOptions, GigaAmRnntModelOptions> {
  return {
    preset: 'gigaam-rnnt',
    supports(modelId?: string) { return modelId === 'gigaam-v3-e2e-rnnt'; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<GigaAmRnntModelOptions>> {
      if (request.options?.source?.kind !== 'direct') throw new Error('GigaAM v3 RNN-T requires local official ONNX artifacts.');
      return { family: 'gigaam-rnnt', modelId: request.modelId ?? 'gigaam-v3-e2e-rnnt',
        classification: { family: 'gigaam-rnnt', ...request.classification },
        resolvedPreset: 'gigaam-rnnt', options: request.options };
    },
  };
}
