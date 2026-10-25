import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { GigaAmModelOptions } from '../models/gigaam-ctc/types.js';

export function createGigaAmCtcPresetFactory(): SpeechPresetFactory<GigaAmModelOptions, GigaAmModelOptions> {
  return {
    preset: 'gigaam-ctc',
    supports(modelId?: string) { return modelId === 'gigaam-multilingual-ctc'; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<GigaAmModelOptions>> {
      if (request.options?.source?.kind !== 'direct') throw new Error('GigaAM CTC requires local official ONNX artifacts.');
      return { family: 'gigaam-ctc', modelId: request.modelId ?? 'gigaam-multilingual-ctc',
        classification: { family: 'gigaam-ctc', ...request.classification },
        resolvedPreset: 'gigaam-ctc', options: request.options };
    },
  };
}
