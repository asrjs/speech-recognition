import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { SenseVoiceModelOptions } from '../models/sensevoice/types.js';

export function createSenseVoicePresetFactory(): SpeechPresetFactory<SenseVoiceModelOptions, SenseVoiceModelOptions> {
  return {
    preset: 'sensevoice',
    supports(modelId?: string) { return modelId === 'FunAudioLLM/SenseVoiceSmall'; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<SenseVoiceModelOptions>> {
      if (request.options?.source?.kind !== 'direct') throw new Error('SenseVoice Small requires local official ONNX artifacts.');
      return { family: 'sensevoice', modelId: request.modelId ?? 'FunAudioLLM/SenseVoiceSmall',
        classification: { family: 'sensevoice', ...request.classification },
        resolvedPreset: 'sensevoice', options: request.options };
    },
  };
}
