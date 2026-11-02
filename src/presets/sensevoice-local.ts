import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { SenseVoiceModelOptions } from '../models/sensevoice/types.js';
import { SENSEVOICE_PUBLISHED_BUNDLE } from './published-bundles.js';

export function createSenseVoicePresetFactory(): SpeechPresetFactory<SenseVoiceModelOptions, SenseVoiceModelOptions> {
  return {
    preset: 'sensevoice',
    supports(modelId?: string) { return modelId === 'FunAudioLLM/SenseVoiceSmall'; },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<SenseVoiceModelOptions>> {
      const source = request.options?.source ?? {
        kind: 'huggingface' as const,
        repoId: SENSEVOICE_PUBLISHED_BUNDLE.repoId,
        revision: SENSEVOICE_PUBLISHED_BUNDLE.revision,
        cmvnFilename: SENSEVOICE_PUBLISHED_BUNDLE.files.cmvn,
      };
      return { family: 'sensevoice', modelId: request.modelId ?? 'FunAudioLLM/SenseVoiceSmall',
        classification: { family: 'sensevoice', ...request.classification },
        resolvedPreset: 'sensevoice', options: { ...request.options, source } };
    },
  };
}
