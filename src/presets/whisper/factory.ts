import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateWhisperSeq2SeqModelFamilyOptions,
  WhisperSeq2SeqModelOptions,
} from '../../models/whisper-seq2seq/index.js';
import { resolveWhisperPresetManifest } from './manifest.js';

export interface CreateWhisperPresetFactoryOptions {
  readonly dependencies?: CreateWhisperSeq2SeqModelFamilyOptions['dependencies'];
}

export function createWhisperPresetFactory(
  _options: CreateWhisperPresetFactoryOptions = {},
): SpeechPresetFactory<WhisperSeq2SeqModelOptions, WhisperSeq2SeqModelOptions> {
  return {
    preset: 'whisper',
    supports(modelId?: string): boolean {
      return modelId ? resolveWhisperPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<WhisperSeq2SeqModelOptions>> {
      const modelId = request.modelId ?? 'openai/whisper-base';
      const manifest = resolveWhisperPresetManifest(modelId);

      return {
        family: 'whisper-seq2seq',
        modelId,
        classification: {
          family: 'whisper',
          ...request.classification,
        },
        resolvedPreset: 'whisper',
        options: {
          ...request.options,
          config: {
            ...manifest?.config,
            ...request.options?.config,
          },
        },
      };
    },
  };
}
