import type { SpeechPresetFactory } from '../types/index.js';
import type { OmnilingualCtcLoadOptions } from '../models/omnilingual-ctc/factory.js';

/** Resolves `omnilingual-ctc` requests; a local artifact source is required. */
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
      if (!request.options?.modelUrl) {
        throw new Error('omnilingual-ctc needs a local artifact source; no Hub export exists.');
      }
      return {
        family: 'omnilingual-ctc',
        modelId: request.modelId ?? 'omnilingual-ctc-300m',
        classification: { family: 'omnilingual-ctc', ...request.classification },
        resolvedPreset: 'omnilingual-ctc',
        options: request.options,
      };
    },
  };
}
