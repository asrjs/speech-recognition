import type { SpeechPresetFactory } from '../types/index.js';
import type { MoonshineLoadOptions } from '../models/moonshine/types.js';

/**
 * Resolves `moonshine` requests. There is no validated Hub export for the browser
 * pair we validated, so a local artifact source is required.
 */
export function createMoonshinePresetFactory(): SpeechPresetFactory<
  MoonshineLoadOptions,
  MoonshineLoadOptions
> {
  return {
    preset: 'moonshine',
    supports(modelId?: string): boolean {
      return modelId === undefined || modelId.toLowerCase().includes('moonshine');
    },
    async resolveModelRequest(request) {
      if (!request.options?.encoderUrl) {
        throw new Error('moonshine needs a local artifact source; no validated Hub export exists.');
      }
      return {
        family: 'moonshine',
        modelId: request.modelId ?? 'moonshine-tiny',
        classification: { family: 'moonshine', ...request.classification },
        resolvedPreset: 'moonshine',
        options: request.options,
      };
    },
  };
}
