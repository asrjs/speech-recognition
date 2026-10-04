import type { SpeechPresetFactory } from '../types/index.js';
import type { Qwen3AsrLoadOptions } from '../models/qwen3-asr/types.js';

/**
 * Resolves `qwen3-asr` requests.
 *
 * The graphs are our own fp32 export from the validated checkpoint, so this
 * preset has no Hub source and requires a local artifact directory.
 */
export function createQwen3AsrPresetFactory(): SpeechPresetFactory<
  Qwen3AsrLoadOptions,
  Qwen3AsrLoadOptions
> {
  return {
    preset: 'qwen3-asr',
    supports(modelId?: string): boolean {
      return modelId === undefined || modelId.toLowerCase().includes('qwen3-asr');
    },
    async resolveModelRequest(request) {
      if (!request.options?.frontUrl) {
        throw new Error(
          'qwen3-asr needs a local artifact directory; no Hub export exists for our fp32 graphs.',
        );
      }
      return {
        family: 'qwen3-asr',
        modelId: request.modelId ?? 'Qwen/Qwen3-ASR-0.6B-hf',
        classification: { family: 'qwen3-asr', ...request.classification },
        resolvedPreset: 'qwen3-asr',
        options: request.options,
      };
    },
  };
}
