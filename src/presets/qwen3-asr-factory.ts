import type { SpeechPresetFactory } from '../types/index.js';
import type { Qwen3AsrModelOptions } from '../models/qwen-asr/types.js';

/**
 * Resolves `qwen3-asr` requests.
 *
 * The graphs are our own fp32 export from the validated checkpoint, so this
 * preset has no Hub source and requires a local artifact directory.
 */
export function createQwen3AsrPresetFactory(): SpeechPresetFactory<
  Qwen3AsrModelOptions,
  Qwen3AsrModelOptions
> {
  return {
    preset: 'qwen3-asr',
    supports(modelId?: string): boolean {
      if (!modelId) return false;
      return modelId.toLowerCase().includes('qwen3-asr');
    },
    async resolveModelRequest(request) {
      if (request.options?.source?.kind !== 'direct') {
        throw new Error(
          'qwen3-asr requires the official local ONNX graph set through source.kind="direct"; legacy frontUrl/decoderUrl folders are incompatible with the arbitrary-audio executor.',
        );
      }
      return {
        family: 'qwen-asr',
        modelId: request.modelId ?? 'Qwen/Qwen3-ASR-0.6B-hf',
        classification: { family: 'qwen3-asr', ...request.classification },
        resolvedPreset: 'qwen3-asr',
        options: request.options,
      };
    },
  };
}
