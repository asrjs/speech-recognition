import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateWhisperSeq2SeqModelFamilyOptions,
  WhisperSeq2SeqModelOptions,
} from '../../models/whisper-seq2seq/index.js';
import { resolveWhisperArtifactSource, resolveWhisperPresetManifest } from './manifest.js';

export interface CreateWhisperPresetFactoryOptions {
  readonly dependencies?: CreateWhisperSeq2SeqModelFamilyOptions['dependencies'];
  readonly useManifestSource?: boolean;
}

export function createWhisperPresetFactory(
  options: CreateWhisperPresetFactoryOptions = {},
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
      const modelId = request.modelId ?? 'onnx-community/whisper-base';
      const manifest = resolveWhisperPresetManifest(modelId);
      const manifestSource = options.useManifestSource
        ? resolveWhisperArtifactSource(modelId)
        : undefined;

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
          source: request.options?.source ?? manifestSource,
        },
      };
    },
  };
}
