import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../types/index.js';
import type { Wav2Vec2ModelOptions } from '../models/wav2vec2/index.js';
import { WAV2VEC2_TURKISH_PUBLISHED_BUNDLE } from './published-bundles.js';

/**
 * Resolves the Turkish XLS-R fine-tune onto the existing wav2vec2 family.
 *
 * XLS-R differs from base-960h in ways that break inference if copied, so the
 * audited deltas (doStableLayerNorm, featExtractNorm layer, conv/hidden 1024,
 * 24 layers, 38-token character vocabulary) are declared here.
 */
export function createWav2Vec2TurkishPresetFactory(): SpeechPresetFactory<
  Wav2Vec2ModelOptions,
  Wav2Vec2ModelOptions
> {
  return {
    preset: 'wav2vec2-tr',
    supports(modelId?: string): boolean {
      if (!modelId) return false;
      const lower = modelId.toLowerCase();
      return lower === 'wav2vec2-tr' || lower === 'wav2vec2-large-xlsr-turkish' ||
        lower === 'mpoyraz/wav2vec2-xls-r-300m-cv7-turkish' ||
        lower === 'wav2vec2-xls-r-300m-cv7-turkish' || lower === 'wav2vec2-turkish';
    },
    async resolveModelRequest(request): Promise<FamilyModelLoadRequest<Wav2Vec2ModelOptions>> {
      const source = request.options?.source ?? {
        kind: 'huggingface' as const,
        repoId: WAV2VEC2_TURKISH_PUBLISHED_BUNDLE.repoId,
        revision: WAV2VEC2_TURKISH_PUBLISHED_BUNDLE.revision,
      };
      return {
        family: 'wav2vec2',
        modelId: request.modelId ?? 'wav2vec2-xls-r-300m-cv7-turkish',
        classification: { family: 'wav2vec2', ...request.classification },
        resolvedPreset: 'wav2vec2-tr',
        options: {
          ...request.options,
          source,
          config: {
            doStableLayerNorm: true,
            featExtractNorm: 'layer',
            convDim: 1024,
            hiddenSize: 1024,
            numHiddenLayers: 24,
            vocabularySize: 38,
            ...request.options?.config,
          },
        },
      };
    },
  };
}
