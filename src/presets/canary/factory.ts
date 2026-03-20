import type { FamilyModelLoadRequest, SpeechPresetFactory } from '../../types/index.js';
import type {
  CreateNemoAedModelFamilyOptions,
  NemoAedModelOptions,
} from '../../models/nemo-aed/index.js';
import {
  DEFAULT_MODEL,
  resolveCanaryArtifactSource,
  resolveCanaryPresetManifest,
} from './manifest.js';

export interface CreateCanaryPresetFactoryOptions {
  readonly dependencies?: CreateNemoAedModelFamilyOptions['dependencies'];
  readonly useManifestSource?: boolean;
}

/**
 * Create the preset factory for Canary-style NeMo AED models.
 *
 * Upstream model-card behavior:
 * - plain audio input defaults to English ASR
 * - language/task/timestamp/PnC behavior is controlled by prompt tokens
 * - NeMo itself performs frontend preprocessing internally
 *
 * Runtime note:
 * This repo still resolves a frontend artifact separately because it runs the
 * encoder and decoder outside NeMo. The manifest therefore keeps the
 * encoder/decoder contract plus the currently selected frontend source.
 */
export function createCanaryPresetFactory(
  options: CreateCanaryPresetFactoryOptions = {},
): SpeechPresetFactory<NemoAedModelOptions, NemoAedModelOptions> {
  return {
    preset: 'canary',
    supports(modelId?: string): boolean {
      return modelId ? resolveCanaryPresetManifest(modelId) !== undefined : true;
    },
    async resolveModelRequest(
      request,
      _context,
    ): Promise<FamilyModelLoadRequest<NemoAedModelOptions>> {
      const modelId = request.modelId ?? DEFAULT_MODEL;
      const manifest = resolveCanaryPresetManifest(modelId);
      const manifestSource = options.useManifestSource
        ? resolveCanaryArtifactSource(modelId)
        : undefined;

      return {
        family: 'nemo-aed',
        modelId,
        classification: {
          family: 'canary',
          ...request.classification,
        },
        resolvedPreset: 'canary',
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

export const createCanaryPresetFamily = createCanaryPresetFactory;
