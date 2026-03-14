import type { ModelClassification, SpeechModelFactory } from '../../types/index.js';
import {
  createHfCtcModelFamily,
  describeHfCtcModel,
  parseHfCtcConfig,
  type CreateHfCtcModelFamilyOptions,
  type HfCtcModelOptions,
  type HfCtcNativeTranscript,
  type HfCtcTranscriptionOptions
} from '../../models/hf-ctc-common/index.js';
import { resolveMedAsrPresetManifest } from './manifest.js';

export interface CreateMedAsrPresetFactoryOptions {
  readonly dependencies?: CreateHfCtcModelFamilyOptions['dependencies'];
}

function medAsrClassificationFallback(classification: Partial<ModelClassification> = {}): boolean {
  return classification.family === 'medasr'
    || (
      classification.ecosystem === 'hf'
      && classification.decoder === 'ctc'
      && classification.family === 'medasr'
    );
}

export function createMedAsrPresetFactory(
  options: CreateMedAsrPresetFactoryOptions = {}
): SpeechModelFactory<HfCtcModelOptions, HfCtcTranscriptionOptions, HfCtcNativeTranscript> {
  return createHfCtcModelFamily({
    family: 'medasr',
    classification: {
      family: 'medasr'
    },
    dependencies: options.dependencies,
    supportsModel(modelId, classification) {
      return resolveMedAsrPresetManifest(modelId) !== undefined || medAsrClassificationFallback(classification);
    },
    resolveConfig(modelId, request) {
      const manifest = resolveMedAsrPresetManifest(modelId);
      return parseHfCtcConfig(modelId, {
        ...manifest?.config,
        ...request.options?.config
      });
    },
    describeModel(modelId, classification, config) {
      const manifest = resolveMedAsrPresetManifest(modelId);
      return manifest?.description ?? describeHfCtcModel(modelId, classification, config);
    }
  });
}
