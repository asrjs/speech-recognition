import type { ModelClassification, SpeechModelFactory } from '../../types/index.js';
import {
  createWhisperSeq2SeqModelFamily,
  describeWhisperSeq2SeqModel,
  parseWhisperSeq2SeqConfig,
  type CreateWhisperSeq2SeqModelFamilyOptions,
  type WhisperNativeTranscript,
  type WhisperSeq2SeqModelOptions,
  type WhisperSeq2SeqTranscriptionOptions
} from '../../models/whisper-seq2seq/index.js';
import { resolveWhisperPresetManifest } from './manifest.js';

export interface CreateWhisperPresetFactoryOptions {
  readonly dependencies?: CreateWhisperSeq2SeqModelFamilyOptions['dependencies'];
}

function whisperClassificationFallback(classification: Partial<ModelClassification> = {}): boolean {
  return classification.family === 'whisper'
    || (
      classification.ecosystem === 'openai'
      && classification.topology === 'aed'
      && classification.family === 'whisper'
    );
}

export function createWhisperPresetFactory(
  options: CreateWhisperPresetFactoryOptions = {}
): SpeechModelFactory<WhisperSeq2SeqModelOptions, WhisperSeq2SeqTranscriptionOptions, WhisperNativeTranscript> {
  return createWhisperSeq2SeqModelFamily({
    family: 'whisper',
    classification: {
      family: 'whisper'
    },
    dependencies: options.dependencies,
    supportsModel(modelId, classification) {
      return resolveWhisperPresetManifest(modelId) !== undefined || whisperClassificationFallback(classification);
    },
    resolveConfig(modelId, request) {
      const manifest = resolveWhisperPresetManifest(modelId);
      return parseWhisperSeq2SeqConfig(modelId, {
        ...manifest?.config,
        ...request.options?.config
      });
    },
    describeModel(modelId, classification, config) {
      const manifest = resolveWhisperPresetManifest(modelId);
      return manifest?.description ?? describeWhisperSeq2SeqModel(modelId, classification, config);
    }
  });
}
