import type { SpeechPresetFactory } from '../types/index.js';
import type { MoonshineLoadOptions } from '../models/moonshine/types.js';
import {
  MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
  MOONSHINE_TINY_PUBLISHED_BUNDLE,
  publishedBundleFileUrl,
} from './published-bundles.js';

/**
 * Resolves Moonshine requests. The registered streaming bundle is experimental;
 * its Hub source enables loading but does not imply broad runtime validation.
 */
export function createMoonshinePresetFactory(
  factoryOptions: { readonly useManifestSource?: boolean } = {},
): SpeechPresetFactory<MoonshineLoadOptions, MoonshineLoadOptions> {
  return {
    preset: 'moonshine',
    supports(modelId?: string): boolean {
      if (!modelId) return false;
      return modelId.toLowerCase().includes('moonshine');
    },
    async resolveModelRequest(request) {
      const modelId = request.modelId ?? 'moonshine-tiny';
      const streamingModel = [
        'moonshine-ai/moonshine-streaming-tiny',
        'moonshine-streaming-tiny',
      ].includes(modelId.toLowerCase());
      const offlineModel = [
        'moonshine-tiny',
        'usefulsensors/moonshine-tiny',
        'moonshine-ai/moonshine-tiny',
      ].includes(modelId.toLowerCase());
      const options =
        request.options?.encoderUrl || request.options?.streaming
          ? request.options
          : streamingModel && factoryOptions.useManifestSource !== false
            ? {
                ...request.options,
                encoderUrl: publishedBundleFileUrl(
                  MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                  'encoder',
                ),
                tokenizerUrl: publishedBundleFileUrl(
                  MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                  'tokenizer',
                ),
                generationUrl: publishedBundleFileUrl(
                  MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                  'generation',
                ),
                encoderBackend: request.options?.encoderBackend,
                cpuThreads: request.options?.cpuThreads,
                wasmPaths: request.options?.wasmPaths,
                maxTokens: request.options?.maxTokens,
                sampleRate: request.options?.sampleRate,
                streaming: {
                  encoderUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'encoder',
                  ),
                  encoderDataUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'encoderData',
                  ),
                  chunkedEncoderUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'chunkedEncoder',
                  ),
                  chunkedEncoderDataUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'chunkedEncoderData',
                  ),
                  prefillUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'decoderPrefill',
                  ),
                  prefillDataUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'decoderPrefillData',
                  ),
                  stepUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'decoderStep',
                  ),
                  stepDataUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'decoderStepData',
                  ),
                  tokenizerUrl: publishedBundleFileUrl(
                    MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
                    'tokenizer',
                  ),
                },
              }
            : offlineModel && factoryOptions.useManifestSource !== false
              ? {
                  ...request.options,
                  encoderUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'encoder'),
                  encoderDataUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'encoderData'),
                  crossKvUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'crossKv'),
                  crossKvDataUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'crossKvData'),
                  decoderUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'decoder'),
                  decoderDataUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'decoderData'),
                  tokenizerUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'tokenizer'),
                  generationUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'generation'),
                  receiptUrl: publishedBundleFileUrl(MOONSHINE_TINY_PUBLISHED_BUNDLE, 'receipt'),
                }
              : undefined;
      if (!options) {
        throw new Error(
          factoryOptions.useManifestSource === false
            ? 'moonshine requires explicit local artifacts when manifest sources are disabled.'
            : 'moonshine requires explicit artifacts for models without a registered Hub bundle.',
        );
      }
      return {
        family: 'moonshine',
        modelId,
        classification: { family: 'moonshine', ...request.classification },
        resolvedPreset: 'moonshine',
        options,
      };
    },
  };
}
