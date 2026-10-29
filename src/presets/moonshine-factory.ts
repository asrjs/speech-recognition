import type { SpeechPresetFactory } from '../types/index.js';
import type { MoonshineLoadOptions } from '../models/moonshine/types.js';
import {
  MOONSHINE_STREAMING_TINY_PUBLISHED_BUNDLE,
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
            : undefined;
      if (!options) {
        throw new Error(
          factoryOptions.useManifestSource === false
            ? 'moonshine requires explicit local artifacts when manifest sources are disabled.'
            : 'moonshine needs local artifacts; only the registered streaming Tiny bundle has a Hub source.',
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
