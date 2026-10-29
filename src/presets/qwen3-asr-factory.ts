import type { SpeechPresetFactory } from '../types/index.js';
import type { Qwen3AsrModelOptions } from '../models/qwen-asr/types.js';
import { publishedBundleFileUrl, QWEN3_ASR_PUBLISHED_BUNDLE } from './published-bundles.js';

/**
 * Resolves `qwen3-asr` requests.
 *
 * Routes the published ONNX bundle through the same direct graph contract used
 * by the local adapter. The bundle remains experimental pending full matrix
 * verification.
 */
export function createQwen3AsrPresetFactory(
  factoryOptions: { readonly useManifestSource?: boolean } = {},
): SpeechPresetFactory<Qwen3AsrModelOptions, Qwen3AsrModelOptions> {
  return {
    preset: 'qwen3-asr',
    supports(modelId?: string): boolean {
      if (!modelId) return false;
      return modelId.toLowerCase().includes('qwen3-asr');
    },
    async resolveModelRequest(request) {
      if (request.options && ('frontUrl' in request.options || 'decoderUrl' in request.options)) {
        throw new Error(
          'qwen3-asr requires source.kind="direct" for custom graphs; legacy frontUrl/decoderUrl folders are incompatible with the arbitrary-audio executor.',
        );
      }
      const source = request.options?.source;
      if (source?.kind === 'direct') {
        return {
          family: 'qwen-asr',
          modelId: request.modelId ?? 'Qwen/Qwen3-ASR-0.6B-hf',
          classification: { family: 'qwen3-asr', ...request.classification },
          resolvedPreset: 'qwen3-asr',
          options: request.options,
        };
      }
      if (source?.kind === 'huggingface' && source.repoId !== QWEN3_ASR_PUBLISHED_BUNDLE.repoId) {
        throw new Error(
          'qwen3-asr only supports its registered ONNX Hub bundle; legacy frontUrl/decoderUrl folders are incompatible with the arbitrary-audio executor.',
        );
      }
      if (
        source?.kind === 'huggingface' &&
        source.revision &&
        source.revision !== QWEN3_ASR_PUBLISHED_BUNDLE.revision
      ) {
        throw new Error(
          `Unsupported Qwen bundle revision "${source.revision}"; select the registered pinned revision.`,
        );
      }
      const baseSource = source?.kind === 'huggingface' ? source : undefined;
      if (baseSource) {
        if (baseSource.encoderQuant && baseSource.encoderQuant !== 'fp32') {
          throw new Error(
            `Unsupported Qwen encoder precision "${baseSource.encoderQuant}"; the published bundle is fp32.`,
          );
        }
        if (baseSource.decoderQuant && baseSource.decoderQuant !== 'fp16') {
          throw new Error(
            `Unsupported Qwen decoder precision "${baseSource.decoderQuant}"; the published bundle is fp16.`,
          );
        }
        if (
          baseSource.encoderPath &&
          baseSource.encoderPath !== QWEN3_ASR_PUBLISHED_BUNDLE.files.encoder
        ) {
          throw new Error(
            'The registered Qwen Hub bundle does not contain the requested encoder graph.',
          );
        }
        if (
          baseSource.decoderPath &&
          baseSource.decoderPath !== QWEN3_ASR_PUBLISHED_BUNDLE.files.decoderPrefill
        ) {
          throw new Error(
            'The registered Qwen Hub bundle uses its paired prefill and step decoder graphs.',
          );
        }
        if (
          baseSource.tokenizerPath &&
          baseSource.tokenizerPath !== QWEN3_ASR_PUBLISHED_BUNDLE.files.tokenizer
        ) {
          throw new Error(
            'The registered Qwen Hub bundle does not contain the requested tokenizer path.',
          );
        }
        if (baseSource.encoderDataPath) {
          throw new Error('The registered Qwen encoder has no external-data sidecar.');
        }
        if (
          baseSource.decoderDataPath &&
          baseSource.decoderDataPath !== QWEN3_ASR_PUBLISHED_BUNDLE.files.decoderData
        ) {
          throw new Error(
            'The registered Qwen Hub bundle does not contain the requested decoder sidecar.',
          );
        }
      }
      if (!source && factoryOptions.useManifestSource === false) {
        throw new Error(
          'qwen3-asr requires an explicit direct graph source when manifest sources are disabled.',
        );
      }
      const options: Qwen3AsrModelOptions = {
        ...request.options,
        source: {
          kind: 'direct',
          artifacts: {
            encoderUrl: publishedBundleFileUrl(QWEN3_ASR_PUBLISHED_BUNDLE, 'encoder'),
            decoderUrl: publishedBundleFileUrl(QWEN3_ASR_PUBLISHED_BUNDLE, 'decoderPrefill'),
            decoderStepUrl: publishedBundleFileUrl(QWEN3_ASR_PUBLISHED_BUNDLE, 'decoderStep'),
            tokenizerUrl: publishedBundleFileUrl(QWEN3_ASR_PUBLISHED_BUNDLE, 'tokenizer'),
            decoderPrefillDataUrl: publishedBundleFileUrl(
              QWEN3_ASR_PUBLISHED_BUNDLE,
              'decoderData',
            ),
            decoderStepDataUrl: publishedBundleFileUrl(QWEN3_ASR_PUBLISHED_BUNDLE, 'decoderData'),
            decoderPrefillDataPath: QWEN3_ASR_PUBLISHED_BUNDLE.files.decoderData,
            decoderStepDataPath: QWEN3_ASR_PUBLISHED_BUNDLE.files.decoderData,
          },
          ...(baseSource?.encoderBackend ? { encoderBackend: baseSource.encoderBackend } : {}),
          ...(baseSource?.decoderBackend ? { decoderBackend: baseSource.decoderBackend } : {}),
          ...(baseSource?.cacheOutputLocation
            ? { cacheOutputLocation: baseSource.cacheOutputLocation }
            : {}),
          ...(baseSource?.wasmPaths ? { wasmPaths: baseSource.wasmPaths } : {}),
          ...(baseSource?.cpuThreads !== undefined ? { cpuThreads: baseSource.cpuThreads } : {}),
          ...(baseSource?.enableProfiling !== undefined
            ? { enableProfiling: baseSource.enableProfiling }
            : {}),
          ...(baseSource?.decoderGraphCapture !== undefined
            ? { decoderGraphCapture: baseSource.decoderGraphCapture }
            : {}),
          ...(baseSource?.decoderFreeDimensionOverrides
            ? { decoderFreeDimensionOverrides: baseSource.decoderFreeDimensionOverrides }
            : {}),
        },
      };
      return {
        family: 'qwen-asr',
        modelId: request.modelId ?? 'Qwen/Qwen3-ASR-0.6B-hf',
        classification: { family: 'qwen3-asr', ...request.classification },
        resolvedPreset: 'qwen3-asr',
        options,
      };
    },
  };
}
