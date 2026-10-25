import type { ModelClassification, ModelInferenceLimits } from '../types/index.js';
import { getAvailableQuantModes, type QuantizationMode } from '../runtime/artifact-quantization.js';
import { CATALOG_DESCRIPTORS as BUILT_IN_MODEL_DESCRIPTORS, CATALOG_LANGUAGES, getCatalogArtifactSource } from './catalog-data.generated.js';

export { BUILT_IN_MODEL_DESCRIPTORS };

export type BuiltInPresetId =
  | 'parakeet'
  | 'canary'
  | 'medasr'
  | 'nemotron'
  | 'whisper'
  | 'wav2vec2'
  | 'wav2vec2-tr'
  | 'moonshine'
  | 'qwen3-asr'
  | 'parakeet-ctc'
  | 'omnilingual-ctc'
  | 'gigaam-ctc'
  | 'gigaam-rnnt'
  | 'sensevoice'
  | 'x-asr';
export type BuiltInModelTask = 'asr' | 'speech-translation';
export type BuiltInWarmupMode = 'expected-text' | 'non-empty';
export type BuiltInControlType = 'boolean' | 'enum' | 'language';
export type BuiltInExecutionBackend = 'webgpu' | 'wasm';

export interface BuiltInControlOption {
  readonly value: string;
  readonly label: string;
  readonly description?: string;
}

export interface BuiltInControlDescriptor {
  readonly key: string;
  readonly label: string;
  readonly description?: string;
  readonly type: BuiltInControlType;
  readonly defaultValue?: string | boolean;
  readonly options?: readonly BuiltInControlOption[];
  readonly visibleWhen?: {
    readonly key: string;
    readonly notEquals?: string | boolean;
    readonly equals?: string | boolean;
  };
}

export interface BuiltInModelCapabilities {
  readonly supportedTasks: readonly BuiltInModelTask[];
  readonly supportsTranslation: boolean;
  readonly supportsPunctuationCapitalization: boolean;
  readonly supportsInverseTextNormalization: boolean;
  readonly supportsWordTimestamps: boolean;
  readonly supportsSegmentTimestamps: boolean;
  readonly supportsPromptControls: boolean;
}

export interface BuiltInModelLoadingDescriptor {
  readonly repoId?: string;
  readonly supportsHubSource: boolean;
  readonly supportsLocalSource: boolean;
  readonly supportsSplitQuantization: boolean;
  readonly supportsSplitBackends: boolean;
  readonly availableEncoderBackends: readonly BuiltInExecutionBackend[];
  readonly availableDecoderBackends: readonly BuiltInExecutionBackend[];
  readonly defaultEncoderBackend: BuiltInExecutionBackend;
  readonly defaultDecoderBackend: BuiltInExecutionBackend;
  readonly availableEncoderQuantizations: readonly ('int8' | 'fp32' | 'fp16')[];
  readonly availableDecoderQuantizations: readonly ('int8' | 'fp32' | 'fp16')[];
  readonly defaultEncoderQuantization: 'int8' | 'fp32' | 'fp16';
  readonly defaultDecoderQuantization: 'int8' | 'fp32' | 'fp16';
  readonly supportsPreprocessorBackend: boolean;
  readonly preprocessorBackends: readonly ('js' | 'onnx')[];
  readonly defaultPreprocessorBackend?: 'js' | 'onnx' | null;
  readonly defaultPreprocessorName?: string | null;
  readonly encoderArtifactBaseName?: string;
  readonly decoderArtifactBaseName?: string;
  readonly defaultRevision: string;
}

export interface BuiltInModelWarmupDescriptor {
  readonly mode: BuiltInWarmupMode;
  readonly expectedTexts?: readonly string[];
  readonly requiredKeywordGroups?: readonly (readonly string[])[];
}

export interface BuiltInModelDescriptor {
  /** Marks a discoverable catalog entry whose local integration is still experimental. */
  readonly status?: 'experimental';
  readonly modelId: string;
  readonly aliases: readonly string[];
  readonly preset: BuiltInPresetId;
  readonly displayName: string;
  readonly description: string;
  readonly classification: Partial<ModelClassification>;
  readonly languages: readonly string[];
  readonly defaultSourceLanguage: string;
  readonly defaultTargetLanguage?: string;
  readonly capabilities: BuiltInModelCapabilities;
  readonly inference: ModelInferenceLimits;
  readonly loading: BuiltInModelLoadingDescriptor;
  readonly controls: readonly BuiltInControlDescriptor[];
  readonly warmup: BuiltInModelWarmupDescriptor;
  readonly docs?: {
    readonly modelCardUrl?: string;
    readonly architectureSummary?: string;
    readonly promptingNote?: string;
  };
}

export interface BuiltInModelOption {
  readonly key: string;
  readonly repoId?: string;
  readonly displayName: string;
  readonly languages: readonly string[];
  readonly preset: BuiltInPresetId;
}

export interface BuiltInDetectedQuantizations {
  readonly repoId?: string;
  readonly revision: string;
  readonly encoderArtifactBaseName?: string;
  readonly decoderArtifactBaseName?: string;
  readonly encoder: readonly QuantizationMode[];
  readonly decoder: readonly QuantizationMode[];
}

export interface BuildBuiltInHubLoadOptionsInput {
  readonly modelId: string;
  readonly revision?: string;
  readonly backend?: string;
  readonly encoderBackend?: BuiltInExecutionBackend;
  readonly decoderBackend?: BuiltInExecutionBackend;
  readonly encoderQuant?: 'int8' | 'fp32' | 'fp16';
  readonly decoderQuant?: 'int8' | 'fp32' | 'fp16';
  readonly preprocessorName?: string;
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface BuildBuiltInTranscriptionOptionsInput {
  readonly sourceLanguage?: string;
  readonly targetLanguage?: string;
  readonly task?: 'asr' | 'translation' | 'ast' | 'translate' | 'speech-translation';
  readonly timestamps?: boolean;
  readonly punctuate?: boolean;
  readonly frameStride?: number;
  readonly returnConfidences?: boolean;
  readonly enableProfiling?: boolean;
}

function normalizeRequestedBackend(backend: string | undefined): string {
  return String(backend ?? '')
    .trim()
    .toLowerCase();
}

function pickBuiltInComponentBackend(
  requestedBackend: BuiltInExecutionBackend | undefined,
  overallBackend: string | undefined,
  availableBackends: readonly BuiltInExecutionBackend[],
  defaultBackend: BuiltInExecutionBackend,
): BuiltInExecutionBackend {
  if (requestedBackend && availableBackends.includes(requestedBackend)) {
    return requestedBackend;
  }

  const normalizedOverall = normalizeRequestedBackend(overallBackend);
  if (normalizedOverall === 'wasm') {
    return availableBackends.includes('wasm') ? 'wasm' : (availableBackends[0] ?? defaultBackend);
  }
  if (normalizedOverall === 'webgpu' || normalizedOverall === 'webgpu-strict') {
    return availableBackends.includes('webgpu')
      ? 'webgpu'
      : (availableBackends[0] ?? defaultBackend);
  }
  if (normalizedOverall === 'webgpu-hybrid') {
    return availableBackends.includes(defaultBackend)
      ? defaultBackend
      : (availableBackends[0] ?? defaultBackend);
  }

  return availableBackends.includes(defaultBackend)
    ? defaultBackend
    : (availableBackends[0] ?? defaultBackend);
}



export function listBuiltInModelDescriptors(): readonly BuiltInModelDescriptor[] {
  return BUILT_IN_MODEL_DESCRIPTORS;
}

export function listBuiltInModelOptions(): readonly BuiltInModelOption[] {
  return BUILT_IN_MODEL_DESCRIPTORS.map((descriptor) => ({
    key: descriptor.modelId,
    repoId: descriptor.loading.repoId,
    displayName: descriptor.displayName,
    languages: descriptor.languages,
    preset: descriptor.preset,
  }));
}

export function detectBuiltInModelQuantizationsFromFiles(
  modelId: string,
  files: readonly string[],
): BuiltInDetectedQuantizations {
  const descriptor = getBuiltInModelDescriptor(modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${modelId}".`);
  }

  const encoderArtifactBaseName = descriptor.loading.encoderArtifactBaseName;
  const decoderArtifactBaseName = descriptor.loading.decoderArtifactBaseName;
  const declaredEncoder = descriptor.loading.availableEncoderQuantizations;
  const declaredDecoder = descriptor.loading.availableDecoderQuantizations;
  const detectedEncoder = encoderArtifactBaseName
    ? getAvailableQuantModes(files, encoderArtifactBaseName)
    : [...declaredEncoder];
  const detectedDecoder = decoderArtifactBaseName
    ? getAvailableQuantModes(files, decoderArtifactBaseName)
    : [...declaredDecoder];

  return {
    repoId: descriptor.loading.repoId,
    revision: descriptor.loading.defaultRevision,
    encoderArtifactBaseName,
    decoderArtifactBaseName,
    encoder: detectedEncoder.length > 0 ? detectedEncoder : [...declaredEncoder],
    decoder: detectedDecoder.length > 0 ? detectedDecoder : [...declaredDecoder],
  };
}

export function resolveBuiltInModelComponentBackends(
  modelId: string,
  input: {
    readonly backend?: string;
    readonly encoderBackend?: BuiltInExecutionBackend;
    readonly decoderBackend?: BuiltInExecutionBackend;
  } = {},
): {
  readonly encoderBackend: BuiltInExecutionBackend;
  readonly decoderBackend: BuiltInExecutionBackend;
} {
  const descriptor = getBuiltInModelDescriptor(modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${modelId}".`);
  }

  return {
    encoderBackend: pickBuiltInComponentBackend(
      input.encoderBackend,
      input.backend,
      descriptor.loading.availableEncoderBackends,
      descriptor.loading.defaultEncoderBackend,
    ),
    decoderBackend: pickBuiltInComponentBackend(
      input.decoderBackend,
      input.backend,
      descriptor.loading.availableDecoderBackends,
      descriptor.loading.defaultDecoderBackend,
    ),
  };
}

export async function detectBuiltInModelRepoQuantizations(
  modelId: string,
  options: {
    readonly revision?: string;
  } = {},
): Promise<BuiltInDetectedQuantizations> {
  const descriptor = getBuiltInModelDescriptor(modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${modelId}".`);
  }

  const repoId = descriptor.loading.repoId;
  const revision = options.revision ?? descriptor.loading.defaultRevision;

  if (!repoId) {
    return {
      repoId,
      revision,
      encoderArtifactBaseName: descriptor.loading.encoderArtifactBaseName,
      decoderArtifactBaseName: descriptor.loading.decoderArtifactBaseName,
      encoder: [...descriptor.loading.availableEncoderQuantizations],
      decoder: [...descriptor.loading.availableDecoderQuantizations],
    };
  }

  const { fetchModelFiles } = await import('../runtime/huggingface.js');
  const files = await fetchModelFiles(repoId, revision);
  const detected = detectBuiltInModelQuantizationsFromFiles(modelId, files);
  return {
    ...detected,
    repoId,
    revision,
  };
}

export function getBuiltInModelDescriptor(modelId: string): BuiltInModelDescriptor | null {
  const normalized = String(modelId || '')
    .trim()
    .toLowerCase();
  return (
    BUILT_IN_MODEL_DESCRIPTORS.find(
      (descriptor) =>
        descriptor.modelId.toLowerCase() === normalized ||
        descriptor.aliases.some((alias) => alias.toLowerCase() === normalized),
    ) ?? null
  );
}

export function getBuiltInModelLanguageName(languageCode: string): string {
  const label = CATALOG_LANGUAGES[languageCode.toLowerCase()];
  if (label !== undefined && label !== languageCode) return label;
  if (String(languageCode).toLowerCase() === 'auto') {
    return 'Auto-detect';
  }
  return languageCode;
}

export function buildBuiltInHubLoadOptions(
  input: BuildBuiltInHubLoadOptionsInput,
): Record<string, unknown> {
  const descriptor = getBuiltInModelDescriptor(input.modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${input.modelId}".`);
  }

  const componentBackends = resolveBuiltInModelComponentBackends(input.modelId, {
    backend: input.backend,
    encoderBackend: input.encoderBackend,
    decoderBackend: input.decoderBackend,
  });

  const base = {
    preset: descriptor.preset,
    modelId: descriptor.modelId,
    backend: input.backend,
  } as Record<string, unknown>;

  switch (descriptor.preset) {
    case 'parakeet': {
      const source = getCatalogArtifactSource(descriptor.modelId);
      if (!source || source.kind !== 'huggingface') {
        throw new Error(`Could not resolve Parakeet artifact source for "${descriptor.modelId}".`);
      }
      return {
        ...base,
        options: {
          source: {
            ...source,
            revision: input.revision || source.revision || descriptor.loading.defaultRevision,
            encoderBackend: componentBackends.encoderBackend,
            decoderBackend: componentBackends.decoderBackend,
            encoderQuant: input.encoderQuant,
            decoderQuant: input.decoderQuant,
            preprocessorName:
              input.preprocessorName ?? descriptor.loading.defaultPreprocessorName ?? undefined,
            preprocessorBackend:
              input.preprocessorBackend ??
              descriptor.loading.defaultPreprocessorBackend ??
              undefined,
            cpuThreads: input.cpuThreads,
            enableProfiling: input.enableProfiling,
          },
        },
      };
    }
    case 'canary': {
      const source = getCatalogArtifactSource(descriptor.modelId);
      if (!source || source.kind !== 'huggingface') {
        // Keep the request intact so the preset can report its artifact/adapter
        // requirement. Do not invent a Hub path for an unconfigured export.
        return {
          ...base,
          options: {
            cpuThreads: input.cpuThreads,
            enableProfiling: input.enableProfiling,
          },
        };
      }
      return {
        ...base,
        options: {
          source: {
            ...source,
            revision: input.revision || source.revision || descriptor.loading.defaultRevision,
            encoderBackend: componentBackends.encoderBackend,
            decoderBackend: componentBackends.decoderBackend,
            encoderQuant: input.encoderQuant,
            decoderQuant: input.decoderQuant,
            preprocessorName:
              input.preprocessorName ?? descriptor.loading.defaultPreprocessorName ?? undefined,
            preprocessorBackend:
              input.preprocessorBackend ??
              descriptor.loading.defaultPreprocessorBackend ??
              undefined,
            cpuThreads: input.cpuThreads,
            enableProfiling: input.enableProfiling,
          },
        },
      };
    }
    case 'medasr': {
      const source = getCatalogArtifactSource(descriptor.modelId);
      if (!source || source.kind !== 'huggingface') {
        throw new Error(`Could not resolve MedASR artifact source for "${descriptor.modelId}".`);
      }
      return {
        ...base,
        options: {
          source: {
            ...source,
            revision: input.revision || source.revision || descriptor.loading.defaultRevision,
            cpuThreads: input.cpuThreads,
            enableProfiling: input.enableProfiling,
          },
        },
      };
    }
    case 'nemotron': {
      const source = getCatalogArtifactSource(descriptor.modelId);
      if (!source || source.kind !== 'huggingface') {
        throw new Error(`Could not resolve Nemotron artifact source for "${descriptor.modelId}".`);
      }
      return {
        ...base,
        options: {
          source: {
            ...source,
            revision: input.revision || source.revision || descriptor.loading.defaultRevision,
            cpuThreads: input.cpuThreads,
            enableProfiling: input.enableProfiling,
          },
        },
      };
    }
    case 'whisper':
      return base;
    default:
      return base;
  }
}

export function buildBuiltInTranscriptionOptions(
  modelId: string,
  input: BuildBuiltInTranscriptionOptionsInput = {},
): Record<string, unknown> {
  const descriptor = getBuiltInModelDescriptor(modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${modelId}".`);
  }

  if (descriptor.preset === 'canary') {
    const sourceLanguage = input.sourceLanguage ?? descriptor.defaultSourceLanguage;
    const normalizedTask = input.task ?? 'asr';
    const targetLanguage =
      normalizedTask === 'asr'
        ? sourceLanguage
        : (input.targetLanguage ?? descriptor.defaultTargetLanguage ?? sourceLanguage);

    return {
      sourceLanguage,
      targetLanguage,
      task: normalizedTask,
      pnc: input.punctuate === undefined ? 'yes' : input.punctuate ? 'yes' : 'no',
      timestamp: input.timestamps ? 'yes' : 'no',
      enableProfiling: input.enableProfiling,
    };
  }

  return {
    returnTimestamps: input.timestamps ?? true,
    returnConfidences: input.returnConfidences ?? true,
    frameStride: input.frameStride,
    enableProfiling: input.enableProfiling,
  };
}
