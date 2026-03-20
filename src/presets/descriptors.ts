import {
  MODELS as CANARY_MODELS,
  getLanguageName as getCanaryLanguageName,
} from './canary/catalog.js';
import type { ModelClassification } from '../types/index.js';
import {
  fetchModelFiles,
  getAvailableQuantModes,
  type QuantizationMode,
} from '../runtime/huggingface.js';
import {
  CANARY_180M_FLASH_DOCS,
  resolveCanaryArtifactSource,
  resolveCanaryPresetManifest,
} from './canary/manifest.js';
import { resolveMedAsrPresetManifest } from './medasr/manifest.js';
import {
  DEFAULT_MODEL as PARAKEET_DEFAULT_MODEL,
  getLanguageName as getParakeetLanguageName,
  MODELS as PARAKEET_MODELS,
} from './parakeet/catalog.js';
import {
  resolveParakeetArtifactSource,
  resolveParakeetPresetManifest,
} from './parakeet/manifest.js';
import { resolveWhisperPresetManifest } from './whisper/manifest.js';

export type BuiltInPresetId = 'parakeet' | 'canary' | 'medasr' | 'whisper';
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

function buildWarmupExpectedTexts(): readonly string[] {
  return [
    'The boy was there when the sun rose.',
    'The boy was there when the sun rose. A rod is used to catch pink salmon.',
  ];
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

function createParakeetDescriptors(): BuiltInModelDescriptor[] {
  return Object.entries(PARAKEET_MODELS).map(([modelId, config]) => {
    const manifest = resolveParakeetPresetManifest(modelId);
    const source = resolveParakeetArtifactSource(modelId);
    const warmupExpectedTexts =
      'warmupExpectedTexts' in config ? config.warmupExpectedTexts : undefined;
    const warmupRequiredKeywordGroups =
      'warmupRequiredKeywordGroups' in config ? config.warmupRequiredKeywordGroups : undefined;

    return {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'parakeet',
      displayName: config.displayName,
      description:
        manifest?.description ??
        `Parakeet ${config.topology === 'rnnt' ? 'RNNT' : 'TDT'} model preset.`,
      classification: manifest?.classification ?? {},
      languages: config.languages,
      defaultSourceLanguage: config.defaultLanguage,
      defaultTargetLanguage: config.defaultLanguage,
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: config.supportsWordTimestamps ?? true,
        supportsSegmentTimestamps: false,
        supportsPromptControls: false,
      },
      loading: {
        repoId: config.repoId,
        supportsHubSource: true,
        supportsLocalSource: true,
        supportsSplitQuantization: true,
        supportsSplitBackends: true,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp16', 'int8', 'fp32'],
        availableDecoderQuantizations: ['fp16', 'int8', 'fp32'],
        defaultEncoderQuantization: 'fp16',
        defaultDecoderQuantization: 'int8',
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js', 'onnx'],
        defaultPreprocessorBackend: source?.preprocessorBackend ?? 'js',
        defaultPreprocessorName: config.preprocessor,
        encoderArtifactBaseName: 'encoder-model',
        decoderArtifactBaseName: 'decoder_joint-model',
        defaultRevision:
          config.defaultRevision ??
          (modelId === PARAKEET_DEFAULT_MODEL
            ? 'feat/fp16-canonical-v2'
            : 'feat/fp16-canonical-v3'),
      },
      controls: [],
      warmup: {
        mode: 'expected-text',
        expectedTexts: warmupExpectedTexts ?? buildWarmupExpectedTexts(),
        requiredKeywordGroups: warmupRequiredKeywordGroups,
      },
    };
  });
}

function createCanaryDescriptors(): BuiltInModelDescriptor[] {
  return Object.entries(CANARY_MODELS).map(([modelId, config]) => {
    const manifest = resolveCanaryPresetManifest(modelId);
    const source = resolveCanaryArtifactSource(modelId);

    return {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'canary',
      displayName: config.displayName,
      description: CANARY_180M_FLASH_DOCS.architecture.summary,
      classification: manifest?.classification ?? {},
      languages: config.languages,
      defaultSourceLanguage: config.defaultSourceLanguage,
      defaultTargetLanguage: config.defaultTargetLanguage,
      capabilities: {
        supportedTasks: ['asr', 'speech-translation'],
        supportsTranslation: true,
        supportsPunctuationCapitalization: true,
        supportsInverseTextNormalization: true,
        supportsWordTimestamps: CANARY_180M_FLASH_DOCS.capabilities.supportsWordTimestamps,
        supportsSegmentTimestamps: CANARY_180M_FLASH_DOCS.capabilities.supportsSegmentTimestamps,
        supportsPromptControls: true,
      },
      loading: {
        repoId: config.repoId,
        supportsHubSource: true,
        supportsLocalSource: false,
        supportsSplitQuantization: true,
        supportsSplitBackends: true,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['webgpu', 'wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'wasm',
        availableEncoderQuantizations: ['fp16', 'int8', 'fp32'],
        availableDecoderQuantizations: ['fp16', 'int8', 'fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'int8',
        supportsPreprocessorBackend: true,
        preprocessorBackends: ['js', 'onnx'],
        defaultPreprocessorBackend: source?.preprocessorBackend ?? 'js',
        defaultPreprocessorName: 'nemo128',
        encoderArtifactBaseName: 'encoder-model',
        decoderArtifactBaseName: 'decoder-model',
        defaultRevision: 'main',
      },
      controls: [
        {
          key: 'task',
          label: 'Task',
          description:
            'Controls whether the decoder performs same-language ASR or speech translation.',
          type: 'enum',
          defaultValue: 'asr',
          options: [
            { value: 'asr', label: 'ASR' },
            { value: 'translation', label: 'Speech Translation' },
          ],
        },
        {
          key: 'targetLanguage',
          label: 'Target Language',
          description: 'Target text language prompt token.',
          type: 'language',
          defaultValue: config.defaultTargetLanguage,
          visibleWhen: {
            key: 'task',
            notEquals: 'asr',
          },
        },
        {
          key: 'punctuate',
          label: 'Punctuation & Capitalization',
          description: 'Injects the Canary PnC prompt toggle.',
          type: 'boolean',
          defaultValue: true,
        },
        {
          key: 'timestamps',
          label: 'Timestamps',
          description: 'Injects the Canary timestamp prompt toggle.',
          type: 'boolean',
          defaultValue: false,
        },
      ],
      warmup: {
        mode: 'expected-text',
        expectedTexts: buildWarmupExpectedTexts(),
      },
      docs: {
        modelCardUrl: CANARY_180M_FLASH_DOCS.modelCardUrl,
        architectureSummary: CANARY_180M_FLASH_DOCS.architecture.summary,
        promptingNote: CANARY_180M_FLASH_DOCS.prompting.note,
      },
    };
  });
}

function createMedAsrDescriptors(): BuiltInModelDescriptor[] {
  const manifest = resolveMedAsrPresetManifest('google/medasr');
  const source = manifest?.source;
  return [
    {
      modelId: 'google/medasr',
      aliases: manifest?.aliases ?? [],
      preset: 'medasr',
      displayName: 'Google MedASR (Medical English)',
      description: manifest?.description ?? 'Google MedASR preset.',
      classification: manifest?.classification ?? {},
      languages: manifest?.config.languages ?? ['en'],
      defaultSourceLanguage: 'en',
      defaultTargetLanguage: 'en',
      capabilities: {
        supportedTasks: ['asr'],
        supportsTranslation: false,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsPromptControls: false,
      },
      loading: {
        repoId: source?.kind === 'huggingface' ? source.repoId : undefined,
        supportsHubSource: true,
        supportsLocalSource: false,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['webgpu', 'wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'webgpu',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        encoderArtifactBaseName: 'model',
        decoderArtifactBaseName: 'model',
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
    },
  ];
}

function createWhisperDescriptors(): BuiltInModelDescriptor[] {
  return ['openai/whisper-base', 'openai/whisper-large-v3'].map((modelId) => {
    const manifest = resolveWhisperPresetManifest(modelId);
    return {
      modelId,
      aliases: manifest?.aliases ?? [],
      preset: 'whisper',
      displayName: modelId === 'openai/whisper-base' ? 'Whisper Base' : 'Whisper Large v3',
      description: manifest?.description ?? 'Whisper preset.',
      classification: manifest?.classification ?? {},
      languages: manifest?.config.languages ?? ['auto'],
      defaultSourceLanguage: 'auto',
      defaultTargetLanguage: 'auto',
      capabilities: {
        supportedTasks: ['asr', 'speech-translation'],
        supportsTranslation: true,
        supportsPunctuationCapitalization: false,
        supportsInverseTextNormalization: false,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsPromptControls: false,
      },
      loading: {
        supportsHubSource: true,
        supportsLocalSource: false,
        supportsSplitQuantization: false,
        supportsSplitBackends: false,
        availableEncoderBackends: ['webgpu', 'wasm'],
        availableDecoderBackends: ['webgpu', 'wasm'],
        defaultEncoderBackend: 'webgpu',
        defaultDecoderBackend: 'webgpu',
        availableEncoderQuantizations: ['fp32'],
        availableDecoderQuantizations: ['fp32'],
        defaultEncoderQuantization: 'fp32',
        defaultDecoderQuantization: 'fp32',
        supportsPreprocessorBackend: false,
        preprocessorBackends: [],
        defaultPreprocessorBackend: null,
        defaultPreprocessorName: null,
        defaultRevision: 'main',
      },
      controls: [],
      warmup: {
        mode: 'non-empty',
      },
    };
  });
}

const BUILT_IN_MODEL_DESCRIPTORS = [
  ...createParakeetDescriptors(),
  ...createCanaryDescriptors(),
  ...createMedAsrDescriptors(),
  ...createWhisperDescriptors(),
] as const;

const LANGUAGE_NAME_RESOLVERS = [getParakeetLanguageName, getCanaryLanguageName] as const;

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
  for (const resolve of LANGUAGE_NAME_RESOLVERS) {
    const label = resolve(languageCode);
    if (label !== languageCode) {
      return label;
    }
  }
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
      const source = resolveParakeetArtifactSource(descriptor.modelId);
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
      const source = resolveCanaryArtifactSource(descriptor.modelId);
      if (!source || source.kind !== 'huggingface') {
        throw new Error(`Could not resolve Canary artifact source for "${descriptor.modelId}".`);
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
      const manifest = resolveMedAsrPresetManifest(descriptor.modelId);
      const source = manifest?.source;
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
