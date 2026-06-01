export const LANGUAGE_NAMES = {
  en: 'English',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  nl: 'Dutch',
  pl: 'Polish',
  ru: 'Russian',
  uk: 'Ukrainian',
  ja: 'Japanese',
  ko: 'Korean',
  zh: 'Chinese',
} as const;

export interface ParakeetModelConfig {
  readonly repoId: string;
  readonly displayName: string;
  readonly languages: readonly string[];
  readonly defaultLanguage: string;
  readonly vocabSize: number;
  readonly featuresSize: number;
  readonly preprocessor: 'nemo80' | 'nemo128';
  readonly subsampling: number;
  readonly predHidden: number;
  readonly predLayers: number;
  readonly topology?: 'tdt' | 'rnnt';
  readonly supportsWordTimestamps?: boolean;
  readonly defaultRevision?: string;
  readonly cacheKeyFallbackRevisions?: readonly string[];
  readonly warmupExpectedTexts?: readonly string[];
  readonly warmupRequiredKeywordGroups?: readonly (readonly string[])[];
}

export interface ParakeetDefaultWeightSetup {
  readonly encoderDefault: 'fp16' | 'fp32' | 'int8';
  readonly decoderDefault: 'fp16' | 'fp32' | 'int8';
  readonly encoderFallback: 'fp16' | 'fp32' | 'int8';
  readonly encoderPreferred: readonly ('fp16' | 'fp32' | 'int8')[];
  readonly decoderPreferred: readonly ('fp16' | 'fp32' | 'int8')[];
}

export const MODELS = {
  'parakeet-tdt-0.6b-v2': {
    repoId: 'ysdede/parakeet-tdt-0.6b-v2-onnx',
    displayName: 'Parakeet TDT 0.6B v2 (English)',
    languages: ['en'],
    defaultLanguage: 'en',
    vocabSize: 1025,
    featuresSize: 128,
    preprocessor: 'nemo128',
    subsampling: 8,
    predHidden: 640,
    predLayers: 2,
    topology: 'tdt',
    supportsWordTimestamps: true,
    defaultRevision: 'main',
    cacheKeyFallbackRevisions: ['feat/fp16-canonical-v2'],
  },
  'parakeet-tdt-0.6b-v3': {
    repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
    displayName: 'Parakeet TDT 0.6B v3 (Multilingual)',
    languages: ['en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'uk', 'ja', 'ko', 'zh'],
    defaultLanguage: 'en',
    vocabSize: 8193,
    featuresSize: 128,
    preprocessor: 'nemo128',
    subsampling: 8,
    predHidden: 640,
    predLayers: 2,
    topology: 'tdt',
    supportsWordTimestamps: true,
    defaultRevision: 'main',
    cacheKeyFallbackRevisions: ['feat/fp16-canonical-v3'],
  },
  'parakeet-realtime-eou-120m-v1': {
    repoId: 'ysdede/parakeet-realtime-eou-120m-v1-onnx',
    displayName: 'Parakeet Realtime EOU 120M v1 (English)',
    languages: ['en'],
    defaultLanguage: 'en',
    vocabSize: 1026,
    featuresSize: 128,
    preprocessor: 'nemo128',
    subsampling: 8,
    predHidden: 640,
    predLayers: 1,
    topology: 'rnnt',
    supportsWordTimestamps: false,
    defaultRevision: '6d6be8e9113b4aa8ae7b4d5dfb655795c084d0c6',
    warmupExpectedTexts: [
      'the boy was there when the sun rose',
      'the boy was there when the sun rose a rod is used to catch pink salmon',
    ],
    warmupRequiredKeywordGroups: [
      ['boy', 'there'],
      ['pink', 'salmon'],
    ],
  },
} satisfies Record<string, ParakeetModelConfig>;

export const DEFAULT_MODEL = 'parakeet-tdt-0.6b-v2' as const;

// Cache repository IDs to keys for O(1) performance impact instead of O(N) linear search
const REPO_ID_TO_KEY = new Map<string, string>(
  Object.entries(MODELS).map(([key, config]) => [config.repoId, key])
);

// Cache language sets for O(1) performance impact instead of O(N) array inclusions
const LANGUAGE_SETS = new WeakMap<ParakeetModelConfig, Set<string>>();

export function getModelConfig(modelKeyOrRepoId: string): ParakeetModelConfig | null {
  if (Object.prototype.hasOwnProperty.call(MODELS, modelKeyOrRepoId)) {
    return MODELS[modelKeyOrRepoId as keyof typeof MODELS];
  }

  // O(1) reverse lookup, replacing O(N) Object.values() linear search
  const key = REPO_ID_TO_KEY.get(modelKeyOrRepoId);
  if (key) {
    return MODELS[key as keyof typeof MODELS];
  }

  return null;
}

export function getModelKeyFromRepoId(repoId: string): string | null {
  // O(1) lookup, replacing O(N) Object.entries() linear search
  return REPO_ID_TO_KEY.get(repoId) ?? null;
}

export function supportsLanguage(modelKeyOrRepoId: string, language: string): boolean {
  const config = getModelConfig(modelKeyOrRepoId);
  if (!config) {
    return false;
  }

  // O(1) average lookup performance, replacing O(N) array.includes() calls
  let languageSet = LANGUAGE_SETS.get(config);
  if (!languageSet) {
    languageSet = new Set(config.languages);
    LANGUAGE_SETS.set(config, languageSet);
  }

  return languageSet.has(language.toLowerCase());
}

export function listModels(): string[] {
  return Object.keys(MODELS);
}

export function getLanguageName(languageCode: string): string {
  return LANGUAGE_NAMES[languageCode.toLowerCase() as keyof typeof LANGUAGE_NAMES] ?? languageCode;
}

export function getParakeetDefaultWeightSetup(
  _modelKeyOrRepoId: string,
  backend = 'webgpu-hybrid',
): ParakeetDefaultWeightSetup {
  if (String(backend).startsWith('webgpu')) {
    return {
      encoderDefault: 'fp16',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['fp16', 'fp32', 'int8'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    };
  }

  return {
    encoderDefault: 'int8',
    decoderDefault: 'int8',
    encoderFallback: 'fp32',
    encoderPreferred: ['int8', 'fp32', 'fp16'],
    decoderPreferred: ['int8', 'fp32', 'fp16'],
  };
}
