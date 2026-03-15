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
  },
} satisfies Record<string, ParakeetModelConfig>;

export const DEFAULT_MODEL = 'parakeet-tdt-0.6b-v2' as const;

export function getModelConfig(modelKeyOrRepoId: string): ParakeetModelConfig | null {
  if (modelKeyOrRepoId in MODELS) {
    return MODELS[modelKeyOrRepoId as keyof typeof MODELS];
  }

  for (const config of Object.values(MODELS)) {
    if (config.repoId === modelKeyOrRepoId) {
      return config;
    }
  }

  return null;
}

export function getModelKeyFromRepoId(repoId: string): string | null {
  for (const [key, config] of Object.entries(MODELS)) {
    if (config.repoId === repoId) {
      return key;
    }
  }
  return null;
}

export function supportsLanguage(modelKeyOrRepoId: string, language: string): boolean {
  const config = getModelConfig(modelKeyOrRepoId);
  if (!config) {
    return false;
  }
  return config.languages.includes(language.toLowerCase());
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
