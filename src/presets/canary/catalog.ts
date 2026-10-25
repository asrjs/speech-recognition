import {
  CANARY_180M_FLASH_DOCS,
  CANARY_1B_V2_DECODER_LAYERS,
  CANARY_1B_V2_ENCODER_LAYERS,
  CANARY_1B_V2_LANGUAGES,
  CANARY_1B_V2_VOCABULARY_SIZE,
  DEFAULT_MODEL,
  resolveCanaryArtifactSource,
} from './manifest.js';

export const LANGUAGE_NAMES = {
  bg: 'Bulgarian',
  hr: 'Croatian',
  cs: 'Czech',
  da: 'Danish',
  nl: 'Dutch',
  en: 'English',
  et: 'Estonian',
  fi: 'Finnish',
  fr: 'French',
  de: 'German',
  el: 'Greek',
  hu: 'Hungarian',
  it: 'Italian',
  lv: 'Latvian',
  lt: 'Lithuanian',
  mt: 'Maltese',
  pl: 'Polish',
  pt: 'Portuguese',
  ro: 'Romanian',
  sk: 'Slovak',
  sl: 'Slovenian',
  es: 'Spanish',
  sv: 'Swedish',
  ru: 'Russian',
  uk: 'Ukrainian',
} as const;

export interface CanaryModelConfig {
  readonly repoId: string;
  readonly displayName: string;
  readonly languages: readonly string[];
  readonly defaultSourceLanguage: string;
  readonly defaultTargetLanguage: string;
  readonly vocabSize: number;
  readonly featuresSize: number;
  readonly encoderLayers: number;
  readonly decoderLayers: number;
  readonly promptFormat: 'canary' | 'canary2';
}

const defaultSource = resolveCanaryArtifactSource(DEFAULT_MODEL);

export const MODELS = {
  [DEFAULT_MODEL]: {
    repoId:
      defaultSource?.kind === 'huggingface'
        ? defaultSource.repoId
        : 'ysdede/canary-180m-flash-onnx',
    displayName: 'Canary 180M Flash (Multilingual AED)',
    languages: ['en', 'de', 'es', 'fr'],
    defaultSourceLanguage: 'en',
    defaultTargetLanguage: 'en',
    vocabSize: 5248,
    featuresSize: 128,
    encoderLayers: CANARY_180M_FLASH_DOCS.architecture.encoderLayers,
    decoderLayers: CANARY_180M_FLASH_DOCS.architecture.decoderLayers,
    promptFormat: 'canary2',
  },
  'nvidia/canary-1b': {
    repoId: 'nvidia/canary-1b',
    displayName: 'Canary 1B (Multilingual AED)',
    languages: ['en', 'de', 'es', 'fr'],
    defaultSourceLanguage: 'en',
    defaultTargetLanguage: 'en',
    vocabSize: 4128,
    featuresSize: 128,
    encoderLayers: 24,
    decoderLayers: 24,
    promptFormat: 'canary',
  },
  'nvidia/canary-1b-v2': {
    repoId: 'nvidia/canary-1b-v2',
    displayName: 'Canary 1B v2 (Multilingual AED)',
    languages: CANARY_1B_V2_LANGUAGES,
    defaultSourceLanguage: 'en',
    defaultTargetLanguage: 'en',
    vocabSize: CANARY_1B_V2_VOCABULARY_SIZE,
    featuresSize: 128,
    encoderLayers: CANARY_1B_V2_ENCODER_LAYERS,
    decoderLayers: CANARY_1B_V2_DECODER_LAYERS,
    promptFormat: 'canary2',
  },
} satisfies Record<string, CanaryModelConfig>;

export function getModelConfig(modelKeyOrRepoId: string): CanaryModelConfig | null {
  if (Object.prototype.hasOwnProperty.call(MODELS, modelKeyOrRepoId)) {
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

export function listModels(): string[] {
  return Object.keys(MODELS);
}

export function getLanguageName(languageCode: string): string {
  return LANGUAGE_NAMES[languageCode.toLowerCase() as keyof typeof LANGUAGE_NAMES] ?? languageCode;
}
