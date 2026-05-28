import { CANARY_180M_FLASH_DOCS, DEFAULT_MODEL, resolveCanaryArtifactSource } from './manifest.js';

export const LANGUAGE_NAMES = {
  en: 'English',
  de: 'German',
  es: 'Spanish',
  fr: 'French',
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
  },
} satisfies Record<string, CanaryModelConfig>;

// ⚡ Bolt: Static mapping for O(1) repoId to key lookups instead of O(N) Object.entries() iterations
const REPO_ID_TO_KEY = new Map<string, string>();
for (const [key, config] of Object.entries(MODELS)) {
  REPO_ID_TO_KEY.set(config.repoId, key);
}

export function getModelConfig(modelKeyOrRepoId: string): CanaryModelConfig | null {
  if (Object.prototype.hasOwnProperty.call(MODELS, modelKeyOrRepoId)) {
    return MODELS[modelKeyOrRepoId as keyof typeof MODELS];
  }

  const key = REPO_ID_TO_KEY.get(modelKeyOrRepoId);
  if (key !== undefined) {
    return MODELS[key as keyof typeof MODELS];
  }

  return null;
}

export function getModelKeyFromRepoId(repoId: string): string | null {
  return REPO_ID_TO_KEY.get(repoId) ?? null;
}

export function listModels(): string[] {
  return Object.keys(MODELS);
}

export function getLanguageName(languageCode: string): string {
  return LANGUAGE_NAMES[languageCode.toLowerCase() as keyof typeof LANGUAGE_NAMES] ?? languageCode;
}
