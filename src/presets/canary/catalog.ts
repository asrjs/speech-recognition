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

// ⚡ Bolt: Pre-compute reverse mapping for O(1) lookups by repoId
const REPO_ID_TO_KEY = new Map<string, string>(
  Object.entries(MODELS).map(([key, config]) => [config.repoId, key]),
);

export function getModelConfig(modelKeyOrRepoId: string): CanaryModelConfig | null {
  if (modelKeyOrRepoId in MODELS) {
    return MODELS[modelKeyOrRepoId as keyof typeof MODELS];
  }

  // ⚡ Bolt: Replaced O(N) linear search and Object.values() allocation with O(1) reverse map lookup
  const key = REPO_ID_TO_KEY.get(modelKeyOrRepoId);
  if (key) {
    return MODELS[key as keyof typeof MODELS];
  }

  return null;
}

export function getModelKeyFromRepoId(repoId: string): string | null {
  // ⚡ Bolt: Replaced O(N) linear search and Object.entries() allocation with O(1) reverse map lookup
  return REPO_ID_TO_KEY.get(repoId) ?? null;
}

export function listModels(): string[] {
  return Object.keys(MODELS);
}

export function getLanguageName(languageCode: string): string {
  return LANGUAGE_NAMES[languageCode.toLowerCase() as keyof typeof LANGUAGE_NAMES] ?? languageCode;
}
