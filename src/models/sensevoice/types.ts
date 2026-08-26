import type { LasrCtcFeatureBatch } from '../lasr-ctc/types.js';

export const SENSEVOICE_LANGUAGES = ['auto', 'zh', 'en', 'yue', 'ja', 'ko'] as const;
export type SenseVoiceLanguage = (typeof SENSEVOICE_LANGUAGES)[number];

export const SENSEVOICE_LANGUAGE_IDS: Readonly<Record<SenseVoiceLanguage, number>> = {
  auto: 0,
  zh: 3,
  en: 4,
  yue: 7,
  ja: 11,
  ko: 12,
};

export const SENSEVOICE_TEXTNORM_IDS = {
  withitn: 14,
  woitn: 15,
} as const;

export interface SenseVoiceFeatureBatch extends LasrCtcFeatureBatch {
  readonly validFrameCount: number;
}

export interface SenseVoicePrompt {
  readonly language: SenseVoiceLanguage;
  readonly languageId: number;
  readonly textnorm: 'withitn' | 'woitn';
  readonly textnormId: number;
}

export interface SenseVoiceNativeMetadata {
  readonly language?: string;
  readonly emotion?: string;
  readonly event?: string;
}
