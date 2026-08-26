import {
  SENSEVOICE_LANGUAGES,
  SENSEVOICE_LANGUAGE_IDS,
  SENSEVOICE_TEXTNORM_IDS,
  type SenseVoiceLanguage,
  type SenseVoicePrompt,
} from './types.js';

export function resolveSenseVoiceLanguage(value: string | undefined): SenseVoiceLanguage {
  const normalized = (value ?? 'auto').trim().toLowerCase();
  return (SENSEVOICE_LANGUAGES as readonly string[]).includes(normalized)
    ? (normalized as SenseVoiceLanguage)
    : 'auto';
}

export function createSenseVoicePrompt(options: {
  readonly language?: string;
  readonly useItn?: boolean;
} = {}): SenseVoicePrompt {
  const language = resolveSenseVoiceLanguage(options.language);
  const textnorm = options.useItn === false ? 'woitn' : 'withitn';
  return {
    language,
    languageId: SENSEVOICE_LANGUAGE_IDS[language],
    textnorm,
    textnormId: SENSEVOICE_TEXTNORM_IDS[textnorm],
  };
}
