import type { TextTokenizer } from '../../tokenizers/index.js';

export const WHISPER_LANGUAGE_TOKEN_START = 50259;
export const WHISPER_LANGUAGE_TOKEN_END = 50357;

const WHISPER_LANGUAGE_TOKEN_PATTERN = /^<\|([a-z]{2,3})\|>$/;

export function selectWhisperLanguageFromLogits(
  tokenizer: Pick<TextTokenizer, 'idsToTokens'>,
  logits: Float32Array,
  vocabSize: number,
): string | undefined {
  if (!tokenizer.idsToTokens || logits.length === 0 || vocabSize <= WHISPER_LANGUAGE_TOKEN_START) {
    return undefined;
  }

  const sliceStart = Math.max(0, logits.length - vocabSize);
  const sliceEnd = Math.min(logits.length, sliceStart + vocabSize);
  let bestLanguage: string | undefined;
  let bestLogit = Number.NEGATIVE_INFINITY;

  for (
    let tokenId = WHISPER_LANGUAGE_TOKEN_START;
    tokenId <= WHISPER_LANGUAGE_TOKEN_END && tokenId < vocabSize;
    tokenId++
  ) {
    const offset = sliceStart + tokenId;
    if (offset >= sliceEnd) break;

    const token = tokenizer.idsToTokens([tokenId])[0] ?? '';
    const match = token.match(WHISPER_LANGUAGE_TOKEN_PATTERN);
    if (!match) continue;

    const logit = logits[offset] ?? Number.NEGATIVE_INFINITY;
    if (logit > bestLogit) {
      bestLogit = logit;
      bestLanguage = match[1];
    }
  }

  return bestLanguage;
}
