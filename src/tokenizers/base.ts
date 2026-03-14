export type TokenizerKind =
  | 'sentencepiece'
  | 'wordpiece'
  | 'aggregate'
  | 'char'
  | 'tiktoken'
  | 'bpe'
  | 'custom';

export interface TokenizerSpec {
  readonly kind: TokenizerKind;
  readonly vocabSize?: number;
  readonly blankTokenId?: number;
  readonly bosTokenId?: number;
  readonly eosTokenId?: number;
  readonly unkTokenId?: number;
  readonly padTokenId?: number;
}

export interface TextTokenizer {
  readonly kind: TokenizerKind;
  readonly vocabSize?: number;
  decode(ids: readonly number[]): string;
  encode?(text: string): readonly number[];
  idsToTokens?(ids: readonly number[]): readonly string[];
  tokensToIds?(tokens: readonly string[]): readonly number[];
}
