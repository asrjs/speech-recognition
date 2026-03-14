import type { TextTokenizer, TokenizerKind } from './base.js';

export class StubTextTokenizer implements TextTokenizer {
  readonly vocabSize?: number;

  constructor(
    readonly kind: TokenizerKind = 'sentencepiece',
    private readonly prefix = 'tok',
    vocabSize?: number
  ) {
    this.vocabSize = vocabSize;
  }

  decode(ids: readonly number[]): string {
    return ids.map((id) => `${this.prefix}${id}`).join(' ');
  }

  encode(text: string): readonly number[] {
    return text
      .split(/\s+/)
      .filter((part) => part.length > 0)
      .map((_, index) => index + 1);
  }
}

export class StubSentencePieceTokenizer extends StubTextTokenizer {
  constructor(vocabSize?: number) {
    super('sentencepiece', 'sp', vocabSize);
  }
}
