import type { TextTokenizer } from '../../tokenizers/index.js';

function parseTokensText(text: string): string[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const idToToken: string[] = [];

  for (const line of lines) {
    const separatorIndex = line.lastIndexOf(' ');
    if (separatorIndex <= 0) {
      continue;
    }

    const token = line.slice(0, separatorIndex);
    const id = Number.parseInt(line.slice(separatorIndex + 1), 10);
    if (!Number.isFinite(id) || token.length === 0) {
      continue;
    }

    idToToken[id] = token;
  }

  return idToToken;
}

function normalizeSentencePieceToken(token: string): string {
  return token.replace(/\u2581/g, ' ');
}

const BLANK_TOKEN_CANDIDATES = ['<blk>', '<epsilon>', '<blank>', '<pad>'] as const;

export class MedAsrTextTokenizer implements TextTokenizer {
  readonly kind = 'sentencepiece' as const;
  readonly vocabSize: number;
  readonly blankId: number;
  readonly blankToken: string;
  private readonly normalizedTokens: readonly string[];

  constructor(readonly idToToken: readonly string[]) {
    this.vocabSize = idToToken.length;

    let resolvedBlankId = -1;
    let resolvedBlankToken = '<epsilon>';
    for (const candidate of BLANK_TOKEN_CANDIDATES) {
      const candidateId = idToToken.findIndex((token) => token === candidate);
      if (candidateId >= 0) {
        resolvedBlankId = candidateId;
        resolvedBlankToken = candidate;
        break;
      }
    }

    this.blankId = resolvedBlankId >= 0 ? resolvedBlankId : 0;
    this.blankToken = resolvedBlankToken;
    this.normalizedTokens = idToToken.map((token) =>
      typeof token === 'string' ? normalizeSentencePieceToken(token) : '',
    );
  }

  static fromText(text: string): MedAsrTextTokenizer {
    return new MedAsrTextTokenizer(parseTokensText(text));
  }

  static async fromUrl(tokensUrl: string): Promise<MedAsrTextTokenizer> {
    const response = await fetch(tokensUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch tokenizer vocabulary at "${tokensUrl}".`);
    }

    const text = await response.text();
    return MedAsrTextTokenizer.fromText(text);
  }

  isSpecialToken(token: string): boolean {
    return /^<[^>]+>$/.test(token);
  }

  decodeTokenPiece(tokenId: number): string {
    if (!Number.isFinite(tokenId) || tokenId < 0 || tokenId >= this.idToToken.length) {
      return '';
    }

    if (tokenId === this.blankId) {
      return '';
    }

    const rawToken = this.idToToken[tokenId];
    if (!rawToken || this.isSpecialToken(rawToken)) {
      return '';
    }

    return this.normalizedTokens[tokenId] ?? '';
  }

  decode(ids: readonly number[]): string {
    const pieces: string[] = [];

    for (const id of ids) {
      const piece = this.decodeTokenPiece(id);
      if (piece.length > 0) {
        pieces.push(piece);
      }
    }

    let text = pieces.join('');
    text = text.replace(/^\s+/, '');
    text = text.replace(/\s+([,.;:!?%)\]\}])/g, '$1');
    text = text.replace(/([\[\(\{])\s+/g, '$1');
    text = text.replace(/\s+/g, ' ');
    return text.trim();
  }
}
