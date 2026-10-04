import type { CtcTokenizerLike } from '../../ctc/types.js';

const BLANK_TOKEN = '<s>';

/**
 * Vocabulary for the omnilingual CTC models.
 *
 * `tokens.txt` holds `<token> <id>` lines, id 0 is `<s>` (the CTC blank), and
 * the vocabulary stores real spaces rather than the SentencePiece U+2581
 * marker, so tokens concatenate directly with no separator and no replacement.
 */
export class OmnilingualCtcTokenizer implements CtcTokenizerLike {
  readonly vocabSize: number;
  readonly blankId: number;
  private readonly idToToken: readonly string[];

  constructor(idToToken: readonly string[]) {
    this.idToToken = idToToken;
    this.vocabSize = idToToken.length;
    const blank = idToToken.findIndex((token) => token === BLANK_TOKEN);
    this.blankId = blank >= 0 ? blank : 0;
  }

  /** Parses `<token> <id>` lines. A bare id with no token keeps an empty piece. */
  static fromText(text: string): OmnilingualCtcTokenizer {
    const maxId = 0;
    const entries: Array<[number, string]> = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.replace(/\s+$/, '');
      if (line.length === 0) continue;
      const separator = line.lastIndexOf(' ');
      if (separator <= 0) continue;
      const token = line.slice(0, separator);
      const id = Number.parseInt(line.slice(separator + 1), 10);
      if (!Number.isFinite(id) || id < 0) continue;
      entries.push([id, token]);
      if (id > maxId) {
        entries[entries.length - 1] = [id, token];
      }
    }
    if (entries.length === 0) {
      throw new Error('Omnilingual CTC tokens.txt produced an empty vocabulary.');
    }

    const highest = entries.reduce((max, [id]) => (id > max ? id : max), 0);
    const idToToken: string[] = new Array<string>(highest + 1).fill('');
    for (const [id, token] of entries) {
      idToToken[id] = token;
    }

    return new OmnilingualCtcTokenizer(idToToken);
  }

  static async fromUrl(url: string, signal?: AbortSignal | null): Promise<OmnilingualCtcTokenizer> {
    const response = await fetch(url, { signal: signal ?? undefined });
    if (!response.ok) {
      throw new Error(
        `Failed to fetch omnilingual CTC vocabulary at "${url}": ${response.status}.`,
      );
    }

    return OmnilingualCtcTokenizer.fromText(await response.text());
  }

  decodeTokenPiece(tokenId: number): string {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= this.idToToken.length) {
      return '';
    }
    const token = this.idToToken[tokenId] ?? '';
    if (token === BLANK_TOKEN || token === '<pad>' || token === '<unk>' || token === '</s>') {
      return '';
    }

    return token;
  }

  decode(ids: readonly number[]): string {
    let text = '';
    for (const id of ids) {
      text += this.decodeTokenPiece(id);
    }

    return text;
  }
}
