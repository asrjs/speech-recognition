/** Character-level CTC tokenizer for Wav2Vec2 models. */

const WORD_SEPARATOR_TOKEN = '|';

export class Wav2Vec2CharTokenizer {
  readonly kind = 'char' as const;
  readonly vocabSize: number;
  readonly blankId: number;

  private readonly idToToken: readonly string[];
  private readonly tokenToId: ReadonlyMap<string, number>;

  constructor(vocab: Record<string, number>) {
    // Build reverse mapping: id → token string
    const maxId = Math.max(...Object.values(vocab));
    const idToToken = new Array<string>(maxId + 1).fill('');
    const tokenToId = new Map<string, number>();

    for (const [token, id] of Object.entries(vocab)) {
      idToToken[id] = token;
      tokenToId.set(token, id);
    }

    this.idToToken = idToToken;
    this.tokenToId = tokenToId;
    this.vocabSize = idToToken.length;

    // Blank ID: <pad> token, fallback to 0
    const padId = tokenToId.get('<pad>');
    this.blankId = padId !== undefined ? padId : 0;
  }

  /**
   * Decode a single token ID to its string piece.
   * Returns '' for blank and special tokens.
   */
  decodeTokenPiece(tokenId: number): string {
    if (!Number.isFinite(tokenId) || tokenId < 0 || tokenId >= this.idToToken.length) {
      return '';
    }

    if (tokenId === this.blankId) {
      return '';
    }

    const rawToken = this.idToToken[tokenId];
    if (!rawToken || rawToken.startsWith('<')) {
      return '';
    }

    // The | token is the word separator — it maps to a space in output.
    if (rawToken === WORD_SEPARATOR_TOKEN) {
      return ' ';
    }

    return rawToken;
  }

  /**
   * Decode an array of CTC-collapsed token IDs to text.
   * This assumes CTC collapse (blank removal + deduplication) has already been done.
   */
  decode(ids: readonly number[]): string {
    const pieces: string[] = [];

    for (const id of ids) {
      const piece = this.decodeTokenPiece(id);
      if (piece.length > 0) {
        pieces.push(piece);
      }
    }

    // Wav2Vec2 outputs uppercase characters; lowercase the final text.
    let text = pieces.join('');
    text = text.toLowerCase();
    // Collapse multiple spaces and trim.
    text = text.replace(/\s+/g, ' ').trim();
    return text;
  }

  /**
   * Encode text to token IDs (for alignment / testing use cases).
   * Text is uppercased; spaces are mapped to the | separator token.
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    const upper = text.toUpperCase();

    for (let i = 0; i < upper.length; i += 1) {
      const char = upper[i] ?? '';

      if (char === ' ') {
        const sepId = this.tokenToId.get(WORD_SEPARATOR_TOKEN);
        if (sepId !== undefined) {
          ids.push(sepId);
        }
        continue;
      }

      const id = this.tokenToId.get(char);
      if (id !== undefined) {
        ids.push(id);
      } else {
        // Map unknown characters to <unk>
        const unkId = this.tokenToId.get('<unk>');
        if (unkId !== undefined) {
          ids.push(unkId);
        }
      }
    }

    return ids;
  }

  /**
   * Load tokenizer from a vocab.json URL (HF format: { token: id, ... }).
   */
  static async fromUrl(url: string): Promise<Wav2Vec2CharTokenizer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch Wav2Vec2 vocabulary at "${url}" (${response.status}).`);
    }

    const vocab = (await response.json()) as Record<string, number>;
    return new Wav2Vec2CharTokenizer(vocab);
  }
}
