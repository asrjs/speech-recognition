import type { TextTokenizer } from '../../tokenizers/index.js';

/** Decoder for the token.txt format emitted by icefall/sherpa-onnx. */
export class XAsrTokenizer implements TextTokenizer {
  readonly kind = 'bpe' as const;
  readonly blankId = 0;

  private constructor(private readonly idToToken: readonly string[]) {}

  static fromText(text: string): XAsrTokenizer {
    const tokens: string[] = [];
    for (const line of text.split(/\r?\n/)) {
      const match = line.trim().match(/^(.*)\s+(-?\d+)$/);
      if (!match) continue;
      const id = Number(match[2]);
      if (Number.isInteger(id) && id >= 0) tokens[id] = match[1] ?? '';
    }
    return new XAsrTokenizer(tokens);
  }

  static async fromUrl(url: string): Promise<XAsrTokenizer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch X-ASR tokens at "${url}".`);
    return XAsrTokenizer.fromText(await response.text());
  }

  decode(ids: readonly number[]): string {
    return ids
      .map((id) => this.idToToken[id] ?? '')
      .filter((token) => token !== '<blk>' && token !== '<eps>' && !token.startsWith('<'))
      .join('')
      .replace(/▁/g, ' ')
      .replace(/^\s+/, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  decodeTokenPiece(id: number): string {
    const token = this.idToToken[id] ?? '';
    if (!token || token === '<blk>' || token === '<eps>' || token.startsWith('<')) return '';
    return token.replace(/▁/g, ' ');
  }

}
