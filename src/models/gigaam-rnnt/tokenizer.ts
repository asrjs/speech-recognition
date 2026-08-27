import { MedAsrTextTokenizer, readTokenizerSourceText } from '../lasr-ctc/tokenizer.js';
import type { AssetAbortSignalLike } from '../../io/abort.js';

const BLANK_TOKEN = '<blk>';

/** Official v3 E2E RNN-T SentencePiece pieces plus an implicit blank at vocab length. */
export class GigaAmRnntTokenizer extends MedAsrTextTokenizer {
  static override fromText(text: string, blankId?: number): GigaAmRnntTokenizer {
    const parsed = text
      .split(/\r?\n/)
      .filter((line) => line.trim().length > 0)
      .map((line) => {
        const separator = line.lastIndexOf(' ');
        return separator > 0
          ? { token: line.slice(0, separator), id: Number.parseInt(line.slice(separator + 1), 10) }
          : undefined;
      })
      .filter((entry): entry is { token: string; id: number } =>
        Boolean(entry && Number.isFinite(entry.id)),
      );
    const idToToken: string[] = [];
    for (const entry of parsed) idToToken[entry.id] = entry.token;
    const resolvedBlank = blankId ?? idToToken.findIndex((token) => token === BLANK_TOKEN);
    const blank = resolvedBlank >= 0 ? resolvedBlank : idToToken.length;
    if (!idToToken[blank]) idToToken[blank] = BLANK_TOKEN;
    return new GigaAmRnntTokenizer(idToToken);
  }

  static override async fromUrl(
    url: string,
    signal?: AssetAbortSignalLike | null,
  ): Promise<GigaAmRnntTokenizer> {
    return GigaAmRnntTokenizer.fromText(
      await readTokenizerSourceText(url, signal, `Failed to fetch GigaAM RNN-T vocabulary at "${url}".`),
    );
  }

  override decode(ids: readonly number[]): string {
    let text = '';
    for (const id of ids) {
      if (!Number.isFinite(id) || id === this.blankId || id < 0 || id >= this.idToToken.length) continue;
      const token = this.idToToken[id];
      if (!token || token === BLANK_TOKEN) continue;
      text += token.replace(/\u2581/g, ' ');
    }
    return text.replace(/\s+/g, ' ').trim();
  }
}
