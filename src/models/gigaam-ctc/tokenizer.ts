import { MedAsrTextTokenizer, readTokenizerSourceText } from '../lasr-ctc/tokenizer.js';
import type { AssetAbortSignalLike } from '../../io/abort.js';

const OFFICIAL_BLANK_TOKEN = '<blk>';

/** GigaAM's character vocabulary uses a final CTC blank that is not a spoken token. */
export class GigaAmTokenizer extends MedAsrTextTokenizer {
  static fromVocabulary(vocab: readonly string[]): GigaAmTokenizer {
    return new GigaAmTokenizer([...vocab, OFFICIAL_BLANK_TOKEN]);
  }

  static override fromText(text: string): GigaAmTokenizer {
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
        Boolean(entry && Number.isFinite(entry.id) && entry.token.length > 0),
      );
    const idToToken: string[] = [];
    for (const entry of parsed) idToToken[entry.id] = entry.token;
    if (!idToToken.includes(OFFICIAL_BLANK_TOKEN)) {
      idToToken.push(OFFICIAL_BLANK_TOKEN);
    }
    return new GigaAmTokenizer(idToToken);
  }

  static override async fromUrl(
    url: string,
    signal?: AssetAbortSignalLike | null,
  ): Promise<GigaAmTokenizer> {
    return GigaAmTokenizer.fromText(
      await readTokenizerSourceText(url, signal, `Failed to fetch GigaAM vocabulary at "${url}".`),
    );
  }

  override decode(ids: readonly number[]): string {
    let text = '';
    for (const id of ids) {
      if (!Number.isFinite(id) || id === this.blankId || id < 0 || id >= this.idToToken.length) {
        continue;
      }
      const token = this.idToToken[id];
      if (!token || token === OFFICIAL_BLANK_TOKEN) {
        continue;
      }
      text += token === '\u2581' ? ' ' : token;
    }
    return text;
  }
}
