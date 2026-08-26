import { MedAsrTextTokenizer } from '../lasr-ctc/tokenizer.js';

/** GigaAM's character vocabulary uses a final <blk> CTC token. */
export class GigaAmTokenizer extends MedAsrTextTokenizer {
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
    return new GigaAmTokenizer(idToToken);
  }

  static override async fromUrl(url: string): Promise<GigaAmTokenizer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch GigaAM vocabulary at "${url}".`);
    return GigaAmTokenizer.fromText(await response.text());
  }
}
