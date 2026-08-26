import { MedAsrTextTokenizer } from '../lasr-ctc/tokenizer.js';

/** GigaAM v3 RNN-T uses an implicit blank immediately after its character vocabulary. */
export class GigaAmRnntTokenizer extends MedAsrTextTokenizer {
  static override fromText(text: string, blankId = 34): GigaAmRnntTokenizer {
    const parsed = text.split(/\r?\n/).filter((line) => line.trim()).map((line) => {
      const separator = line.lastIndexOf(' ');
      return separator > 0 ? { token: line.slice(0, separator), id: Number.parseInt(line.slice(separator + 1), 10) } : undefined;
    }).filter((entry): entry is { token: string; id: number } => Boolean(entry && Number.isFinite(entry.id)));
    const idToToken: string[] = [];
    for (const entry of parsed) idToToken[entry.id] = entry.token;
    if (!idToToken[blankId]) idToToken[blankId] = '<blk>';
    return new GigaAmRnntTokenizer(idToToken);
  }

  static override async fromUrl(url: string): Promise<GigaAmRnntTokenizer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch GigaAM RNN-T vocabulary at "${url}".`);
    return GigaAmRnntTokenizer.fromText(await response.text());
  }
}
