import { MedAsrTextTokenizer } from '../lasr-ctc/tokenizer.js';

/** SentencePiece vocabulary reader used by the SenseVoice ONNX export. */
export class SenseVoiceTokenizer extends MedAsrTextTokenizer {
  static override fromText(text: string): SenseVoiceTokenizer {
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const idToToken: string[] = [];
    for (const line of lines) {
      const separator = line.lastIndexOf(' ');
      if (separator <= 0) continue;
      const token = line.slice(0, separator);
      const id = Number.parseInt(line.slice(separator + 1), 10);
      if (Number.isInteger(id) && id >= 0 && token.length > 0) idToToken[id] = token;
    }
    return new SenseVoiceTokenizer(idToToken);
  }

  static override async fromUrl(url: string): Promise<SenseVoiceTokenizer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch SenseVoice vocabulary at "${url}".`);
    return SenseVoiceTokenizer.fromText(await response.text());
  }
}
