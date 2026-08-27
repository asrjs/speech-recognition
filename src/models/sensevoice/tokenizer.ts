import { MedAsrTextTokenizer, readTokenizerSourceText } from '../lasr-ctc/tokenizer.js';
import type { AssetAbortSignalLike } from '../../io/abort.js';

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

  static override async fromUrl(
    url: string,
    signal?: AssetAbortSignalLike | null,
  ): Promise<SenseVoiceTokenizer> {
    return SenseVoiceTokenizer.fromText(
      await readTokenizerSourceText(url, signal, `Failed to fetch SenseVoice vocabulary at "${url}".`),
    );
  }
}
