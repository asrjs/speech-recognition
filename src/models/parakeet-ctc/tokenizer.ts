import { MedAsrTextTokenizer } from '../lasr-ctc/tokenizer.js';
import type { AssetAbortSignalLike } from '../../io/abort.js';

/**
 * ParakeetForCTC ships a sentencepiece vocabulary whose final id is the CTC
 * pad/blank (`<pad>`, 1024 for the 110M checkpoint), not a `<blk>` token, so the
 * generic blank lookup would resolve the wrong id.
 */
const PAD_TOKEN = '<pad>';

export class ParakeetCtcTokenizer extends MedAsrTextTokenizer {
  /** Loads the HuggingFace fast-tokenizer `model.vocab` mapping in id order. */
  static fromTokenizerJsonText(text: string): ParakeetCtcTokenizer {
    const parsed = JSON.parse(text) as {
      model?: { vocab?: Record<string, number> | ReadonlyArray<readonly [string, number]> };
      vocab?: Record<string, number> | ReadonlyArray<readonly [string, number]>;
      added_tokens?: ReadonlyArray<{ readonly id: number; readonly content: string }>;
    };
    const vocab = parsed.model?.vocab ?? parsed.vocab;
    if (!vocab) {
      throw new Error('Parakeet CTC tokenizer.json has no model.vocab mapping.');
    }

    const idToToken: string[] = [];
    if (Array.isArray(vocab)) {
      for (let index = 0; index < vocab.length; index += 1) {
        const token = vocab[index]?.[0];
        if (typeof token === 'string') {
          idToToken[index] = token;
        }
      }
    } else {
      for (const [token, id] of Object.entries(vocab)) {
        if (Number.isInteger(id) && id >= 0) {
          idToToken[id] = token;
        }
      }
    }
    // The CTC pad lives in added_tokens, outside model.vocab, so a vocabulary
    // built from model.vocab alone would stop one id short of the model's
    // vocab_size and mis-resolve the blank.
    for (const added of parsed.added_tokens ?? []) {
      if (Number.isInteger(added.id) && added.id >= 0 && typeof added.content === 'string') {
        idToToken[added.id] = added.content;
      }
    }

    if (idToToken.length === 0) {
      throw new Error('Parakeet CTC tokenizer.json produced an empty vocabulary.');
    }

    return new ParakeetCtcTokenizer(idToToken);
  }

  static override async fromUrl(
    url: string,
    signal?: AssetAbortSignalLike | null,
  ): Promise<ParakeetCtcTokenizer> {
    const response = await fetch(url, { signal: signal as AbortSignal | undefined });
    if (!response.ok) {
      throw new Error(`Failed to fetch Parakeet CTC tokenizer at "${url}": ${response.status}.`);
    }

    return ParakeetCtcTokenizer.fromTokenizerJsonText(await response.text());
  }

  /** The CTC blank is this model's pad id, not a `<blk>` entry. */
  get padTokenId(): number {
    const index = this.idToToken.indexOf(PAD_TOKEN);
    return index >= 0 ? index : this.idToToken.length - 1;
  }

  override decode(ids: readonly number[]): string {
    const padId = this.padTokenId;
    let text = '';
    for (const id of ids) {
      if (!Number.isFinite(id) || id === padId || id < 0 || id >= this.idToToken.length) {
        continue;
      }
      if (this.idToToken[id] === PAD_TOKEN) {
        continue;
      }
      // Reuse the base sentencepiece normalization so U+2581 word markers become
      // spaces and special tokens drop out.
      text += this.decodeTokenPiece(id);
    }

    return text;
  }
}
