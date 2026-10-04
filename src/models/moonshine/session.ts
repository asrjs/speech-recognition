/** Moonshine decoder inputs, named as the exported graph declares them. */
const SELF_ATTENTION = /\.decoder\./;

export interface MoonshineSessionOptions {
  /** ORT module, used to build input tensors. */
  readonly ort: {
    readonly Tensor: new (
      type: string,
      data: ArrayBufferView | BigInt64Array,
      dims: readonly number[],
    ) => unknown;
  };
  readonly encoder: {
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
    readonly outputNames?: readonly string[];
  };
  readonly decoder: {
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    readonly inputMetadata: unknown;
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  readonly tokenizer: MoonshineTokenizer;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  /** Nested `[batch][heads][length][headDim]` seed arrays, keyed by graph input name. */
  readonly seedCache: Record<string, SeedTensor>;
  readonly maxTokens?: number;
  /**
   * Bids on tokens emitted within the last N steps are discarded outright.
   *
   * Greedy decoding on audio beyond roughly 30 seconds falls into a repetition
   * attractor: the transcript cycles through a handful of tokens until the cap.
   * A soft repetition penalty was measured and changed nothing at 1.1, so this
   * is a hard ban rather than a logit scale. With a window of 40 the same clip
   * decoded to coherent LibriVox text with every token distinct.
   *
   * The evidence was measured on the seed-free cross-attention runner, not on
   * this seed-cache session, which still ships the retired architecture. The
   * ban semantics are identical; only the evidence trail is from the runner.
   *
   * End of sequence is never suppressed because it is never appended to
   * `tokenIds`, so a ban can never trap the decoder in a non-terminating loop.
   */
  readonly repetitionBan?: number;
}

export interface MoonshineDecodeResult {
  readonly text: string;
  readonly tokenIds: readonly number[];
  readonly encoderFrames: number;
  readonly steps: number;
}

/** Nested `[batch][heads][length][headDim]` seed values. */
export type SeedTensor = number[][][][];

interface TensorLike {
  readonly dims: readonly number[];
  readonly data: ArrayLike<number>;
}

/**
 * Moonshine runs an encoder once and then an autoregressive decoder over the
 * encoder states.
 *
 * Two facts were measured against the exported graph and drive this code. The
 * graph returns self-attention history only, so cross-attention key and value
 * stay constant for the utterance and come from the checkpoint seed cache. The
 * seed is mapped by input name because the checkpoint cache order differs from
 * the graph's own names.
 */
export class MoonshineSession {
  private readonly options: MoonshineSessionOptions;
  constructor(options: MoonshineSessionOptions) {
    this.options = options;
  }

  private seedTensor(name: string): unknown {
    const seed = this.options.seedCache[name];
    if (!seed) {
      throw new Error(`Moonshine seed cache is missing ${name}.`);
    }
    const { Tensor } = this.options.ort;
    const batch = seed.length > 0 ? seed.length : 1;
    const heads = seed[0]?.length ?? 1;
    const length = seed[0]?.[0]?.length ?? 1;
    const headDim = seed[0]?.[0]?.[0]?.length ?? 36;
    const flat = new Float32Array(batch * heads * length * headDim);
    let offset = 0;
    for (let b = 0; b < batch; b += 1) {
      for (let h = 0; h < heads; h += 1) {
        for (let t = 0; t < length; t += 1) {
          for (let d = 0; d < headDim; d += 1) {
            flat[offset] = seed[b]![h]![t]![d] ?? 0;
            offset += 1;
          }
        }
      }
    }

    return new Tensor('float32', flat, [batch, heads, length, headDim]);
  }

  private initialFeeds(frames: number): Record<string, unknown> {
    const { decoder, seedCache } = this.options;
    const { Tensor } = this.options.ort;
    const feeds: Record<string, unknown> = {};
    for (const name of decoder.inputNames) {
      if (name === 'input_ids') {
        feeds[name] = new Tensor(
          'int64',
          BigInt64Array.from([BigInt(this.options.bosTokenId)]),
          [1, 1],
        );
        continue;
      }
      const seed = seedCache[name];
      if (seed) {
        feeds[name] = this.seedTensor(name);
      } else {
        // Cross-attention entries without a seed fall back to the frame count.
        feeds[name] = new Tensor('float32', new Float32Array(frames * 8 * 36), [1, 8, frames, 36]);
      }
    }

    return feeds;
  }

  async transcribe(pcm: Float32Array): Promise<MoonshineDecodeResult> {
    const { encoder, decoder } = this.options;
    const encoderOut = await encoder.run({
      input_values: new this.options.ort.Tensor('float32', pcm, [1, pcm.length]),
    });
    const states = (
      Array.isArray(encoderOut)
        ? encoderOut[0]
        : (encoderOut as Record<string, unknown>)[encoder.outputNames?.[0] ?? 'last_hidden_state']
    ) as TensorLike;
    const frames = states.dims[1] ?? 0;

    const feeds = this.initialFeeds(frames);
    const tokenIds: number[] = [];
    const limit = this.options.maxTokens ?? 256;
    let steps = 0;
    let current: Record<string, unknown> = feeds;

    for (; steps < limit; steps += 1) {
      const result = await decoder.run(current);
      const mapped = Array.isArray(result)
        ? { [decoder.outputNames[0] ?? 'logits']: result[0] }
        : (result as Record<string, unknown>);
      const logits = mapped.logits as TensorLike | undefined;
      if (!logits) {
        throw new Error('Moonshine decoder returned no logits.');
      }
      let bestId = 0;
      let best = Number.NEGATIVE_INFINITY;
      const ban = this.options.repetitionBan ?? 0;
      const recent = ban > 0 ? new Set(tokenIds.slice(-ban)) : null;
      for (let i = 0; i < logits.data.length; i += 1) {
        if (recent?.has(i)) {
          continue;
        }
        const value = logits.data[i] ?? Number.NEGATIVE_INFINITY;
        if (value > best) {
          best = value;
          bestId = i;
        }
      }
      if (bestId === this.options.eosTokenId) {
        steps += 1;
        break;
      }
      tokenIds.push(bestId);

      const present = Array.isArray(result)
        ? new Map(decoder.outputNames.map((name, index) => [name, result[index]]))
        : new Map(Object.entries(mapped));
      const next: Record<string, unknown> = {};
      for (const name of decoder.inputNames) {
        if (name === 'input_ids') {
          next[name] = new this.options.ort.Tensor(
            'int64',
            BigInt64Array.from([BigInt(bestId)]),
            [1, 1],
          );
          continue;
        }
        if (SELF_ATTENTION.test(name)) {
          const layer = name.split('.')[1];
          const slot = name.split('.')[3];
          next[name] = present.get(`present.${layer}.decoder.${slot}`) ?? current[name];
          continue;
        }
        // Cross-attention history is constant; reuse the seeded tensor.
        next[name] = current[name];
      }
      current = next;
    }

    return {
      text: this.options.tokenizer.decode(tokenIds),
      tokenIds,
      encoderFrames: frames,
      steps,
    };
  }
}

/** SentencePiece tokenizer for the Moonshine vocabulary. */
export class MoonshineTokenizer {
  readonly vocabSize: number;
  private readonly idToToken: readonly string[];

  constructor(
    idToToken: readonly string[],
    readonly unkTokenId: number,
  ) {
    this.idToToken = idToToken;
    this.vocabSize = idToToken.length;
  }

  static fromTokenizerJson(
    tokenizer: { model?: { vocab?: Record<string, number> }; vocab?: Record<string, number> },
    unkTokenId: number,
  ): MoonshineTokenizer {
    const vocab = tokenizer.model?.vocab ?? tokenizer.vocab ?? {};
    const highest = Object.values(vocab).reduce((max, id) => (id > max ? id : max), 0);
    const idToToken: string[] = new Array<string>(highest + 1).fill('');
    for (const [token, id] of Object.entries(vocab)) {
      idToToken[id] = token;
    }

    return new MoonshineTokenizer(idToToken, unkTokenId);
  }

  decodeTokenPiece(tokenId: number): string {
    const token = this.idToToken[tokenId];
    if (!token || token === '<unk>' || token === '<pad>') {
      return '';
    }

    return token.replace(/▁/g, ' ');
  }

  decode(ids: readonly number[]): string {
    let text = '';
    for (const id of ids) {
      text += this.decodeTokenPiece(id);
    }

    return text;
  }
}
