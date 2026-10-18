/**
 * Moonshine decoder session.
 *
 * Architecture, as measured during the port:
 *
 * - Cross-attention key and value are a fixed projection of the encoder states,
 *   so they are computed once by a small graph and stay constant for the whole
 *   utterance. Utterance length is therefore not pinned to any exported seed.
 * - Self-attention history starts empty, exactly like PyTorch's `DynamicCache`:
 *   the decoder creates the BOS entry itself.
 * - Past state crosses the graph boundary as two stacked tensors, `self_past`
 *   and `cross_past`, each `[slots, batch, heads, length, headDim]`. Slots are
 *   layer-major with key and value adjacent, so slot `2 * layer` is the key.
 *
 * The projection emits `[batch, frames, heads * headDim]` while the cache wants
 * `[batch, heads, frames, headDim]`. That is a transpose, not a reinterpret:
 * reshaping alone keeps the element count and silently scrambles the data, which
 * surfaces as a repeating token loop rather than an error.
 */

export interface MoonshineSessionOptions {
  /** ORT module, used to build input tensors. */
  readonly ort: {
    readonly Tensor: new (type: string, data: ArrayBufferView, dims: readonly number[]) => unknown;
  };
  readonly encoder: {
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  /** Projects encoder states into per-layer cross-attention key and value. */
  readonly crossKv: {
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  readonly decoder: {
    readonly inputNames: readonly string[];
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  readonly tokenizer: MoonshineTokenizer;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  /** Past-state geometry, read from the export receipt rather than hardcoded. */
  readonly slots: number;
  readonly heads: number;
  readonly headDim: number;
  readonly maxTokens?: number;
  /**
   * Bids on tokens emitted within the last N steps are discarded outright.
   *
   * Greedy decoding on audio beyond roughly 30 seconds falls into a repetition
   * attractor: the transcript cycles through a handful of tokens until the cap.
   * A soft repetition penalty was measured and changed nothing at 1.1, so this is
   * a hard ban rather than a logit scale. With a window of 40 the same clip
   * decoded to coherent LibriVox text with every token distinct.
   *
   * End of sequence is never suppressed because it is never appended to the
   * emitted ids, so a ban can never trap the decoder in a non-terminating loop.
   */
  readonly repetitionBan?: number;
}

export interface MoonshineDecodeResult {
  readonly text: string;
  /** Actual emitted token IDs when the execution path exposes them. */
  readonly tokenIds?: readonly number[];
  /** Actual token count when IDs are not exposed by a streaming decoder. */
  readonly tokenCount?: number;
  readonly encoderFrames: number;
  readonly steps: number;
}

interface TensorLike {
  readonly dims: readonly [number, number, number, ...number[]];
  readonly data: Float32Array;
}

/** Reorders `[batch, frames, heads * headDim]` into `[batch, heads, frames, headDim]`. */
function toCacheLayout(tensor: TensorLike, heads: number): Float32Array {
  const [batch, frames, hidden] = tensor.dims;
  const headDim = hidden / heads;
  const out = new Float32Array(batch * heads * frames * headDim);
  let cursor = 0;
  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let f = 0; f < frames; f += 1) {
        const base = b * frames * hidden + f * hidden + h * headDim;
        for (let d = 0; d < headDim; d += 1) {
          out[cursor] = tensor.data[base + d] ?? 0;
          cursor += 1;
        }
      }
    }
  }

  return out;
}

export class MoonshineSession {
  private readonly options: MoonshineSessionOptions;

  constructor(options: MoonshineSessionOptions) {
    this.options = options;
  }

  private static asTensors(
    result: Record<string, unknown> | unknown[] | Promise<Record<string, unknown> | unknown[]>,
    names: readonly string[],
  ): TensorLike[] {
    const settled = result instanceof Promise ? undefined : result;
    if (Array.isArray(settled)) {
      return settled as TensorLike[];
    }

    return names.map((name) => (settled as Record<string, unknown>)[name] as TensorLike);
  }

  /** Cross-attention memory, constant for the whole utterance. */
  private async crossPast(states: TensorLike): Promise<unknown> {
    const { crossKv, slots, heads, headDim } = this.options;
    const frames = states.dims[1];
    const projected = MoonshineSession.asTensors(
      await crossKv.run({ encoder_states: this.tensor('float32', states.data, states.dims) }),
      crossKv.outputNames,
    );
    const block = frames * heads * headDim;
    const data = new Float32Array(slots * block);
    for (let slot = 0; slot < slots && slot < projected.length; slot += 1) {
      data.set(toCacheLayout(projected[slot]!, heads), slot * block);
    }

    return this.tensor('float32', data, [slots, 1, heads, frames, headDim]);
  }

  private tensor(type: string, data: Float32Array, dims: readonly number[]): unknown {
    return new this.options.ort.Tensor(type, data, dims);
  }

  private async initialFeeds(states: TensorLike): Promise<Record<string, unknown>> {
    const { decoder, bosTokenId, slots, heads, headDim } = this.options;
    const feeds: Record<string, unknown> = {
      self_past: this.tensor('float32', new Float32Array(0), [slots, 1, heads, 0, headDim]),
      cross_past: await this.crossPast(states),
      // Moonshine's attention skips the whole cross-attention block when
      // encoder_hidden_states is absent, so this input is what opens the branch.
      // It is a gate rather than a data path: the graph reads key and value from
      // cross_past, which stays constant for the utterance.
      encoder_states: this.tensor('float32', states.data, states.dims),
    };
    if (decoder.inputNames.includes('input_ids')) {
      feeds['input_ids'] = this.tensor(
        'int64',
        BigInt64Array.from([BigInt(bosTokenId)]) as unknown as Float32Array,
        [1, 1],
      );
    }

    return feeds;
  }

  async transcribe(pcm: Float32Array): Promise<MoonshineDecodeResult> {
    const { encoder, decoder } = this.options;
    const encoderOut = await encoder.run({
      input_values: this.tensor('float32', pcm, [1, pcm.length]),
    });
    const states = MoonshineSession.asTensors(encoderOut, encoder.outputNames)[0]!;
    const frames = states.dims[1];

    let feeds = await this.initialFeeds(states);
    const tokenIds: number[] = [];
    const limit = this.options.maxTokens ?? 256;
    const ban = this.options.repetitionBan ?? 0;
    let steps = 0;

    for (; steps < limit; steps += 1) {
      const result = await decoder.run(feeds);
      const outputs = MoonshineSession.asTensors(result, decoder.outputNames);
      const logits = outputs[0]!;
      const banned = ban > 0 ? new Set(tokenIds.slice(-ban)) : null;
      let bestId = 0;
      let best = Number.NEGATIVE_INFINITY;
      for (let index = 0; index < logits.data.length; index += 1) {
        if (banned?.has(index)) {
          continue;
        }
        const value = logits.data[index] ?? Number.NEGATIVE_INFINITY;
        if (value > best) {
          best = value;
          bestId = index;
        }
      }
      if (bestId === this.options.eosTokenId) {
        steps += 1;
        break;
      }
      tokenIds.push(bestId);

      const next: Record<string, unknown> = {
        input_ids: this.tensor(
          'int64',
          BigInt64Array.from([BigInt(bestId)]) as unknown as Float32Array,
          [1, 1],
        ),
        // Cross-attention memory stays constant; only self-attention grows.
        cross_past: feeds['cross_past'],
        encoder_states: feeds['encoder_states'],
        self_past: outputs[1],
      };
      feeds = next;
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

  /** Accepts either a token-to-id list or an id-to-token map. */
  static fromTokenizerJson(
    tokenizer: { model?: { vocab?: Record<string, number> }; vocab?: Record<string, number> },
    unkTokenId: number,
  ): MoonshineTokenizer {
    const source = tokenizer.model?.vocab ?? tokenizer.vocab ?? {};
    if (Array.isArray(source)) {
      return new MoonshineTokenizer(source as unknown as string[], unkTokenId);
    }
    const entries = Object.entries(source).map(([token, id]) => [Number(id), token] as const);
    entries.sort((a, b) => a[0] - b[0]);

    return new MoonshineTokenizer(
      entries.map(([, token]) => token),
      unkTokenId,
    );
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

    return text.trim();
  }
}
