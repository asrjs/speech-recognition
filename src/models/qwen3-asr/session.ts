/**
 * Qwen3-ASR decode session.
 *
 * Architecture, as measured during the port:
 *
 * - The audio front takes 128-bin mel features and a 2-D feature mask and emits
 *   one embedding per audio frame. The two inputs have different ranks, which
 *   is why the mask is [batch, frames] while the features are [batch, bins, frames].
 * - The prompt is text, then one placeholder per audio frame, then more text. Only
 *   the audio side changes between inputs, so the text embeddings are baked and
 *   concatenated around the audio embeddings.
 * - The decoder carries a KV cache as two stacked tensors, layer-major.
 * - Output is a language tag, the language name, the `<asr_text>` marker, then the
 *   words. The marker is an added token, so it is absent from `model.vocab` and
 *   would otherwise decode to nothing and glue the language tag to the transcript.
 */

export interface Qwen3AsrSessionOptions {
  readonly ort: {
    readonly Tensor: new (
      type: string,
      data: ArrayBufferView | ArrayBufferLike,
      dims: readonly number[],
    ) => unknown;
  };
  readonly front: {
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  readonly decoder: {
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  readonly embed: {
    readonly outputNames: readonly string[];
    run(feeds: Record<string, unknown>): Promise<Record<string, unknown> | unknown[]>;
  };
  readonly tokenizer: Qwen3AsrTokenizer;
  /** Baked text embeddings around the audio span. */
  readonly prefixEmbeds: Float32Array;
  readonly suffixEmbeds: Float32Array;
  /** Layers, kv heads and head dimension of the stacked cache. */
  readonly layers: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly maxTokens?: number;
}

export interface Qwen3AsrDecodeResult {
  readonly text: string;
  readonly language: string;
  readonly tokenIds: readonly number[];
  readonly audioFrames: number;
  readonly steps: number;
}

interface TensorLike {
  readonly dims: readonly number[];
  readonly data: Float32Array;
}

/** Measured from the checkpoint: the token between the language name and the words. */
const ASR_TEXT_MARKER = 151704;

const EOS_IDS: ReadonlySet<number> = new Set([151643, 151645]);

/** Qwen3-ASR vocabulary plus the added tokens the plain map does not carry. */
export class Qwen3AsrTokenizer {
  private readonly idToToken = new Map<number, string>();

  constructor(tokenizer: {
    model?: { vocab?: Record<string, number> };
    vocab?: Record<string, number>;
    added_tokens?: { id?: number; content?: string }[];
    added_tokens_decoder?: Record<string, { id?: number; content?: string }>;
  }) {
    const source = tokenizer.model?.vocab ?? tokenizer.vocab ?? {};
    for (const [token, id] of Object.entries(source)) {
      this.idToToken.set(Number(id), token);
    }
    const added = tokenizer.added_tokens ?? Object.values(tokenizer.added_tokens_decoder ?? {});
    for (const entry of added) {
      const content = entry?.content;
      const id = entry?.id;
      if (typeof content === 'string' && typeof id === 'number') {
        this.idToToken.set(id, content);
      }
    }
  }

  /** Qwen is byte-level BPE, so the space marker is U+0120, not SentencePiece U+2581. */
  tokenText(id: number): string {
    return (this.idToToken.get(id) ?? '').replace(/Ġ/g, ' ');
  }
}

export class Qwen3AsrSession {
  private readonly options: Qwen3AsrSessionOptions;

  constructor(options: Qwen3AsrSessionOptions) {
    this.options = options;
  }

  private tensor(type: string, data: Float32Array, dims: readonly number[]): unknown {
    return new this.options.ort.Tensor(type, data, dims);
  }

  private zeroPast(): unknown {
    const { layers, kvHeads, headDim } = this.options;
    return this.tensor('float32', new Float32Array(0), [layers, 1, kvHeads, 0, headDim]);
  }

  private static read(
    result: Record<string, unknown> | unknown[],
    name: string,
    fallback: string,
  ): TensorLike {
    if (Array.isArray(result)) {
      return result[0] as TensorLike;
    }
    return (result[name] ?? result[fallback]) as TensorLike;
  }

  async transcribe(
    features: Float32Array,
    featureDims: readonly number[],
    mask: Float32Array,
    maskDims: readonly number[],
  ): Promise<Qwen3AsrDecodeResult> {
    const { front, decoder, embed, tokenizer } = this.options;
    const frontOut = await front.run({
      input_features: this.tensor('float32', features, featureDims),
      input_features_mask: this.tensor('float32', mask, maskDims),
    });
    const audio = Qwen3AsrSession.read(frontOut, 'audio_embeds', front.outputNames[0]!);
    const frames = audio.dims[0]!;
    const hidden = audio.dims[1]!;
    const prefixLength = this.options.prefixEmbeds.length / hidden;
    const suffixLength = this.options.suffixEmbeds.length / hidden;
    const promptLength = prefixLength + frames + suffixLength;

    const prompt = new Float32Array(promptLength * hidden);
    prompt.set(this.options.prefixEmbeds.subarray(0, prefixLength * hidden), 0);
    prompt.set(audio.data.subarray(0, frames * hidden), prefixLength * hidden);
    prompt.set(
      this.options.suffixEmbeds.subarray(0, suffixLength * hidden),
      (prefixLength + frames) * hidden,
    );

    let result = await decoder.run({
      inputs_embeds: this.tensor('float32', prompt, [1, promptLength, hidden]),
      past_key: this.zeroPast(),
      past_value: this.zeroPast(),
    });
    let presentKey = (result as Record<string, unknown>)['present_key'];
    let presentValue = (result as Record<string, unknown>)['present_value'];

    const limit = this.options.maxTokens ?? 256;
    const tokenIds: number[] = [];
    for (let step = 0; step < limit; step += 1) {
      const logits = (result as Record<string, unknown>)['logits'] as {
        data: Float32Array;
        dims: readonly number[];
      };
      const vocab = logits.dims[2]!;
      const lastRow = logits.data.length / vocab - 1;
      let bestId = 0;
      let best = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < vocab; i += 1) {
        const value = logits.data[lastRow * vocab + i]!;
        if (value > best) {
          best = value;
          bestId = i;
        }
      }
      if (EOS_IDS.has(bestId)) {
        break;
      }
      tokenIds.push(bestId);
      const embedded = Qwen3AsrSession.read(
        await embed.run({
          input_ids: this.tensor(
            'int64',
            BigInt64Array.from([BigInt(bestId)]) as unknown as Float32Array,
            [1, 1],
          ),
        }),
        'embeddings',
        embed.outputNames[0]!,
      );
      result = await decoder.run({
        inputs_embeds: embedded,
        past_key: presentKey,
        past_value: presentValue,
      });
      presentKey = (result as Record<string, unknown>)['present_key'];
      presentValue = (result as Record<string, unknown>)['present_value'];
    }
    const markerIndex = tokenIds.indexOf(ASR_TEXT_MARKER);
    const words = markerIndex >= 0 ? tokenIds.slice(markerIndex + 1) : tokenIds;
    const language =
      markerIndex >= 2
        ? tokenIds
            .slice(0, markerIndex)
            .map((id) => tokenizer.tokenText(id))
            .join('')
            .trim()
        : '';

    return {
      text: words
        .map((id) => tokenizer.tokenText(id))
        .join('')
        .trim(),
      language,
      tokenIds,
      audioFrames: frames,
      steps: tokenIds.length + 1,
    };
  }
}
