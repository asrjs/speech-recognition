import { describe, expect, it } from 'vitest';
import {
  copyAndReleaseWhisperPresentKv,
  createWhisperBatchedCachePosition,
  createWhisperBatchedStepFeedTensors,
  disposeOwnedDecoderFeeds,
  disposeOwnedWhisperTensors,
  disposeWhisperEncoderKvCache,
  disposeOrtKv,
  disposeReplacedOrtKv,
  withOwnedDecoderFeeds,
  WhisperOnnxExecutor,
} from '../src/models/whisper-seq2seq/executor.js';
import { DEFAULT_WHISPER_CLASSIFICATION, parseWhisperSeq2SeqConfig } from '../src/models/whisper-seq2seq/config.js';
import type { OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';

class TrackingTensor implements OrtTensorLike<Float32Array> {
  disposed = 0;

  constructor(
    readonly type: string,
    readonly data: Float32Array | Uint16Array | BigInt64Array,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed += 1;
  }
}

function createExecutor(): WhisperOnnxExecutor {
  return new WhisperOnnxExecutor(
    'whisper-decoder-dispose',
    DEFAULT_WHISPER_CLASSIFICATION,
    parseWhisperSeq2SeqConfig('whisper-decoder-dispose', {
      maxSourcePositions: 4,
      maxTargetPositions: 8,
    }),
    'wasm',
    undefined,
  );
}

describe('Whisper decoder-step tensor dispose', () => {
  it('disposes owned feeds that are not in the retain set', () => {
    const keep = new TrackingTensor('float32', new Float32Array(2), [1, 2]);
    const owned = new TrackingTensor('int64', new BigInt64Array([1n]), [1, 1]);
    disposeOwnedDecoderFeeds({ keep, owned }, new Set([keep]));
    expect(keep.disposed).toBe(0);
    expect(owned.disposed).toBe(1);
  });

  it.each([
    { label: 'success', rejects: false },
    { label: 'decoder throw', rejects: true },
  ])('releases callback-owned decoder feeds after $label', async ({ rejects }) => {
    for (let attempt = 0; attempt < (rejects ? 1 : 2); attempt++) {
      const borrowed = new TrackingTensor('float32', new Float32Array(2), [1, 1]);
      const owned = new TrackingTensor('int64', new BigInt64Array([7n]), [1, 1]);
      const feeds = { borrowed, owned };
      const operation = rejects
        ? () => Promise.reject(new Error('decoder boom'))
        : async () => 'decoded';

      if (rejects) {
        await expect(withOwnedDecoderFeeds(feeds, new Set([borrowed]), operation))
          .rejects.toThrow(/decoder boom/);
      } else {
        await expect(withOwnedDecoderFeeds(feeds, new Set([borrowed]), operation))
          .resolves.toBe('decoded');
      }

      expect(borrowed.disposed).toBe(0);
      expect(owned.disposed).toBe(1);
    }
  });

  it('disposes transcription-owned encoder and cached tensors once by identity', () => {
    const sharedTensor = new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]);
    const owned = new Set([sharedTensor, sharedTensor]);

    disposeOwnedWhisperTensors(owned);
    disposeOwnedWhisperTensors(owned);

    expect(sharedTensor.disposed).toBe(1);
    expect(owned.size).toBe(0);
  });

  it('releases every scalar encoder-KV cache entry and clears its map', () => {
    const sharedTensor = new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]);
    const cache = new Map([
      ['encoder.key', { tensor: sharedTensor }],
      ['encoder.value', { tensor: sharedTensor }],
    ]);

    disposeWhisperEncoderKvCache(cache);
    disposeWhisperEncoderKvCache(cache);

    expect(sharedTensor.disposed).toBe(1);
    expect(cache.size).toBe(0);
  });

  it('disposes replaced KV tensors but keeps reused encoder cache', () => {
    const encoder = new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]);
    const oldDecoder = new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]);
    const nextDecoder = new TrackingTensor('float32', new Float32Array(8), [1, 1, 2, 4]);
    disposeReplacedOrtKv(
      { 'past_key_values.0.encoder.key': encoder, 'past_key_values.0.decoder.key': oldDecoder },
      { 'past_key_values.0.encoder.key': encoder, 'past_key_values.0.decoder.key': nextDecoder },
    );
    expect(encoder.disposed).toBe(0);
    expect(oldDecoder.disposed).toBe(1);
    expect(nextDecoder.disposed).toBe(0);
    disposeOrtKv({ 'past_key_values.0.encoder.key': encoder, 'past_key_values.0.decoder.key': nextDecoder });
    expect(encoder.disposed).toBe(1);
    expect(nextDecoder.disposed).toBe(1);
  });

  it('disposes decoder-step input_ids and logits while keeping present KV and encoder states', async () => {
    const created: TrackingTensor[] = [];
    class Tensor extends TrackingTensor {
      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        super(type, data as Float32Array, dims);
        created.push(this);
      }
    }
    const encoderHidden = new TrackingTensor('float32', new Float32Array(8), [1, 2, 4]);
    const presentKey = new TrackingTensor('float32', new Float32Array(4), [1, 1, 1, 4]);
    const logits = new TrackingTensor('float32', new Float32Array([0, 4, 1]), [1, 1, 3]);
    let feedsSeen: Record<string, TrackingTensor> | undefined;
    const executor = createExecutor();
    const loaded = {
      ort: { Tensor },
      decoderSession: {
        inputNames: ['input_ids', 'encoder_hidden_states'],
        async run(feeds: Record<string, TrackingTensor>) {
          feedsSeen = feeds;
          return {
            logits,
            'present.0.decoder.key': presentKey,
          };
        },
      },
      modelConfig: {
        decoderLayers: 1,
        decoderAttentionHeads: 1,
        headDim: 4,
      },
    };

    const result = await (executor as unknown as {
      runDecoderStep: (
        loadedState: unknown,
        encoder: unknown,
        tokens: readonly number[],
        past: Record<string, never>,
        isFirstStep: boolean,
      ) => Promise<{ lastLogits: Float32Array; pastKeyValues: Record<string, TrackingTensor> }>;
    }).runDecoderStep(loaded, encoderHidden, [50258], {}, true);

    expect(feedsSeen?.input_ids.disposed).toBe(1);
    expect(encoderHidden.disposed).toBe(0);
    expect(logits.disposed).toBe(1);
    expect(presentKey.disposed).toBe(0);
    expect(result.pastKeyValues['past_key_values.0.decoder.key']).toBe(presentKey);
    expect(Array.from(result.lastLogits)).toEqual([0, 4, 1]);
    expect(created.some((tensor) => tensor === feedsSeen?.input_ids)).toBe(true);
  });

  it('feeds declared cache_position from decoder KV length and disposes it', async () => {
    const created: TrackingTensor[] = [];
    const positionsSeen: bigint[][] = [];
    const positionDims: (readonly number[])[] = [];
    class Tensor extends TrackingTensor {
      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        super(type, data as Float32Array, dims);
        created.push(this);
      }
    }
    const executor = createExecutor();
    const loaded = {
      ort: { Tensor },
      decoderSession: {
        inputNames: [
          'input_ids',
          'encoder_hidden_states',
          'past_key_values.0.decoder.key',
          'cache_position',
        ],
        async run(feeds: Record<string, TrackingTensor>) {
          const position = feeds.cache_position;
          positionsSeen.push(Array.from(position.data as BigInt64Array));
          positionDims.push(position.dims);
          const cacheLength = feeds['past_key_values.0.decoder.key']?.dims[2] ?? 0;
          const nextLength = cacheLength + (feeds.input_ids.dims[1] ?? 0);
          return {
            logits: new Tensor('float32', new Float32Array([0, 1]), [1, 1, 2]),
            'present.0.decoder.key': new Tensor('float32', new Float32Array(nextLength * 4), [1, 1, nextLength, 4]),
          };
        },
      },
      modelConfig: { decoderLayers: 1, decoderAttentionHeads: 1, headDim: 4 },
    };
    const encoder = new TrackingTensor('float32', new Float32Array(8), [1, 2, 4]);
    const call = (executor as unknown as {
      runDecoderStep(
        loaded: unknown,
        encoder: TrackingTensor,
        tokens: readonly number[],
        past: Record<string, TrackingTensor>,
        first: boolean,
      ): Promise<{ pastKeyValues: Record<string, TrackingTensor> }>;
    }).runDecoderStep.bind(executor);

    const first = await call(loaded, encoder, [10, 11, 12, 13], {}, true);
    await call(loaded, encoder, [10, 11, 12, 13, 14], first.pastKeyValues, false);

    expect(positionsSeen).toEqual([[0n, 1n, 2n, 3n], [4n]]);
    expect(positionDims).toEqual([[4], [1]]);
    expect(created.filter((tensor) => tensor.type === 'int64').every((tensor) => tensor.disposed === 1)).toBe(true);
  });

  it('builds one shared 1D cache_position for a two-beam, single-token step', () => {
    const beam = (length: number) => ({
      'past_key_values.0.decoder.key': {
        data: new Float32Array(length * 4),
        dims: [1, 1, length, 4],
      },
    });
    const sharedPosition = createWhisperBatchedCachePosition([beam(4), beam(4)], 1);
    expect(Array.from(sharedPosition)).toEqual([4n]);
    expect(sharedPosition.length).toBe(1);
    expect(() => createWhisperBatchedCachePosition([beam(4), beam(5)], 1))
      .toThrow(/different decoder cache_position lengths/);
  });

  it('builds the actual batched-step feeds with B input rows and one shared position', () => {
    const created: TrackingTensor[] = [];
    class Tensor extends TrackingTensor {
      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        super(type, data as Float32Array, dims);
        created.push(this);
      }
    }
    const beam = (length: number) => ({
      'past_key_values.0.decoder.key': {
        data: new Float32Array(length * 4),
        dims: [1, 1, length, 4],
      },
    });
    const feeds = createWhisperBatchedStepFeedTensors(
      { Tensor } as never,
      [20, 21],
      [beam(4), beam(4)],
      true,
    ) as Record<string, TrackingTensor>;

    expect(feeds.input_ids.dims).toEqual([2, 1]);
    expect(Array.from(feeds.input_ids.data as BigInt64Array)).toEqual([20n, 21n]);
    expect(feeds.cache_position.dims).toEqual([1]);
    expect(Array.from(feeds.cache_position.data as BigInt64Array)).toEqual([4n]);

    created.length = 0;
    expect(() => createWhisperBatchedStepFeedTensors(
      { Tensor } as never,
      [20, 21],
      [beam(4), beam(5)],
      true,
    )).toThrow(/different decoder cache_position lengths/);
    expect(created).toHaveLength(0);
  });

  it('disposes cloned KV feeds and still retains the caller cache on throw', async () => {
    const liveKv = new TrackingTensor('float32', new Float32Array([1, 2, 3, 4]), [1, 1, 1, 4]);
    const created: TrackingTensor[] = [];
    class Tensor extends TrackingTensor {
      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        super(type, data as Float32Array, dims);
        created.push(this);
      }
    }
    const executor = createExecutor();
    const loaded = {
      ort: { Tensor },
      decoderStepSession: {
        inputNames: ['input_ids', 'past_key_values.0.decoder.key', 'cache_position'],
        async run() {
          throw new Error('decoder boom');
        },
      },
    };

    await expect(
      (executor as unknown as {
        runDecoderStepMultiToken: (
          loadedState: unknown,
          tokenIds: readonly number[],
          pastKv: Record<string, TrackingTensor>,
        ) => Promise<unknown>;
      }).runDecoderStepMultiToken(
        loaded,
        [50258],
        { 'past_key_values.0.decoder.key': liveKv },
      ),
    ).rejects.toThrow(/decoder boom/);

    expect(liveKv.disposed).toBe(0);
    expect(created.some((tensor) => tensor.disposed === 1)).toBe(true);
    const positionFeed = created.find((tensor) => tensor.type === 'int64' && tensor.dims.length === 1);
    expect(positionFeed?.data).toEqual(new BigInt64Array([1n]));
    expect(positionFeed?.disposed).toBe(1);
  });

  it('feeds multi-token cache_position from the split graph decoder cache length', async () => {
    const created: TrackingTensor[] = [];
    let positionSeen: TrackingTensor | undefined;
    class Tensor extends TrackingTensor {
      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        super(type, data as Float32Array, dims);
        created.push(this);
      }
    }
    const executor = createExecutor();
    const sourceCache = new TrackingTensor('float32', new Float32Array(16), [1, 1, 4, 4]);
    const loaded = {
      ort: { Tensor },
      decoderStepSession: {
        inputNames: ['input_ids', 'past_key_values.0.decoder.key', 'cache_position'],
        async run(feeds: Record<string, TrackingTensor>) {
          positionSeen = feeds.cache_position;
          return {
            logits: new Tensor('float32', new Float32Array(8), [1, 2, 4]),
            'present.0.decoder.key': new Tensor('float32', new Float32Array(24), [1, 1, 6, 4]),
          };
        },
      },
    };
    await (executor as unknown as {
      runDecoderStepMultiToken(
        loaded: unknown,
        tokens: readonly number[],
        past: Record<string, TrackingTensor>,
      ): Promise<unknown>;
    }).runDecoderStepMultiToken(
      loaded,
      [20, 21],
      { 'past_key_values.0.decoder.key': sourceCache },
    );

    expect(positionSeen?.type).toBe('int64');
    expect(positionSeen?.dims).toEqual([2]);
    expect(positionSeen?.data).toEqual(new BigInt64Array([4n, 5n]));
    expect(positionSeen?.disposed).toBe(1);
    expect(sourceCache.disposed).toBe(0);
    expect(created.some((tensor) => tensor === positionSeen)).toBe(true);
  });

  it('copies callback present-KV data then disposes Ort wrappers except retained encoder cache', () => {
    const encoder = new TrackingTensor('float32', new Float32Array([1, 2, 3, 4]), [1, 1, 1, 4]);
    const decoder = new TrackingTensor('float32', new Float32Array([5, 6, 7, 8]), [1, 1, 1, 4]);
    const retained = new Map([
      [encoder, { data: new Float32Array([9, 9, 9, 9]), dims: encoder.dims, type: 'float32' as const }],
    ]);
    const copied = copyAndReleaseWhisperPresentKv(
      {
        'past_key_values.0.encoder.key': encoder,
        'past_key_values.0.decoder.key': decoder,
      },
      retained,
    );
    expect(encoder.disposed).toBe(0);
    expect(decoder.disposed).toBe(1);
    expect(copied['past_key_values.0.encoder.key']?.data).toBe(retained.get(encoder)?.data);
    expect(copied['past_key_values.0.decoder.key']?.data).not.toBe(decoder.data);
    expect(Array.from(copied['past_key_values.0.decoder.key']?.data as Float32Array)).toEqual([5, 6, 7, 8]);
  });
});
