import { describe, expect, it } from 'vitest';
import {
  copyAndReleaseWhisperPresentKv,
  disposeOwnedDecoderFeeds,
  disposeOrtKv,
  disposeReplacedOrtKv,
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
