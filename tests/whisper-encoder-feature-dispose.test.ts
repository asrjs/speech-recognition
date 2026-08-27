import { describe, expect, it } from 'vitest';
import {
  maybeCastWhisperFeatureTensor,
  runWhisperEncoderOnFeatures,
} from '../src/models/whisper-seq2seq/executor.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';

class TrackingTensor implements OrtTensorLike<Float32Array> {
  disposed = 0;

  constructor(
    readonly type: string,
    readonly data: Float32Array | Uint16Array,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed += 1;
  }
}

class TrackingOrtTensor {
  disposed = 0;
  readonly type: string;
  readonly data: ArrayBufferView;
  readonly dims: readonly number[];

  constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }

  dispose(): void {
    this.disposed += 1;
  }
}

describe('Whisper encoder feature tensor dispose', () => {
  it('disposes the original mel tensor when recasting to float16', async () => {
    const source = new TrackingTensor('float32', new Float32Array([1, -2, 0.5, 4]), [1, 1, 4]);
    const recast = await maybeCastWhisperFeatureTensor(
      source,
      { inputMetadata: [{ name: 'input_features', type: 'float16' }] } as OrtSessionLike,
      { Tensor: TrackingOrtTensor } as unknown as OrtModuleLike,
    );

    expect(source.disposed).toBe(1);
    expect(recast).not.toBe(source);
    expect(recast.type).toBe('float16');
  });

  it('leaves a matching-dtype mel tensor for the encoder helper to free', async () => {
    const source = new TrackingTensor('float32', new Float32Array([1, 2, 3, 4]), [1, 1, 4]);
    const recast = await maybeCastWhisperFeatureTensor(
      source,
      { inputMetadata: [{ name: 'input_features', type: 'float32' }] } as OrtSessionLike,
      { Tensor: TrackingOrtTensor } as unknown as OrtModuleLike,
    );

    expect(recast).toBe(source);
    expect(source.disposed).toBe(0);
  });

  it('disposes the encoder feed after run, including when the graph throws', async () => {
    const feed = new TrackingTensor('float32', new Float32Array(8), [1, 2, 4]);
    const session = {
      async run() {
        expect(feed.disposed).toBe(0);
        return { last_hidden_state: feed };
      },
    } as unknown as OrtSessionLike;

    const outputs = await runWhisperEncoderOnFeatures(session, feed);
    expect(feed.disposed).toBe(1);
    expect(outputs.last_hidden_state).toBe(feed);

    const failing = new TrackingTensor('float32', new Float32Array(8), [1, 2, 4]);
    await expect(
      runWhisperEncoderOnFeatures(
        {
          async run() {
            throw new Error('encoder boom');
          },
        } as unknown as OrtSessionLike,
        failing,
      ),
    ).rejects.toThrow(/encoder boom/);
    expect(failing.disposed).toBe(1);
  });
});
