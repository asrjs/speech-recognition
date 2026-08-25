import { afterEach, describe, expect, it } from 'vitest';
import {
  cloneDecoderKvDataForInput,
  canShareWhisperEncoderKvAcrossBatch,
  concatDecoderKvDataForBatch,
  float32ToFloat16Bits,
  maybeCastWhisperFeatureTensor,
  sliceDecoderKvDataForBatch,
} from '../src/models/whisper-seq2seq/executor.js';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/index.js';

const originalFloat16Array = (globalThis as any).Float16Array;

afterEach(() => {
  (globalThis as any).Float16Array = originalFloat16Array;
});

describe('Whisper fp16 decoder-step KV inputs', () => {
  it('only broadcasts encoder KV when sibling beams share the same batch-one buffer', () => {
    const shared = new Float32Array([1, 2, 3, 4]);
    expect(
      canShareWhisperEncoderKvAcrossBatch([
        { data: shared, dims: [1, 2, 2, 1], type: 'float32' },
        { data: shared, dims: [1, 2, 2, 1], type: 'float32' },
      ]),
    ).toBe(true);

    expect(
      canShareWhisperEncoderKvAcrossBatch([
        { data: shared, dims: [1, 2, 2, 1], type: 'float32' },
        { data: new Float32Array(shared), dims: [1, 2, 2, 1], type: 'float32' },
      ]),
    ).toBe(false);
    expect(
      canShareWhisperEncoderKvAcrossBatch([
        { data: shared, dims: [2, 2, 2, 1], type: 'float32' },
        { data: shared, dims: [2, 2, 2, 1], type: 'float32' },
      ]),
    ).toBe(false);
  });

  it('converts normal and subnormal values to IEEE fp16 with round-to-nearest-even', () => {
    const values = new Float32Array([-2, 2, 2 ** -24, 2 ** -25, 1 + 2 ** -11, 1 + 3 * 2 ** -11]);

    expect(Array.from(float32ToFloat16Bits(values))).toEqual([
      0xc000, 0x4000, 0x0001, 0x0000, 0x3c00, 0x3c02,
    ]);
  });

  it('casts float32 mel features when the encoder declares a float16 input', async () => {
    class Tensor {
      readonly type: string;
      readonly data: ArrayBufferView;
      readonly dims: readonly number[];

      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    }

    const source = {
      type: 'float32',
      data: new Float32Array([1, -2, 0.5, 4]),
      dims: [1, 1, 4],
    };
    const cast = await maybeCastWhisperFeatureTensor(
      source,
      { inputMetadata: [{ name: 'input_features', type: 'float16' }] } as any,
      { Tensor } as any,
    );

    expect(cast.type).toBe('float16');
    expect(cast.dims).toEqual(source.dims);
    expect(cast.data).toBeInstanceOf(Uint16Array);
    expect(cast.data).not.toBe(source.data);
  });

  it('wraps raw fp16 KV bits with Float16Array for callback-based split decoding', () => {
    (globalThis as any).Float16Array = class Float16Array extends Uint16Array {};

    const source = new Uint16Array([1, 2, 3, 4]);
    const cloned = cloneDecoderKvDataForInput(source, 'float16');

    expect(cloned.type).toBe('float16');
    expect(cloned.data.constructor.name).toBe('Float16Array');
    expect(cloned.data).not.toBe(source);
    expect(Array.from(cloned.data as Uint16Array)).toEqual([1, 2, 3, 4]);
  });

  it('concatenates raw fp16 KV bits as Float16Array for batched beam inputs', () => {
    (globalThis as any).Float16Array = class Float16Array extends Uint16Array {};

    const first = new Uint16Array([1, 2]);
    const second = new Uint16Array([3, 4]);
    const batched = concatDecoderKvDataForBatch(
      [
        { data: first, type: 'float16' },
        { data: second, type: 'float16' },
      ],
      'float16',
    );

    expect(batched.type).toBe('float16');
    expect(batched.data.constructor.name).toBe('Float16Array');
    expect(Array.from(batched.data as Uint16Array)).toEqual([1, 2, 3, 4]);

    first[0] = 9;
    expect(Array.from(batched.data as Uint16Array)).toEqual([1, 2, 3, 4]);
  });

  it('splits batched KV outputs with zero-copy typed-array views', () => {
    const source = new Uint16Array([1, 2, 3, 4]);
    const slice = sliceDecoderKvDataForBatch(source, 1, 2);

    expect(slice).toBeInstanceOf(Uint16Array);
    expect(slice.buffer).toBe(source.buffer);
    expect(Array.from(slice as Uint16Array)).toEqual([2, 3]);

    source[1] = 9;
    expect(slice[0]).toBe(9);
  });

  it('re-wraps Uint16Array fp16 KV data with Float16Array when the runtime provides it', async () => {
    (globalThis as any).Float16Array = class Float16Array extends Uint16Array {};

    const feedTypes: Record<string, string> = {};
    const executor = new WhisperOnnxExecutor(
      'mock-whisper',
      {},
      {
        ecosystem: 'openai',
        architecture: 'whisper-seq2seq',
        processorArchitecture: 'whisper-mel',
        encoderArchitecture: 'whisper-transformer',
        decoderArchitecture: 'transformer-decoder',
        sampleRate: 16000,
        melBins: 80,
        maxSourcePositions: 1500,
        maxTargetPositions: 448,
        languages: ['en'],
        tokenizer: { kind: 'tiktoken' },
      },
      'webgpu',
      undefined,
    );

    const loaded = {
      ort: {
        Tensor: class Tensor {
          readonly type: string;
          readonly data: ArrayBufferView;
          readonly dims: readonly number[];

          constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
            if (type === 'float16' && data.constructor.name !== 'Float16Array') {
              throw new Error(`expected Float16Array, got ${data.constructor.name}`);
            }
            this.type = type;
            this.data = data;
            this.dims = dims;
          }
        },
      },
      decoderStepSession: {
        async run(
          feeds: Record<string, { readonly type?: string; readonly data?: ArrayBufferView }>,
        ) {
          for (const [name, value] of Object.entries(feeds)) {
            feedTypes[name] = value.data?.constructor.name ?? '';
          }
          return {
            logits: {
              type: 'float32',
              data: new Float32Array([0, 1, 0]),
              dims: [1, 1, 3],
            },
          };
        },
      },
    };

    await (executor as any).runDecoderStepMultiToken(loaded, [2], {
      'past_key_values.0.decoder.key': {
        type: 'float16',
        data: new Uint16Array([1, 2, 3, 4]),
        dims: [1, 1, 2, 2],
      },
    });

    expect(feedTypes['past_key_values.0.decoder.key']).toBe('Float16Array');
  });

  it('passes adapter-prepared KV tensors without cloning them again', async () => {
    let tensorCreateCount = 0;
    let receivedPastKv: unknown;
    class Tensor {
      readonly type: string;
      readonly data: ArrayBufferView;
      readonly dims: readonly number[];

      constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
        tensorCreateCount += 1;
        this.type = type;
        this.data = data;
        this.dims = dims;
      }
    }

    const preparedPastKv = {
      type: 'float16',
      data: new Uint16Array([1, 2, 3, 4]),
      dims: [1, 1, 2, 2],
    };
    const executor = new WhisperOnnxExecutor(
      'mock-whisper',
      {},
      {
        ecosystem: 'openai',
        architecture: 'whisper-seq2seq',
        processorArchitecture: 'whisper-mel',
        encoderArchitecture: 'whisper-transformer',
        decoderArchitecture: 'transformer-decoder',
        sampleRate: 16000,
        melBins: 80,
        maxSourcePositions: 1500,
        maxTargetPositions: 448,
        languages: ['en'],
        tokenizer: { kind: 'tiktoken' },
      },
      'webgpu',
      undefined,
    );

    const loaded = {
      ort: { Tensor },
      decoderStepSession: {
        async run(feeds: Record<string, unknown>) {
          receivedPastKv = feeds['past_key_values.0.decoder.key'];
          return {
            logits: {
              type: 'float32',
              data: new Float32Array([0, 1, 0]),
              dims: [1, 1, 3],
            },
          };
        },
      },
    };

    await (executor as any).runDecoderStepMultiToken(
      loaded,
      [2],
      { 'past_key_values.0.decoder.key': preparedPastKv },
      { preparedPastKv: true },
    );

    expect(tensorCreateCount).toBe(1); // input_ids only
    expect(receivedPastKv).toBe(preparedPastKv);
  });
});
