import { describe, expect, it, vi } from 'vitest';
import { OnnxNemoPreprocessor } from '../src/models/nemo-tdt/preprocessor.js';
import type { OrtModuleLike, OrtTensorLike } from '../src/models/nemo-tdt/ort.js';

function fixture(allocationFailure?: { readonly at: number; readonly error: Error }) {
  const inputs: OrtTensorLike[] = [];
  let allocationCount = 0;
  class Tensor<TData extends ArrayBufferView> implements OrtTensorLike<TData> {
    readonly dispose = vi.fn();
    constructor(
      _type: string,
      readonly data: TData,
      readonly dims: readonly number[],
    ) {
      if (++allocationCount === allocationFailure?.at) throw allocationFailure.error;
      inputs.push(this);
    }
  }
  const outputs: Record<string, OrtTensorLike> = {
    features: { data: new Float32Array([1, 2, 3, 4, 5, 6]), dims: [1, 2, 3], dispose: vi.fn() },
    features_lens: { data: BigInt64Array.from([2n]), dims: [1], dispose: vi.fn() },
  };
  const run = vi.fn(async () => outputs);
  const release = vi.fn();
  const ort: OrtModuleLike = {
    env: { wasm: {} },
    Tensor,
    InferenceSession: { create: vi.fn(async () => ({ run, release })) },
  };
  return {
    outputs,
    inputs,
    run,
    release,
    processor: new OnnxNemoPreprocessor(ort, 'processor.onnx'),
  };
}

const audio = {
  channels: [new Float32Array(160)],
  sampleRate: 16000,
  numberOfChannels: 1,
  numberOfFrames: 160,
  durationSeconds: 0.01,
};

describe('NeMo ONNX processor output contract', () => {
  it.each(['features', 'features_lens', 'waveform', 'length'] as const)(
    'attempts every tensor cleanup after %s disposal fails and remains reusable',
    async (target) => {
      const f = fixture();
      const error = new Error('Tensor cleanup failed');
      f.run.mockImplementationOnce(async () => {
        const tensor =
          target === 'waveform'
            ? f.inputs[0]!
            : target === 'length'
              ? f.inputs[1]!
              : f.outputs[target]!;
        vi.mocked(tensor.dispose!).mockImplementationOnce(() => {
          throw error;
        });
        return f.outputs;
      });
      try {
        await expect(f.processor.process(audio)).rejects.toBe(error);
        for (const tensor of [...f.inputs, ...Object.values(f.outputs)])
          expect(tensor.dispose).toHaveBeenCalledOnce();
        const result = await f.processor.process(audio);
        expect(result.features).toEqual(Float32Array.from([1, 2, 3, 4, 5, 6]));
        for (const tensor of f.inputs) expect(tensor.dispose).toHaveBeenCalledOnce();
        for (const tensor of Object.values(f.outputs))
          expect(tensor.dispose).toHaveBeenCalledTimes(2);
      } finally {
        await f.processor.dispose();
      }
      expect(f.release).toHaveBeenCalledOnce();
    },
  );

  it.each([new Error('Inference failed'), undefined])(
    'retains an inference failure alongside all input cleanup failures (%s)',
    async (primary) => {
      const f = fixture();
      const cleanup = [new Error('Waveform cleanup failed'), new Error('Length cleanup failed')];
      f.run.mockImplementationOnce(async () => {
        f.inputs.forEach((tensor, index) =>
          vi.mocked(tensor.dispose!).mockImplementationOnce(() => {
            throw cleanup[index];
          }),
        );
        throw primary;
      });
      try {
        await expect(f.processor.process(audio)).rejects.toMatchObject({
          cause: primary,
          errors: [primary, ...cleanup],
        });
        for (const tensor of f.inputs) expect(tensor.dispose).toHaveBeenCalledOnce();
        for (const tensor of Object.values(f.outputs))
          expect(tensor.dispose).not.toHaveBeenCalled();
      } finally {
        await f.processor.dispose();
      }
    },
  );

  it('retains validation failure and all output cleanup failures', async () => {
    const f = fixture();
    (f.outputs.features!.data as Float32Array)[0] = NaN;
    const cleanup = [new Error('Features cleanup failed'), new Error('Length cleanup failed')];
    Object.values(f.outputs).forEach((tensor, index) =>
      vi.mocked(tensor.dispose!).mockImplementationOnce(() => {
        throw cleanup[index];
      }),
    );
    try {
      const failure = await f.processor.process(audio).catch((error: unknown) => error);
      expect(failure).toBeInstanceOf(AggregateError);
      const aggregate = failure as AggregateError;
      expect(aggregate.errors).toHaveLength(3);
      expect(aggregate.errors[0]).toBe(aggregate.cause);
      expect(aggregate.errors[0].message).toMatch(/NeMo preprocessor output contract/);
      expect(aggregate.errors.slice(1)).toEqual(cleanup);
      for (const tensor of [...f.inputs, ...Object.values(f.outputs)])
        expect(tensor.dispose).toHaveBeenCalledOnce();
    } finally {
      await f.processor.dispose();
    }
  });

  it('reports multiple cleanup failures after successful inference', async () => {
    const f = fixture();
    const cleanup = [new Error('Features cleanup failed'), new Error('Length cleanup failed')];
    Object.values(f.outputs).forEach((tensor, index) =>
      vi.mocked(tensor.dispose!).mockImplementationOnce(() => {
        throw cleanup[index];
      }),
    );
    try {
      await expect(f.processor.process(audio)).rejects.toMatchObject({
        cause: cleanup[0],
        errors: cleanup,
      });
      for (const tensor of [...f.inputs, ...Object.values(f.outputs)])
        expect(tensor.dispose).toHaveBeenCalledOnce();
    } finally {
      await f.processor.dispose();
    }
  });

  it('disposes aliased output tensors once', async () => {
    const f = fixture();
    f.outputs.extra = f.outputs.features!;
    try {
      await f.processor.process(audio);
      for (const tensor of [...f.inputs, ...Object.values(f.outputs)])
        expect(tensor.dispose).toHaveBeenCalledOnce();
    } finally {
      await f.processor.dispose();
    }
  });

  it.each([1, 2])(
    'cleans partial inputs after allocation %i fails and remains reusable',
    async (at) => {
      const error = new Error('Input allocation failed');
      const f = fixture({ at, error });
      try {
        await expect(f.processor.process(audio)).rejects.toBe(error);
        expect(f.run).not.toHaveBeenCalled();
        expect(f.inputs).toHaveLength(at - 1);
        for (const tensor of f.inputs) expect(tensor.dispose).toHaveBeenCalledOnce();
        const result = await f.processor.process(audio);
        expect(result.features).toEqual(Float32Array.from([1, 2, 3, 4, 5, 6]));
        expect(f.run).toHaveBeenCalledOnce();
        for (const tensor of f.inputs) expect(tensor.dispose).toHaveBeenCalledOnce();
      } finally {
        await f.processor.dispose();
        await f.processor.dispose();
      }
      expect(f.release).toHaveBeenCalledOnce();
    },
  );

  it.each(['int64', 'int32'])(
    'accepts %s lengths and owns the returned feature copy',
    async (dtype) => {
      const f = fixture();
      if (dtype === 'int32')
        f.outputs.features_lens = { data: Int32Array.from([2]), dims: [1], dispose: vi.fn() };
      const result = await f.processor.process(audio);
      expect(result).toEqual({
        features: Float32Array.from([1, 2, 3, 4, 5, 6]),
        frameCount: 3,
        validLength: 2,
      });
      expect(result.features).not.toBe(f.outputs.features!.data);
      for (const tensor of [...f.inputs, ...Object.values(f.outputs)])
        expect(tensor.dispose).toHaveBeenCalledOnce();
      await f.processor.dispose();
      expect(f.release).toHaveBeenCalledOnce();
    },
  );

  it.each([
    [
      'missing features',
      (o: Record<string, OrtTensorLike>) => {
        delete o.features;
      },
    ],
    [
      'wrong rank',
      (o: Record<string, OrtTensorLike>) => {
        o.features = { ...o.features!, dims: [2, 3] };
      },
    ],
    [
      'batch two',
      (o: Record<string, OrtTensorLike>) => {
        o.features = { ...o.features!, dims: [2, 1, 3] };
      },
    ],
    [
      'fractional dimension',
      (o: Record<string, OrtTensorLike>) => {
        o.features = { ...o.features!, dims: [1, 2, 3.5] };
      },
    ],
    [
      'wrong count',
      (o: Record<string, OrtTensorLike>) => {
        o.features = { ...o.features!, dims: [1, 2, 4] };
      },
    ],
    [
      'wrong dtype',
      (o: Record<string, OrtTensorLike>) => {
        o.features = { ...o.features!, data: new Float64Array(6) };
      },
    ],
    [
      'nonfinite features',
      (o: Record<string, OrtTensorLike>) => {
        (o.features!.data as Float32Array)[0] = NaN;
      },
    ],
    [
      'missing lengths',
      (o: Record<string, OrtTensorLike>) => {
        delete o.features_lens;
      },
    ],
    [
      'wrong length dtype',
      (o: Record<string, OrtTensorLike>) => {
        o.features_lens = { ...o.features_lens!, data: Float32Array.from([2]) };
      },
    ],
    [
      'negative length',
      (o: Record<string, OrtTensorLike>) => {
        o.features_lens = { ...o.features_lens!, data: BigInt64Array.from([-1n]) };
      },
    ],
    [
      'length exceeds frames',
      (o: Record<string, OrtTensorLike>) => {
        o.features_lens = { ...o.features_lens!, data: BigInt64Array.from([4n]) };
      },
    ],
    [
      'length rank',
      (o: Record<string, OrtTensorLike>) => {
        o.features_lens = { ...o.features_lens!, dims: [1, 1] };
      },
    ],
  ] as const)('rejects %s and releases tensors', async (_name, mutate) => {
    const f = fixture();
    mutate(f.outputs);
    try {
      await expect(f.processor.process(audio)).rejects.toThrow(/NeMo preprocessor output contract/);
      for (const tensor of [...f.inputs, ...Object.values(f.outputs)])
        expect(tensor.dispose).toHaveBeenCalledOnce();
    } finally {
      await f.processor.dispose();
    }
  });
});
