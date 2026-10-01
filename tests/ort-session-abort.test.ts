import { describe, expect, it, vi } from 'vitest';

import { AssetLoadAbortedError, withOrtCreateAbort } from '../src/io/abort.js';
import { createOrtSession, type OrtModuleLike } from '../src/models/lasr-ctc/ort.js';
import {
  createOrtSession as createNemoTdtOrtSession,
  type OrtModuleLike as NemoTdtOrtModuleLike,
} from '../src/models/nemo-tdt/ort.js';
import { OnnxNemoPreprocessor } from '../src/models/nemo-tdt/preprocessor.js';
import type { OrtModuleLike as NemoOrtModuleLike } from '../src/models/nemo-tdt/ort.js';

describe('createOrtSession abort', () => {
  it('waits for asynchronous release before rejecting an aborted wrapped creation', async () => {
    const signal = { aborted: false };
    let finishRelease = () => {};
    const releaseGate = new Promise<void>((resolve) => {
      finishRelease = resolve;
    });
    const release = vi.fn(() => releaseGate);
    const ort = withOrtCreateAbort(
      {
        InferenceSession: {
          create: async (_url: string) => {
            signal.aborted = true;
            return { release };
          },
        },
      },
      signal,
    );
    let settled = false;
    const creation = ort.InferenceSession.create('local-model.onnx').catch((error: unknown) => {
      settled = true;
      return error;
    });
    try {
      await vi.waitFor(() => expect(release).toHaveBeenCalledTimes(1));
      expect(settled).toBe(false);
    } finally {
      finishRelease();
      await expect(creation).resolves.toBeInstanceOf(AssetLoadAbortedError);
    }
  });

  it('releases the session when abort is observed after InferenceSession.create', async () => {
    const signal = { aborted: false };
    const release = vi.fn();
    const session = { release };
    const ort = {
      env: { wasm: {} },
      Tensor: class {},
      InferenceSession: {
        create: vi.fn(async () => {
          signal.aborted = true;
          return session;
        }),
      },
    };

    await expect(
      createOrtSession(ort as unknown as OrtModuleLike, 'https://example.com/model.onnx', {
        backendId: 'wasm',
        signal,
      }),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('does not create a session when abort is already signaled', async () => {
    const create = vi.fn();
    const ort = {
      env: { wasm: {} },
      Tensor: class {},
      InferenceSession: { create },
    };

    await expect(
      createOrtSession(ort as unknown as OrtModuleLike, 'https://example.com/model.onnx', {
        backendId: 'wasm',
        signal: { aborted: true },
      }),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
    expect(create).not.toHaveBeenCalled();
  });
});

describe('NeMo TDT decoder output placement', () => {
  it('keeps gpu-buffer diagnostics CPU-backed in Node', async () => {
    const create = vi.fn(async () => ({ release: vi.fn() }));
    const ort = {
      env: { wasm: {} },
      Tensor: class {},
      InferenceSession: { create },
    };

    await createNemoTdtOrtSession(
      ort as unknown as NemoTdtOrtModuleLike,
      'https://example.com/decoder.onnx',
      {
        backendId: 'webgpu',
        preferredOutputLocation: {
          outputs: 'cpu',
          output_states_1: 'gpu-buffer',
          output_states_2: 'gpu-buffer',
        },
        webgpuOptions: { storageBufferCacheMode: 'simple' },
      },
    );

    expect(create).toHaveBeenCalledWith(
      'https://example.com/decoder.onnx',
      expect.objectContaining({
        preferredOutputLocation: {
          outputs: 'cpu',
          output_states_1: 'cpu',
          output_states_2: 'cpu',
        },
        executionProviders: [
          expect.objectContaining({
            name: 'webgpu',
            storageBufferCacheMode: 'simple',
          }),
        ],
      }),
    );
  });
});

describe('OnnxNemoPreprocessor abort', () => {
  function fixture(release: () => void | Promise<void>) {
    const features = { data: new Float32Array([0.5]), dims: [1, 1, 1], dispose: vi.fn() };
    const lengths = { data: BigInt64Array.from([1n]), dims: [1], dispose: vi.fn() };
    const create = vi.fn(async () => ({
      release,
      run: async () => ({ features, features_lens: lengths }),
    }));
    const ort = {
      env: { wasm: {} },
      Tensor: class {
        constructor(
          readonly type: string,
          readonly data: ArrayBufferView,
          readonly dims: readonly number[],
        ) {}
      },
      InferenceSession: { create },
    };
    return {
      preprocessor: new OnnxNemoPreprocessor(
        ort as unknown as NemoOrtModuleLike,
        'https://example.test/processor.onnx',
        false,
        undefined,
        'cpu',
        3,
      ),
      audio: {
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: 160,
        durationSeconds: 0.01,
        channels: [new Float32Array(160)],
      },
      create,
      features,
      lengths,
    };
  }

  it('uses the requested CPU provider and releases copied output tensors', async () => {
    const setup = fixture(vi.fn());
    try {
      expect((await setup.preprocessor.process(setup.audio)).features[0]).toBe(0.5);
      expect(setup.create).toHaveBeenCalledWith(
        'https://example.test/processor.onnx',
        expect.objectContaining({
          executionProviders: ['cpu'],
          intraOpNumThreads: 3,
          interOpNumThreads: 1,
        }),
      );
      expect(setup.features.dispose).toHaveBeenCalledTimes(1);
      expect(setup.lengths.dispose).toHaveBeenCalledTimes(1);
    } finally {
      await setup.preprocessor.dispose();
    }
  });

  it('waits for async release with concurrent idempotent disposal', async () => {
    let finishRelease = () => {};
    const gate = new Promise<void>((resolve) => {
      finishRelease = resolve;
    });
    const release = vi.fn(() => gate);
    const setup = fixture(release);
    await setup.preprocessor.process(setup.audio);
    let settled = false;
    const disposal = Promise.all([setup.preprocessor.dispose(), setup.preprocessor.dispose()]).then(
      () => {
        settled = true;
      },
    );
    try {
      await vi.waitFor(() => expect(release).toHaveBeenCalled());
      expect(release).toHaveBeenCalledTimes(1);
      expect(settled).toBe(false);
    } finally {
      finishRelease();
      await disposal;
    }
  });

  it('releases the session when abort is observed after InferenceSession.create', async () => {
    const signal = { aborted: false };
    const release = vi.fn();
    const ort = {
      env: { wasm: {} },
      Tensor: class {
        constructor(
          readonly type: string,
          readonly data: ArrayBufferView,
          readonly dims: readonly number[],
        ) {}
      },
      InferenceSession: {
        create: vi.fn(async () => {
          signal.aborted = true;
          return { release, run: vi.fn() };
        }),
      },
    };
    const preprocessor = new OnnxNemoPreprocessor(
      ort as unknown as NemoOrtModuleLike,
      'https://example.com/preprocessor.onnx',
      false,
      signal,
    );

    await expect(
      preprocessor.process({
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: 160,
        durationSeconds: 0.01,
        channels: [new Float32Array(160)],
      }),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
    expect(release).toHaveBeenCalledTimes(1);
  });
});
