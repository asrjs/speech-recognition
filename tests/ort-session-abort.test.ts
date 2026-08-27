import { describe, expect, it, vi } from 'vitest';

import { AssetLoadAbortedError } from '../src/io/abort.js';
import { createOrtSession, type OrtModuleLike } from '../src/models/lasr-ctc/ort.js';
import { OnnxNemoPreprocessor } from '../src/models/nemo-tdt/preprocessor.js';
import type { OrtModuleLike as NemoOrtModuleLike } from '../src/models/nemo-tdt/ort.js';

describe('createOrtSession abort', () => {
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

describe('OnnxNemoPreprocessor abort', () => {
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
