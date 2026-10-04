import { beforeEach, describe, expect, it, vi } from 'vitest';

const loadOmnilingualCtc = vi.fn();

vi.mock('../src/models/omnilingual-ctc/model.js', () => ({
  loadOmnilingualCtc: (...args: unknown[]) => loadOmnilingualCtc(...args),
}));

import { createOmnilingualCtcModelFamily } from '../src/models/omnilingual-ctc/factory.js';
import type { AudioInputLike } from '../src/types/index.js';

function fakeModel() {
  let disposed = false;
  return {
    dispose: vi.fn(() => {
      disposed = true;
    }),
    transcribe: vi.fn(async () => {
      if (disposed) {
        throw new Error('model disposed');
      }
      return { utteranceText: 'hello', isFinal: true };
    }),
    get isDisposed() {
      return disposed;
    },
  };
}

const AUDIO: AudioInputLike = new Float32Array([0, 0.1, -0.1, 0.2]);

function context() {
  return {
    runtime: {} as never,
    backend: { id: 'native-cpu' } as never,
    hooks: {} as never,
    signal: null,
  };
}

describe('omnilingual-ctc model factory lifecycle', () => {
  beforeEach(() => {
    loadOmnilingualCtc.mockReset();
  });

  it('gives each session its own graph so a disposed session cannot poison the next', async () => {
    const models = [fakeModel(), fakeModel()];
    loadOmnilingualCtc.mockImplementation(async () => models.shift()!);
    const factory = createOmnilingualCtcModelFamily();
    const model = await factory.createModel(
      {
        family: 'omnilingual-ctc',
        modelId: 'test/omni',
        options: { modelUrl: 'm', tokensUrl: 't' },
      },
      context(),
    );

    const first = await model.createSession();
    const second = await model.createSession();
    expect(loadOmnilingualCtc).toHaveBeenCalledTimes(2);

    // Disposing the first session must not affect the second.
    await first.dispose();
    await expect(second.transcribe(AUDIO)).resolves.toBeDefined();

    await second.dispose();
    await model.dispose();
  });

  it('disposes every live session when the model is disposed', async () => {
    const models = [fakeModel(), fakeModel()];
    loadOmnilingualCtc.mockImplementation(async () => models.shift()!);
    const factory = createOmnilingualCtcModelFamily();
    const model = await factory.createModel(
      {
        family: 'omnilingual-ctc',
        modelId: 'test/omni',
        options: { modelUrl: 'm', tokensUrl: 't' },
      },
      context(),
    );

    const first = await model.createSession();
    const second = await model.createSession();
    await model.dispose();

    expect(models).toHaveLength(0);
    expect(loadOmnilingualCtc.mock.results.length).toBe(2);
    await expect(first.transcribe(AUDIO)).rejects.toThrow('model disposed');
    await expect(second.transcribe(AUDIO)).rejects.toThrow('model disposed');
  });

  it('refuses to create a session after the model is disposed', async () => {
    loadOmnilingualCtc.mockImplementation(async () => fakeModel());
    const factory = createOmnilingualCtcModelFamily();
    const model = await factory.createModel(
      {
        family: 'omnilingual-ctc',
        modelId: 'test/omni',
        options: { modelUrl: 'm', tokensUrl: 't' },
      },
      context(),
    );

    await model.dispose();
    await expect(model.createSession()).rejects.toThrow('was disposed');
  });
});
