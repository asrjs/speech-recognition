import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createWasmBackend } from '../src/inference/backends/wasm/index.js';
import { createNemoTdtModelFamily } from '../src/models/nemo-tdt/model.js';
import { createWhisperSeq2SeqModelFamily } from '../src/models/whisper-seq2seq/model.js';
import { createRuntimeHooks } from '../src/runtime/logging.js';
import { createSpeechRuntime } from '../src/runtime/session.js';

const executor = vi.hoisted(() => ({
  ready: vi.fn<() => Promise<void>>(),
  dispose: vi.fn<() => Promise<void>>(),
}));
vi.mock('../src/models/whisper-seq2seq/executor.js', () => ({
  WhisperOnnxExecutor: class {
    ready() {
      return executor.ready();
    }
    dispose() {
      return executor.dispose();
    }
  },
}));

beforeEach(() => {
  executor.ready.mockReset().mockResolvedValue();
  executor.dispose.mockReset().mockResolvedValue();
});
afterEach(() => vi.restoreAllMocks());

async function createModel(family: 'nemo-tdt' | 'whisper-seq2seq') {
  const context = {
    runtime: createSpeechRuntime(),
    backend: createWasmBackend(),
    hooks: createRuntimeHooks(),
  };
  if (family === 'nemo-tdt') {
    return createNemoTdtModelFamily({
      dependencies: {
        executor: {
          ready: executor.ready,
          dispose: executor.dispose,
          async transcribe() {
            return { utteranceText: '', isFinal: true, words: [], tokens: [] };
          },
        },
      },
    }).createModel({ family, modelId: 'test/tdt' }, context);
  }
  return createWhisperSeq2SeqModelFamily().createModel(
    {
      family,
      modelId: 'test/whisper',
      options: {
        source: {
          kind: 'direct',
          artifacts: {
            encoderUrl: 'file:///unused/encoder.onnx',
            decoderUrl: 'file:///unused/decoder.onnx',
            tokenizerUrl: 'file:///unused/tokenizer.json',
          },
        },
      },
    },
    context,
  );
}

describe.each(['nemo-tdt', 'whisper-seq2seq'] as const)(
  '%s failed session creation (simulated)',
  (family) => {
    it('cleans the unreturned session before propagating its initialization error', async () => {
      const failure = new Error('session initialization failed');
      executor.ready.mockRejectedValueOnce(failure);
      const model = await createModel(family);
      try {
        await expect(model.createSession()).rejects.toBe(failure);
        expect(executor.dispose).toHaveBeenCalledTimes(1);
      } finally {
        await model.dispose();
      }
      expect(executor.dispose).toHaveBeenCalledTimes(1);
    });

    it('retains initialization and cleanup failures without leaving hidden ownership', async () => {
      const failure = new Error('session initialization failed');
      const cleanupFailure = new Error('session cleanup failed');
      executor.ready.mockRejectedValueOnce(failure);
      executor.dispose.mockRejectedValueOnce(cleanupFailure);
      const model = await createModel(family);
      const outcome = await model.createSession().catch((error: unknown) => error);
      // Public dispose may return void or a promise; either must finish without
      // revisiting an unreturned failed session.
      await model.dispose();
      expect(outcome).toBeInstanceOf(AggregateError);
      expect((outcome as AggregateError).errors).toEqual([failure, cleanupFailure]);
      expect((outcome as AggregateError).cause).toBe(failure);
      expect(executor.dispose).toHaveBeenCalledTimes(1);
    });

    it('keeps a successfully initialized session owned until explicit disposal', async () => {
      const model = await createModel(family);
      const session = await model.createSession();
      expect(executor.dispose).not.toHaveBeenCalled();
      await session.dispose();
      await model.dispose();
      expect(executor.dispose).toHaveBeenCalledTimes(1);
    });

    it.each(['model', 'runtime'] as const)(
      'awaits every owned resource before reporting a %s cleanup failure',
      async (scope) => {
        const failure = new Error('first cleanup failed');
        let finishCleanup = () => {};
        const gate = new Promise<void>((resolve) => {
          finishCleanup = resolve;
        });
        const first = await createModel(family);
        const second = scope === 'model' ? first : await createModel(family);
        await first.createSession();
        await second.createSession();
        const runtime = createSpeechRuntime({ backends: [createWasmBackend()] });
        if (scope === 'runtime') {
          const models = [first, second];
          runtime.registerModelFamily({
            family,
            supports: () => true,
            async createModel() {
              return models.shift()!;
            },
          });
          await runtime.loadModel({ family, modelId: 'test/first', backend: 'wasm' });
          await runtime.loadModel({ family, modelId: 'test/second', backend: 'wasm' });
        }
        executor.dispose.mockRejectedValueOnce(failure).mockImplementationOnce(() => gate);
        const owner = scope === 'model' ? first : runtime;
        let settled = false;
        let repeatedSettled = false;
        const disposal = Promise.resolve(owner.dispose()).catch((error: unknown) => {
          settled = true;
          return error;
        });
        const repeated = Promise.resolve(owner.dispose()).catch((error: unknown) => {
          repeatedSettled = true;
          return error;
        });
        try {
          await vi.waitFor(() => expect(executor.dispose).toHaveBeenCalledTimes(2));
          await new Promise<void>((resolve) => setImmediate(resolve));
          expect(settled).toBe(false);
          expect(repeatedSettled).toBe(false);
        } finally {
          finishCleanup();
          await Promise.allSettled([disposal, repeated]);
          await Promise.allSettled([first.dispose(), second.dispose(), runtime.dispose()]);
        }
        expect(await disposal).toBe(failure);
        expect(await repeated).toBe(failure);
        expect(executor.dispose).toHaveBeenCalledTimes(2);
      },
    );

    it('rejects creation after model disposal before acquiring execution resources', async () => {
      const model = await createModel(family);
      await model.dispose();
      let createdSession: Awaited<ReturnType<typeof model.createSession>> | undefined;
      try {
        const creation = model.createSession().then((session) => {
          createdSession = session;
          return session;
        });
        await expect(creation).rejects.toMatchObject({ code: 'model-disposed' });
        expect(executor.ready).not.toHaveBeenCalled();
        expect(executor.dispose).not.toHaveBeenCalled();
      } finally {
        await createdSession?.dispose();
      }
    });

    it('rejects disposal-raced creation and awaits the same ongoing cleanup', async () => {
      let finishReady = () => {};
      let finishCleanup = () => {};
      const readyGate = new Promise<void>((resolve) => {
        finishReady = resolve;
      });
      const cleanupGate = new Promise<void>((resolve) => {
        finishCleanup = resolve;
      });
      executor.ready.mockImplementationOnce(() => readyGate);
      executor.dispose.mockImplementationOnce(() => cleanupGate);
      const model = await createModel(family);
      let creationSettled = false;
      const creation = model.createSession().then(
        (session) => {
          creationSettled = true;
          return session;
        },
        (error: unknown) => {
          creationSettled = true;
          return error;
        },
      );
      let repeatedDisposeSettled = false;
      let disposal: Promise<void> | undefined;
      let repeatedDisposal: Promise<void> | undefined;
      try {
        await vi.waitFor(() => expect(executor.ready).toHaveBeenCalledTimes(1));
        disposal = Promise.resolve(model.dispose());
        repeatedDisposal = Promise.resolve(model.dispose()).then(() => {
          repeatedDisposeSettled = true;
        });
        await vi.waitFor(() => expect(executor.dispose).toHaveBeenCalledTimes(1));
        finishReady();
        // Drain the promise queue while the explicitly controlled cleanup is
        // still pending; no wall-clock timeout determines lifecycle behavior.
        await new Promise<void>((resolve) => setImmediate(resolve));
        expect(creationSettled).toBe(false);
        expect(repeatedDisposeSettled).toBe(false);
      } finally {
        finishReady();
        finishCleanup();
        await Promise.allSettled([creation, disposal, repeatedDisposal]);
      }
      expect(await creation).toMatchObject({ code: 'model-disposed' });
      expect(executor.dispose).toHaveBeenCalledTimes(1);
    });
  },
);
