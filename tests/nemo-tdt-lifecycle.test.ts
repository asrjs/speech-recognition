import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  DEFAULT_NEMO_TDT_CLASSIFICATION,
  parseNemoTdtConfig,
} from '../src/models/nemo-tdt/config.js';
import { OrtNemoTdtExecutor } from '../src/models/nemo-tdt/executor.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';

const mocks = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock('../src/models/nemo-tdt/ort.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/models/nemo-tdt/ort.js')>()),
  initOrt: async () => ({ env: { wasm: {} } }),
  createOrtSession: mocks.create,
}));

function session(release = vi.fn<() => void | Promise<void>>()) {
  return { run: vi.fn(async () => ({})), release };
}

function executor() {
  return new OrtNemoTdtExecutor(
    'test/native-tdt',
    DEFAULT_NEMO_TDT_CLASSIFICATION,
    parseNemoTdtConfig('test/native-tdt'),
    'native-cpu',
    {
      source: {
        kind: 'direct',
        preprocessorBackend: 'js',
        artifacts: {
          encoderUrl: 'file:///test/encoder.onnx',
          decoderUrl: 'file:///test/decoder.onnx',
          tokenizerUrl: 'file:///test/vocab.txt',
        },
      },
    },
  );
}

beforeEach(() => {
  mocks.create.mockReset();
  vi.spyOn(ParakeetTokenizer, 'fromUrl').mockResolvedValue(
    new ParakeetTokenizer(['<blk>', 'test']),
  );
});
afterEach(() => vi.restoreAllMocks());

describe('NeMo TDT resource ownership (simulated, not model parity)', () => {
  it('releases a created encoder when decoder initialization fails', async () => {
    const encoder = session();
    const failure = new Error('decoder initialization failed');
    mocks.create.mockResolvedValueOnce(encoder).mockRejectedValueOnce(failure);
    const model = executor();
    await expect(model.ready()).rejects.toBe(failure);
    await model.dispose();
    expect(encoder.release).toHaveBeenCalledTimes(1);
  });

  it('waits for both async releases and makes disposal idempotent', async () => {
    let finishRelease = () => {};
    const gate = new Promise<void>((resolve) => {
      finishRelease = resolve;
    });
    const encoder = session(vi.fn(() => gate));
    const decoder = session();
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = executor();
    await model.ready();
    let settled = false;
    const disposal = model.dispose().then(() => {
      settled = true;
    });
    try {
      await vi.waitFor(() => expect(decoder.release).toHaveBeenCalledTimes(1));
      expect(settled).toBe(false);
    } finally {
      finishRelease();
      await disposal;
    }
    await model.dispose();
    expect(encoder.release).toHaveBeenCalledTimes(1);
    expect(decoder.release).toHaveBeenCalledTimes(1);
  });

  it.each(['sync', 'async'] as const)(
    'retains load and %s partial-cleanup failures',
    async (mode) => {
      const loadFailure = new Error('decoder initialization failed');
      const cleanupFailure = new Error('encoder release failed');
      const encoder = session(
        vi.fn(() => {
          if (mode === 'sync') throw cleanupFailure;
          return Promise.reject(cleanupFailure);
        }),
      );
      mocks.create.mockResolvedValueOnce(encoder).mockRejectedValueOnce(loadFailure);
      const model = executor();
      const failure = await model.ready().catch((error: unknown) => error);
      expect(failure).toBeInstanceOf(AggregateError);
      expect((failure as AggregateError).cause).toBe(loadFailure);
      expect((failure as AggregateError).errors).toEqual([loadFailure, cleanupFailure]);
      await model.dispose();
      expect(encoder.release).toHaveBeenCalledTimes(1);
    },
  );

  it('releases sibling resources even when a release fails, and reports the error', async () => {
    const failure = new Error('release failed');
    const encoder = session(
      vi.fn(() => {
        throw failure;
      }),
    );
    const decoder = session();
    mocks.create.mockResolvedValueOnce(encoder).mockResolvedValueOnce(decoder);
    const model = executor();
    await model.ready();
    await expect(model.dispose()).rejects.toBeInstanceOf(AggregateError);
    expect(decoder.release).toHaveBeenCalledTimes(1);
  });

  it('retains the disposal-race cause when partial encoder cleanup rejects', async () => {
    const cleanupFailure = new Error('encoder release failed during disposal');
    const encoder = session(
      vi.fn(async () => {
        throw cleanupFailure;
      }),
    );
    let finishEncoder = () => {};
    const acquisition = new Promise<ReturnType<typeof session>>((resolve) => {
      finishEncoder = () => resolve(encoder);
    });
    mocks.create.mockReturnValueOnce(acquisition);
    const model = executor();
    const readiness = model.ready().catch((error: unknown) => error);
    await vi.waitFor(() => expect(mocks.create).toHaveBeenCalledTimes(1));
    const disposal = model.dispose();
    finishEncoder();
    const failure = await readiness;
    await disposal;
    expect(failure).toBeInstanceOf(AggregateError);
    const aggregate = failure as AggregateError;
    expect(aggregate.cause).toBeInstanceOf(Error);
    expect((aggregate.cause as Error).message).toContain('disposed during load');
    expect(aggregate.errors).toEqual([aggregate.cause, cleanupFailure]);
    expect(encoder.release).toHaveBeenCalledTimes(1);
    expect(mocks.create).toHaveBeenCalledTimes(1);
  });
});
