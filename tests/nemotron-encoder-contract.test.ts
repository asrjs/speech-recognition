import { beforeEach, describe, expect, it, vi } from 'vitest';

const testMocks = vi.hoisted(() => ({
  createOrtSession: vi.fn(),
  initOrt: vi.fn(),
  releaseOrtSession: vi.fn(),
  loadTokenizer: vi.fn(),
}));

vi.mock('../src/models/nemo-tdt/ort.js', () => ({
  createOrtSession: testMocks.createOrtSession,
  initOrt: testMocks.initOrt,
  releaseOrtSession: testMocks.releaseOrtSession,
  disposeOrtOutputs: vi.fn(),
}));

vi.mock('../src/models/nemo-tdt/tokenizer.js', () => ({
  ParakeetTokenizer: { fromUrl: testMocks.loadTokenizer },
}));

import { OrtNemotronRnntExecutor } from '../src/models/nemotron-rnnt/executor.js';
import { parseNemotronRnntConfig } from '../src/models/nemotron-rnnt/config.js';

const legacyEncoderInputs = [
  'audio_signal',
  'length',
  'cache_last_channel',
  'cache_last_time',
  'cache_last_channel_len',
  'lang_id',
];

const directSource = {
  kind: 'direct' as const,
  artifacts: {
    encoderUrl: 'encoder.onnx',
    decoderUrl: 'decoder.onnx',
    jointUrl: 'joint.onnx',
    tokenizerUrl: 'tokens.txt',
  },
};

function createExecutor() {
  return new OrtNemotronRnntExecutor(
    'nemotron-test',
    {} as never,
    parseNemotronRnntConfig('nemotron-test'),
    'wasm',
    directSource,
  );
}

describe('Nemotron encoder graph contract', () => {
  beforeEach(() => {
    testMocks.createOrtSession.mockReset();
    testMocks.releaseOrtSession.mockReset().mockResolvedValue(undefined);
    testMocks.initOrt.mockReset().mockResolvedValue({ Tensor: class {} });
    testMocks.loadTokenizer.mockReset().mockResolvedValue({ vocabSize: 13088 });
  });

  it('rejects an input_features encoder before loading decoder or joint and releases it', async () => {
    const encoderSession = { inputNames: ['input_features', 'prompt_ids'] };
    testMocks.createOrtSession.mockResolvedValue(encoderSession);
    const executor = createExecutor();

    await expect(executor.ready()).rejects.toThrow(
      /Unsupported Nemotron encoder graph: missing required legacy RNNT inputs/,
    );

    expect(testMocks.createOrtSession).toHaveBeenCalledTimes(1);
    expect(testMocks.releaseOrtSession).toHaveBeenCalledExactlyOnceWith(encoderSession);
    await executor.dispose();
    expect(testMocks.releaseOrtSession).toHaveBeenCalledExactlyOnceWith(encoderSession);
  });

  it('fails closed and releases an encoder whose input metadata is unavailable', async () => {
    const encoderSession = { inputNames: undefined };
    testMocks.createOrtSession.mockResolvedValue(encoderSession);
    const executor = createExecutor();

    await expect(executor.ready()).rejects.toThrow(
      /Cannot validate Nemotron encoder graph: ONNX session metadata did not expose inputNames/,
    );

    expect(testMocks.createOrtSession).toHaveBeenCalledTimes(1);
    expect(testMocks.releaseOrtSession).toHaveBeenCalledExactlyOnceWith(encoderSession);
    await executor.dispose();
  });

  it('rejects extra inputs that this executor cannot feed', async () => {
    const encoderSession = { inputNames: [...legacyEncoderInputs, 'input_features'] };
    testMocks.createOrtSession.mockResolvedValue(encoderSession);
    const executor = createExecutor();

    await expect(executor.ready()).rejects.toThrow(/unsupported inputs \(input_features\)/);
    expect(testMocks.createOrtSession).toHaveBeenCalledTimes(1);
    expect(testMocks.releaseOrtSession).toHaveBeenCalledExactlyOnceWith(encoderSession);
    await executor.dispose();
    expect(testMocks.releaseOrtSession).toHaveBeenCalledExactlyOnceWith(encoderSession);
  });

  it('allows an encoder with all six legacy inputs to proceed to decoder and joint loading', async () => {
    const encoderSession = { inputNames: legacyEncoderInputs };
    const predictorSession = { inputNames: ['targets', 'h_in', 'c_in'] };
    const jointSession = { inputNames: ['encoder_output', 'decoder_output'] };
    testMocks.createOrtSession
      .mockResolvedValueOnce(encoderSession)
      .mockResolvedValueOnce(predictorSession)
      .mockResolvedValueOnce(jointSession);
    const executor = createExecutor();

    await expect(executor.ready()).resolves.toBeUndefined();
    expect(testMocks.createOrtSession).toHaveBeenCalledTimes(3);

    await executor.dispose();
    expect(testMocks.releaseOrtSession).toHaveBeenCalledTimes(3);
  });
});
