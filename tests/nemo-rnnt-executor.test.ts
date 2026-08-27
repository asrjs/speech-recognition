import { beforeEach, describe, expect, it, vi } from 'vitest';

import { PipelineAbortedError } from '../src/pipeline/composition.js';
import { fetchModelFiles } from '../src/runtime/huggingface.js';
import {
  DEFAULT_NEMO_RNNT_CLASSIFICATION,
  parseNemoRnntConfig,
} from '../src/models/nemo-rnnt/config.js';
import { OrtNemoRnntExecutor } from '../src/models/nemo-rnnt/executor.js';
import { ParakeetTokenizer } from '../src/models/nemo-rnnt/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/nemo-rnnt/ort.js';
import type { NemoRnntArtifactSource, NemoRnntModelConfig } from '../src/models/nemo-rnnt/types.js';
import type { AssetProvider, AudioBufferLike, ResolvedAssetHandle } from '../src/types/index.js';

vi.mock('../src/runtime/huggingface.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/runtime/huggingface.js')>();
  return {
    ...actual,
    fetchModelFiles: vi.fn(actual.fetchModelFiles),
  };
});

class MockTensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
  disposed = false;

  constructor(
    readonly data: TData,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed = true;
  }
}

interface MockDecoderStep {
  readonly logits: readonly number[];
  readonly outputDims?: readonly number[];
}

class MockEncoderSession implements OrtSessionLike {
  constructor(private readonly encoderTensor: OrtTensorLike<Float32Array>) {}

  async run(): Promise<Record<string, OrtTensorLike>> {
    return {
      outputs: this.encoderTensor,
    };
  }
}

class MockDecoderSession implements OrtSessionLike {
  readonly targetHistory: number[] = [];
  readonly emittedStates: MockTensor<Float32Array>[] = [];
  readonly emittedLogits: MockTensor<Float32Array>[] = [];
  private callIndex = 0;

  constructor(
    private readonly steps: readonly MockDecoderStep[],
    private readonly stateDims: readonly number[],
    private readonly throwOnCallIndex?: number,
  ) {}

  resetCalls(): void {
    this.callIndex = 0;
    this.targetHistory.length = 0;
    this.emittedStates.length = 0;
    this.emittedLogits.length = 0;
  }

  async run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> {
    if (this.throwOnCallIndex && this.callIndex + 1 === this.throwOnCallIndex) {
      this.callIndex += 1;
      throw new Error(`Forced decoder failure on invocation #${this.callIndex}.`);
    }
    const step = this.steps[this.callIndex];
    this.callIndex += 1;
    if (!step) {
      throw new Error(`Unexpected decoder invocation #${this.callIndex}.`);
    }

    const targetTensor = feeds.targets as OrtTensorLike<Int32Array>;
    this.targetHistory.push(Number(targetTensor.data[0] ?? -1));

    const outputState1 = new MockTensor(
      new Float32Array(this.stateDims.reduce((size, dim) => size * dim, 1)),
      this.stateDims,
    );
    const outputState2 = new MockTensor(
      new Float32Array(this.stateDims.reduce((size, dim) => size * dim, 1)),
      this.stateDims,
    );
    this.emittedStates.push(outputState1, outputState2);

    const outputLogits = new MockTensor(
      new Float32Array([...new Array(step.logits.length).fill(-10), ...step.logits]),
      step.outputDims ?? [1, 1, 2, step.logits.length],
    );
    this.emittedLogits.push(outputLogits);

    return {
      outputs: outputLogits,
      output_states_1: outputState1,
      output_states_2: outputState2,
    };
  }
}

function createMockOrt(): OrtModuleLike {
  class RuntimeTensor<TData extends ArrayBufferView> extends MockTensor<TData> {
    readonly type: 'float32' | 'int32' | 'int64';

    constructor(type: 'float32' | 'int32' | 'int64', data: TData, dims: readonly number[]) {
      super(data, dims);
      this.type = type;
    }
  }

  return {
    env: {
      wasm: {},
    },
    Tensor: RuntimeTensor,
    InferenceSession: {
      async create(): Promise<OrtSessionLike> {
        throw new Error('MockOrt.InferenceSession.create should not be called in executor tests.');
      },
    },
  };
}

function createAudio(sampleRate = 16000, frames = 1600): AudioBufferLike {
  const mono = new Float32Array(frames);
  return {
    sampleRate,
    numberOfChannels: 1,
    numberOfFrames: frames,
    durationSeconds: frames / sampleRate,
    channels: [mono],
  };
}

function createResolvedHandle(filename: string): ResolvedAssetHandle {
  return {
    request: {
      id: filename,
      filename,
    },
    async *openStream() {
      yield new Uint8Array();
    },
    async readBytes() {
      return new Uint8Array();
    },
    async readText() {
      return '';
    },
    async readJson<T>() {
      return {} as T;
    },
    async getLocator(target) {
      return target === 'url' ? `blob:test/${filename}` : null;
    },
    dispose() {},
  };
}

function createRecordingAssetProvider(requests: string[]): AssetProvider {
  return {
    canResolve: () => true,
    async resolve(request) {
      const filename = request.filename ?? '';
      requests.push(filename);
      return createResolvedHandle(filename);
    },
  };
}

function createExecutorHarness(options: {
  readonly config?: Partial<NemoRnntModelConfig>;
  readonly frameCount?: number;
  readonly featureSize?: number;
  readonly logits: readonly MockDecoderStep[];
  readonly throwOnDecoderCallIndex?: number;
  readonly vocab?: readonly string[];
  readonly source?: NemoRnntArtifactSource;
}) {
  const vocab = options.vocab ?? ['▁hello', '▁world', '<EOU>'];
  const config = parseNemoRnntConfig('test-nemo-rnnt', {
    subsamplingFactor: 4,
    frameShiftSeconds: 0.01,
    melBins: 2,
    vocabularySize: vocab.length,
    predictionLayers: 1,
    predictionHiddenSize: 4,
    tokenizer: {
      kind: 'sentencepiece',
      blankTokenId: vocab.length,
    },
    ...options.config,
  });
  const tokenizer = new ParakeetTokenizer(vocab, {
    blankId: config.tokenizer.blankTokenId,
  });
  const frameCount = options.frameCount ?? 2;
  const featureSize = options.featureSize ?? 4;
  const encoderData = new Float32Array(featureSize * frameCount);
  for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
    for (let featureIndex = 0; featureIndex < featureSize; featureIndex += 1) {
      encoderData[featureIndex * frameCount + frameIndex] = frameIndex + featureIndex / 10;
    }
  }

  const encoderSession = new MockEncoderSession(
    new MockTensor(encoderData, [1, featureSize, frameCount]),
  );
  const decoderSession = new MockDecoderSession(
    options.logits,
    [config.predictionLayers ?? 1, 1, config.predictionHiddenSize ?? 4],
    options.throwOnDecoderCallIndex,
  );
  const executor = new OrtNemoRnntExecutor(
    'test-nemo-rnnt',
    DEFAULT_NEMO_RNNT_CLASSIFICATION,
    config,
    'wasm',
    options.source ? { source: options.source } : undefined,
  ) as OrtNemoRnntExecutor & { loadStatePromise?: Promise<unknown> };

  executor.loadStatePromise = Promise.resolve({
    ort: createMockOrt(),
    tokenizer,
    encoderSession,
    decoderSession,
    preprocessorBackend: options.source?.preprocessorBackend ?? 'onnx',
    preprocessor: {
      async process() {
        return {
          features: new Float32Array(config.melBins * frameCount),
          frameCount,
          validLength: frameCount,
        };
      },
    },
    warnings: [],
  });

  return {
    config,
    tokenizer,
    executor,
    decoderSession,
  };
}

beforeEach(() => {
  vi.mocked(fetchModelFiles).mockReset();
  vi.mocked(fetchModelFiles).mockResolvedValue([]);
});

describe('nemo-rnnt Hugging Face artifact materialization', () => {
  it('skips optional external-data probes when the repo listing shows the sidecars are absent', async () => {
    vi.mocked(fetchModelFiles).mockResolvedValue([
      'encoder-model.fp16.onnx',
      'decoder_joint-model.int8.onnx',
      'vocab.txt',
    ]);

    const requests: string[] = [];
    const config = parseNemoRnntConfig('test-nemo-rnnt', {
      subsamplingFactor: 4,
      frameShiftSeconds: 0.01,
      melBins: 2,
      vocabularySize: 3,
      predictionLayers: 1,
      predictionHiddenSize: 4,
      tokenizer: {
        kind: 'sentencepiece',
        blankTokenId: 3,
      },
    });
    const executor = new OrtNemoRnntExecutor(
      'test-nemo-rnnt',
      DEFAULT_NEMO_RNNT_CLASSIFICATION,
      config,
      'wasm',
      undefined,
      { assetProvider: createRecordingAssetProvider(requests) },
    ) as OrtNemoRnntExecutor & {
      sourceOptions: NemoRnntArtifactSource;
      materializeHuggingFaceArtifacts(
        artifacts: Record<string, string | undefined>,
      ): Promise<Record<string, string | undefined>>;
    };

    executor.sourceOptions = {
      kind: 'huggingface',
      repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
      revision: 'main',
      preprocessorBackend: 'js',
    };

    const artifacts = await executor.materializeHuggingFaceArtifacts({
      encoderUrl: 'https://example.test/encoder-model.fp16.onnx',
      decoderUrl: 'https://example.test/decoder_joint-model.int8.onnx',
      tokenizerUrl: 'https://example.test/vocab.txt',
      encoderFilename: 'encoder-model.fp16.onnx',
      decoderFilename: 'decoder_joint-model.int8.onnx',
    });

    expect(fetchModelFiles).toHaveBeenCalledWith('ysdede/parakeet-tdt-0.6b-v3-onnx', 'main');
    expect(requests).toEqual([
      'encoder-model.fp16.onnx',
      'decoder_joint-model.int8.onnx',
      'vocab.txt',
    ]);
    expect(artifacts.encoderDataUrl).toBeUndefined();
    expect(artifacts.decoderDataUrl).toBeUndefined();
  });
});

describe('nemo-rnnt executor decode loop', () => {
  it('emits multiple symbols on one frame, advances on blank, and strips EOU from user text', async () => {
    const harness = createExecutorHarness({
      logits: [
        { logits: [10.0, 0.0, 0.0, 0.0] },
        { logits: [0.0, 10.0, 0.0, 0.0] },
        { logits: [0.0, 0.0, 0.0, 10.0] },
        { logits: [0.0, 0.0, 10.0, 0.0] },
        { logits: [0.0, 0.0, 0.0, 10.0] },
      ],
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        returnTokenIds: true,
        returnFrameIndices: true,
        returnLogProbs: true,
        returnDecoderState: true,
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('hello world');
    expect(result.rawUtteranceText).toBe('hello world<EOU>');
    expect(result.control).toEqual({
      containsEou: true,
      containsEob: false,
      eouTokenId: 2,
      eobTokenId: undefined,
    });
    expect(result.words).toEqual([
      expect.objectContaining({ text: 'hello', startTime: 0, endTime: 0 }),
      expect.objectContaining({ text: 'world', startTime: 0, endTime: 0 }),
    ]);
    expect(result.tokens).toEqual([
      expect.objectContaining({
        id: 0,
        text: 'hello',
        frameIndex: 0,
      }),
      expect.objectContaining({
        id: 1,
        text: 'world',
        frameIndex: 0,
      }),
    ]);
    expect(result.specialTokens).toEqual([
      expect.objectContaining({
        id: 2,
        text: '<EOU>',
        frameIndex: 1,
        kind: 'eou',
      }),
    ]);
    expect(result.debug?.tokenIds).toEqual([0, 1, 2]);
    expect(result.debug?.frameIndices).toEqual([0, 0, 1]);
    expect(result.decoderState?.dims1).toEqual([1, 1, 4]);
    expect(harness.decoderSession.targetHistory).toEqual([3, 0, 1, 1, 2]);
  });

  it('keeps BDT encoder layout for long utterances where frame count exceeds feature size', async () => {
    const harness = createExecutorHarness({
      frameCount: 5,
      featureSize: 2,
      logits: [
        { logits: [0.0, 0.0, 10.0] },
        { logits: [0.0, 0.0, 10.0] },
        { logits: [0.0, 0.0, 10.0] },
        { logits: [0.0, 0.0, 10.0] },
        { logits: [10.0, 0.0, 0.0] },
        { logits: [0.0, 0.0, 10.0] },
      ],
      vocab: ['▁hello', '<EOU>'],
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        returnTokenIds: true,
        returnFrameIndices: true,
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('hello');
    expect(result.tokens).toEqual([
      expect.objectContaining({
        id: 0,
        text: 'hello',
        frameIndex: 4,
      }),
    ]);
    expect(result.debug?.frameIndices).toEqual([4]);
  });

  it('disposes decoder state tensors when decoding fails after state has advanced', async () => {
    const harness = createExecutorHarness({
      logits: [{ logits: [10.0, 0.0, 0.0] }],
      vocab: ['▁hello', '<EOU>'],
      throwOnDecoderCallIndex: 2,
    });

    await expect(
      harness.executor.transcribe(
        createAudio(),
        {
          returnDecoderState: true,
        },
        {} as never,
      ),
    ).rejects.toThrow('Forced decoder failure');

    expect(harness.decoderSession.emittedStates).toHaveLength(2);
    expect(harness.decoderSession.emittedStates.every((state) => state.disposed)).toBe(true);
  });

  it('disposes transient decoder tensors when post-run processing throws', async () => {
    const harness = createExecutorHarness({
      logits: [{ logits: [10.0, 0.0, 0.0], outputDims: [1, 1, 1, 1] }],
      vocab: ['▁hello', '<EOU>'],
    });

    await expect(
      harness.executor.transcribe(
        createAudio(),
        {
          returnDecoderState: true,
        },
        {} as never,
      ),
    ).rejects.toThrow('decoder output is too small');

    expect(harness.decoderSession.emittedLogits).toHaveLength(1);
    expect(harness.decoderSession.emittedLogits[0]?.disposed).toBe(true);
    expect(harness.decoderSession.emittedStates).toHaveLength(2);
    expect(harness.decoderSession.emittedStates.every((state) => state.disposed)).toBe(true);
  });

  it('stops the joint/decoder loop on abort, disposes tensors, and can decode again', async () => {
    const harness = createExecutorHarness({
      logits: [
        { logits: [10.0, 0.0, 0.0, 0.0] },
        { logits: [0.0, 10.0, 0.0, 0.0] },
        { logits: [0.0, 0.0, 0.0, 10.0] },
        { logits: [0.0, 0.0, 10.0, 0.0] },
        { logits: [0.0, 0.0, 0.0, 10.0] },
      ],
    });
    const signal = { aborted: false };
    let abortAfterFirst = true;
    const originalRun = harness.decoderSession.run.bind(harness.decoderSession);
    harness.decoderSession.run = async (feeds: Record<string, unknown>) => {
      const result = await originalRun(feeds);
      if (abortAfterFirst && harness.decoderSession.targetHistory.length === 1) signal.aborted = true;
      return result;
    };

    await expect(
      harness.executor.transcribe(createAudio(), { signal }, {} as never),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(harness.decoderSession.targetHistory).toHaveLength(1);
    expect(harness.decoderSession.emittedLogits).toHaveLength(1);
    expect(harness.decoderSession.emittedLogits.every((tensor) => tensor.disposed)).toBe(true);
    expect(harness.decoderSession.emittedStates.every((state) => state.disposed)).toBe(true);

    abortAfterFirst = false;
    signal.aborted = false;
    harness.decoderSession.resetCalls();
    const result = await harness.executor.transcribe(
      createAudio(),
      { returnTokenIds: true },
      {} as never,
    );
    expect(result.utteranceText).toBe('hello world');
    expect(result.debug?.tokenIds).toEqual([0, 1, 2]);
    await harness.executor.dispose();
  });
});
