import { describe, expect, it } from 'vitest';

import { confidenceFromLogits } from '../src/inference/index.js';
import {
  DEFAULT_NEMO_TDT_CLASSIFICATION,
  parseNemoTdtConfig,
} from '../src/models/nemo-tdt/config.js';
import { OrtNemoTdtExecutor } from '../src/models/nemo-tdt/executor.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/nemo-tdt/ort.js';
import type { NemoTdtArtifactSource, NemoTdtModelConfig } from '../src/models/nemo-tdt/types.js';
import type { AudioBufferLike, TranscriptionProgressEvent } from '../src/types/index.js';

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
  private callIndex = 0;

  constructor(
    private readonly steps: readonly MockDecoderStep[],
    private readonly stateDims: readonly number[],
  ) {}

  async run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> {
    const step = this.steps[this.callIndex];
    this.callIndex += 1;
    if (!step) {
      throw new Error(`Unexpected decoder invocation #${this.callIndex}.`);
    }

    const targetTensor = feeds.targets as OrtTensorLike<Int32Array>;
    this.targetHistory.push(Number(targetTensor.data[0] ?? -1));

    return {
      outputs: new MockTensor(new Float32Array(step.logits), [1, step.logits.length]),
      output_states_1: new MockTensor(
        new Float32Array(this.stateDims.reduce((size, dim) => size * dim, 1)),
        this.stateDims,
      ),
      output_states_2: new MockTensor(
        new Float32Array(this.stateDims.reduce((size, dim) => size * dim, 1)),
        this.stateDims,
      ),
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

function createExecutorHarness(options: {
  readonly config?: Partial<NemoTdtModelConfig>;
  readonly frameCount?: number;
  readonly featureSize?: number;
  readonly logits: readonly MockDecoderStep[];
  readonly vocab?: readonly string[];
  readonly source?: NemoTdtArtifactSource;
}) {
  const config = parseNemoTdtConfig('test-nemo-tdt', {
    subsamplingFactor: 4,
    frameShiftSeconds: 0.01,
    melBins: 2,
    vocabularySize: options.vocab?.length ?? 3,
    predictionLayers: 1,
    predictionHiddenSize: 4,
    ...options.config,
  });
  const tokenizer = new ParakeetTokenizer(options.vocab ?? ['<blk>', '▁hello', '▁world']);
  const frameCount = options.frameCount ?? 3;
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
  const decoderSession = new MockDecoderSession(options.logits, [
    config.predictionLayers ?? 1,
    1,
    config.predictionHiddenSize ?? 4,
  ]);
  const executor = new OrtNemoTdtExecutor(
    'test-nemo-tdt',
    DEFAULT_NEMO_TDT_CLASSIFICATION,
    config,
    'wasm',
    options.source ? { source: options.source } : undefined,
  ) as OrtNemoTdtExecutor & { loadStatePromise?: Promise<unknown> };

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

describe('nemo-tdt executor decode loop', () => {
  it('handles token duration steps and clamps timestamps to remaining frames', async () => {
    const harness = createExecutorHarness({
      frameCount: 3,
      logits: [
        { logits: [0.1, 10.0, 0.0, 8.0, 1.0, 0.5] },
        { logits: [9.0, 0.0, 0.0, 0.0, 8.0, 0.0] },
        { logits: [0.0, 0.0, 10.0, 0.0, 0.0, 9.0] },
      ],
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        returnTokenIds: true,
        returnFrameIndices: true,
        returnTdtSteps: true,
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('hello world');
    expect(result.tokens).toEqual([
      expect.objectContaining({
        id: 1,
        text: 'hello',
        startTime: 0,
        endTime: 0.04,
        frameIndex: 0,
        tdtStep: 0,
      }),
      expect.objectContaining({
        id: 2,
        text: 'world',
        startTime: 0.04,
        endTime: 0.12,
        frameIndex: 1,
        tdtStep: 2,
      }),
    ]);
    expect(result.words).toEqual([
      expect.objectContaining({ text: 'hello', startTime: 0, endTime: 0.04 }),
      expect.objectContaining({ text: 'world', startTime: 0.04, endTime: 0.12 }),
    ]);
    expect(result.debug?.tokenIds).toEqual([1, 2]);
    expect(result.debug?.frameIndices).toEqual([0, 1]);
    expect(result.debug?.tdtSteps).toEqual([0, 2]);
  });

  it('advances to the next frame after a blank decode step without emitting a token', async () => {
    const harness = createExecutorHarness({
      frameCount: 2,
      logits: [
        { logits: [9.0, 0.0, 0.0, 8.0, 1.0, 0.0] },
        { logits: [0.0, 10.0, 0.0, 0.0, 8.0, 1.0] },
      ],
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        returnFrameIndices: true,
        returnTdtSteps: true,
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('hello');
    expect(result.tokens).toEqual([
      expect.objectContaining({
        id: 1,
        startTime: 0.04,
        endTime: 0.08,
        frameIndex: 1,
      }),
    ]);
    expect(result.debug?.frameIndices).toEqual([1]);
    expect(result.debug?.tdtSteps).toEqual([1]);
    expect(harness.decoderSession.targetHistory).toEqual([0, 0]);
  });

  it('aggregates frame confidences per encoder frame instead of per decode step', async () => {
    const logitsA = [0.0, 4.0, -2.0, 9.0, 1.0, 0.0];
    const logitsB = [0.0, -1.0, 3.0, 9.0, 1.0, 0.0];
    const logitsC = [5.0, 0.0, 0.0, 0.0, 1.0, 9.0];
    const harness = createExecutorHarness({
      config: {
        maxSymbolsPerStep: 2,
      },
      frameCount: 3,
      logits: [{ logits: logitsA }, { logits: logitsB }, { logits: logitsC }],
    });

    const result = await harness.executor.transcribe(createAudio(), {}, {} as never);

    const expectedFrame0 =
      (confidenceFromLogits(new Float32Array(logitsA), 1, 3).confidence +
        confidenceFromLogits(new Float32Array(logitsB), 2, 3).confidence) /
      2;
    const expectedFrame1 = confidenceFromLogits(new Float32Array(logitsC), 0, 3).confidence;

    expect(result.confidence?.frames).toHaveLength(2);
    expect(result.confidence?.frames?.[0]).toBeCloseTo(expectedFrame0, 6);
    expect(result.confidence?.frames?.[1]).toBeCloseTo(expectedFrame1, 6);
    expect(result.confidence?.frameAverage).toBeCloseTo((expectedFrame0 + expectedFrame1) / 2, 6);
  });

  it('returns a decoder-state snapshot when requested', async () => {
    const harness = createExecutorHarness({
      config: {
        maxSymbolsPerStep: 1,
      },
      frameCount: 2,
      logits: [
        { logits: [0.1, 10.0, 0.0, 8.0, 1.0, 0.0] },
        { logits: [9.0, 0.0, 0.0, 0.0, 8.0, 0.0] },
      ],
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        returnDecoderState: true,
      },
      {} as never,
    );

    expect(result.decoderState).toBeDefined();
    expect(result.decoderState?.dims1).toEqual([1, 1, 4]);
    expect(result.decoderState?.dims2).toEqual([1, 1, 4]);
    expect(result.decoderState?.s1).toBeInstanceOf(Float32Array);
    expect(result.decoderState?.s2).toBeInstanceOf(Float32Array);
    expect(result.decoderState?.s1.length).toBe(4);
    expect(result.decoderState?.s2.length).toBe(4);
  });

  it('respects a non-default maxSymbolsPerStep by forcing frame advancement earlier', async () => {
    const harness = createExecutorHarness({
      config: {
        maxSymbolsPerStep: 1,
      },
      frameCount: 3,
      logits: [
        { logits: [0.0, 10.0, 0.0, 8.0, 1.0, 0.0] },
        { logits: [0.0, 0.0, 10.0, 8.0, 1.0, 0.0] },
        { logits: [9.0, 0.0, 0.0, 0.0, 8.0, 0.0] },
      ],
    });

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        returnFrameIndices: true,
        returnTdtSteps: true,
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('hello world');
    expect(result.debug?.frameIndices).toEqual([0, 1]);
    expect(result.tokens).toEqual([
      expect.objectContaining({
        id: 1,
        startTime: 0,
        endTime: 0.04,
        frameIndex: 0,
      }),
      expect.objectContaining({
        id: 2,
        startTime: 0.04,
        endTime: 0.08,
        frameIndex: 1,
      }),
    ]);
  });

  it('uses the tokenizer vocabulary size instead of stale config metadata when slicing decoder outputs', async () => {
    const harness = createExecutorHarness({
      config: {
        vocabularySize: 3,
      },
      frameCount: 2,
      vocab: ['▁The', '▁boy', '▁was', '▁there', '<blk>'],
      logits: [
        { logits: [0.0, 0.0, 0.0, 10.0, 0.0, 9.0, 1.0] },
        { logits: [0.0, 0.0, 0.0, 0.0, 10.0, 9.0, 1.0] },
        { logits: [0.0, 0.0, 0.0, 0.0, 10.0, 9.0, 1.0] },
      ],
    });

    const result = await harness.executor.transcribe(createAudio(), {}, {} as never);

    expect(result.utteranceText).toBe('there');
    expect(result.tokens).toEqual([
      expect.objectContaining({
        id: 3,
        text: 'there',
        startTime: 0,
        endTime: 0.04,
        frameIndex: 0,
      }),
    ]);
    expect(result.warnings).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: 'nemo-tdt.vocabulary-size-mismatch',
        }),
      ]),
    );
  });

  it('throws when decoder outputs omit the TDT duration head', async () => {
    const harness = createExecutorHarness({
      frameCount: 1,
      logits: [{ logits: [0.0, 10.0, 0.0] }],
    });

    await expect(harness.executor.transcribe(createAudio(), {}, {} as never)).rejects.toThrow(
      'missing required TDT duration logits',
    );
  });

  it('emits stage progress, decode progress, and metrics during transcription', async () => {
    const harness = createExecutorHarness({
      frameCount: 3,
      logits: [
        { logits: [0.1, 10.0, 0.0, 8.0, 1.0, 0.5] },
        { logits: [9.0, 0.0, 0.0, 0.0, 8.0, 0.0] },
        { logits: [0.0, 0.0, 10.0, 0.0, 0.0, 9.0] },
      ],
    });
    const events: TranscriptionProgressEvent[] = [];

    const result = await harness.executor.transcribe(
      createAudio(),
      {
        onProgress(event) {
          events.push(event);
        },
      },
      {} as never,
    );

    expect(result.utteranceText).toBe('hello world');
    expect(events[0]).toEqual(
      expect.objectContaining({
        stage: 'start',
        progress: 0,
        modelId: 'test-nemo-tdt',
        backendId: 'wasm',
      }),
    );
    expect(events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ stage: 'preprocess', progress: 0.2 }),
        expect.objectContaining({ stage: 'encode', progress: 0.4 }),
        expect.objectContaining({ stage: 'postprocess', progress: 0.95 }),
        expect.objectContaining({ stage: 'complete', progress: 1 }),
      ]),
    );

    const decodeEvents = events.filter((event) => event.stage === 'decode');
    expect(decodeEvents.length).toBeGreaterThan(0);
    expect(decodeEvents.some((event) => (event.completedUnits ?? 0) >= 1)).toBe(true);
    expect(decodeEvents.at(-1)?.completedUnits).toBe(3);
    expect(decodeEvents.at(-1)?.totalUnits).toBe(3);
    expect(decodeEvents.some((event) => typeof event.remainingMs === 'number')).toBe(true);
    expect(events.at(-1)?.metrics?.totalMs).toBeGreaterThan(0);
    expect(events.at(-1)?.metrics?.decodeMs).toBeGreaterThanOrEqual(0);
    expect(result.metrics?.wallMs).toBeGreaterThanOrEqual(result.metrics?.totalMs ?? 0);
    expect(result.metrics?.audioDurationSec).toBeCloseTo(0.1, 4);
    expect(result.metrics?.requestedPreprocessorBackend).toBe('onnx');
    expect(result.metrics?.preprocessorBackend).toBe('onnx');
    expect(result.metrics?.encoderFrameCount).toBe(3);
    expect(result.metrics?.decodeIterations).toBeGreaterThanOrEqual(3);
    expect(result.metrics?.emittedTokenCount).toBe(2);
    expect(result.metrics?.emittedWordCount).toBe(2);
  });

  it('reports requested and effective preprocessor backends when js preprocessing is enabled', async () => {
    const harness = createExecutorHarness({
      frameCount: 1,
      logits: [
        { logits: [0.0, 10.0, 0.0, 9.0, 1.0, 0.0] },
        { logits: [9.0, 0.0, 0.0, 8.0, 1.0, 0.0] },
      ],
      source: {
        kind: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v2-onnx',
        preprocessorBackend: 'js',
      },
    });

    const result = await harness.executor.transcribe(createAudio(), {}, {} as never);

    expect(result.metrics?.requestedPreprocessorBackend).toBe('js');
    expect(result.metrics?.preprocessorBackend).toBe('js');
  });
});
