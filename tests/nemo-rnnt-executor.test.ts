import { describe, expect, it } from 'vitest';

import {
  DEFAULT_NEMO_RNNT_CLASSIFICATION,
  parseNemoRnntConfig,
} from '../src/models/nemo-rnnt/config.js';
import { OrtNemoRnntExecutor } from '../src/models/nemo-rnnt/executor.js';
import { ParakeetTokenizer } from '../src/models/nemo-rnnt/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/nemo-rnnt/ort.js';
import type { NemoRnntArtifactSource, NemoRnntModelConfig } from '../src/models/nemo-rnnt/types.js';
import type { AudioBufferLike } from '../src/types/index.js';

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
      outputs: new MockTensor(
        new Float32Array([...new Array(step.logits.length).fill(-10), ...step.logits]),
        [1, 1, 2, step.logits.length],
      ),
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
  readonly config?: Partial<NemoRnntModelConfig>;
  readonly frameCount?: number;
  readonly featureSize?: number;
  readonly logits: readonly MockDecoderStep[];
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
  const decoderSession = new MockDecoderSession(options.logits, [
    config.predictionLayers ?? 1,
    1,
    config.predictionHiddenSize ?? 4,
  ]);
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
});
