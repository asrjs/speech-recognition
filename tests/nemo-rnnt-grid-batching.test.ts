import { describe, expect, it } from 'vitest';

import {
  DEFAULT_NEMO_RNNT_CLASSIFICATION,
  parseNemoRnntConfig,
} from '../src/models/nemo-rnnt/config.js';
import { OrtNemoRnntExecutor } from '../src/models/nemo-rnnt/executor.js';
import { ParakeetTokenizer } from '../src/models/nemo-rnnt/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/nemo-rnnt/ort.js';
import type { NemoRnntModelConfig } from '../src/models/nemo-rnnt/types.js';
import type { AudioBufferLike } from '../src/types/index.js';

// Grid-capable mock of the fused NeMo RNNT decoder-joint graph. The real
// eou-120m v1 export scores every slot of a [1, features, width] encoder
// batch independently against the fed (target, recurrent state) pair,
// verified bit-exact against single-frame runs for the fp32/fp16 graphs.
// The mock derives each row from the same pure function of (frame, target)
// that a single-frame run observes and ASSERTS the feature-major layout so
// a transposed fill cannot pass silently.

const HELLO = 0;
const WORLD = 1;
const BLANK = 3; // blankTokenId = vocab.length
const VOCAB = ['hello', 'world', '<EOU>'];
const DIST = 4; // distributionSize = max(vocabSize, blankId + 1)
const ROW_LENGTH = 2 * DIST; // real export emits [1, width, 2, dist]

type RnntScript = (frame: number, target: number) => number;

class MockTensor implements OrtTensorLike<ArrayBufferView> {
  disposed = false;
  constructor(
    readonly data: ArrayBufferView,
    readonly dims: readonly number[],
  ) {}
  dispose(): void {
    this.disposed = true;
  }
}

interface GridCall {
  readonly width: number;
  readonly startFrame: number;
  readonly target: number;
}

class GridDecoderSession implements OrtSessionLike {
  readonly calls: GridCall[] = [];
  rejectRuns = 0;

  constructor(private readonly script: RnntScript) {}

  async run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> {
    if (this.rejectRuns > 0) {
      this.rejectRuns -= 1;
      throw new Error('simulated decoder rejection');
    }
    const encoder = feeds.encoder_outputs as OrtTensorLike<Float32Array>;
    const dims = [...encoder.dims];
    const width = dims[dims.length - 1]!;
    const target = Number((feeds.targets as OrtTensorLike<Int32Array>).data[0] ?? -1);
    // Feature 0 of the fixture stores the frame index; the executor must
    // feed a feature-major [1, features, width] tensor, so element (0, w)
    // at flat offset w identifies frame startFrame + w.
    const startFrame = Math.round(encoder.data[0] ?? -1);
    for (let column = 0; column < width; column += 1) {
      const frame = Math.round(encoder.data[column] ?? -1);
      if (frame !== startFrame + column) {
        throw new Error(
          'Grid layout violation at column ' + column + ' of run starting ' + startFrame,
        );
      }
    }
    this.calls.push({ width, startFrame, target });
    const values = new Float32Array(width * ROW_LENGTH).fill(-10);
    for (let row = 0; row < width; row += 1) {
      const token = this.script(startFrame + row, target);
      values[row * ROW_LENGTH + (ROW_LENGTH - DIST) + token] = 5;
    }
    const outputs = new MockTensor(values, [1, width, 2, DIST]);
    const state1 = new MockTensor(new Float32Array(4), [1, 1, 4]);
    const state2 = new MockTensor(new Float32Array(4), [1, 1, 4]);
    return { outputs, output_states_1: state1, output_states_2: state2 };
  }
}

function createMockOrt(): OrtModuleLike {
  class RuntimeTensor<TData extends ArrayBufferView> extends MockTensor {
    readonly type: 'float32' | 'int32' | 'int64';
    constructor(type: 'float32' | 'int32' | 'int64', data: TData, dims: readonly number[]) {
      super(data, dims);
      this.type = type;
    }
  }
  return {
    env: { wasm: {} },
    Tensor: RuntimeTensor,
    InferenceSession: {
      async create(): Promise<OrtSessionLike> {
        throw new Error('MockOrt.InferenceSession.create should not be called.');
      },
    },
  };
}

function createAudio(): AudioBufferLike {
  return {
    sampleRate: 16000,
    numberOfChannels: 1,
    numberOfFrames: 1600,
    durationSeconds: 0.1,
    channels: [new Float32Array(1600)],
  };
}

function createFixture(options: {
  readonly frameCount: number;
  readonly script: RnntScript;
  readonly config?: Partial<NemoRnntModelConfig>;
  readonly decoderQuantization?: 'fp32' | 'fp16' | 'int8';
}) {
  const config = parseNemoRnntConfig('test-rnnt-grid', {
    subsamplingFactor: 4,
    frameShiftSeconds: 0.01,
    melBins: 2,
    vocabularySize: VOCAB.length,
    predictionLayers: 1,
    predictionHiddenSize: 4,
    tokenizer: {
      kind: 'sentencepiece',
      blankTokenId: VOCAB.length,
    },
    ...options.config,
  });
  const featureSize = 8;
  // BDT layout ([1, features, frames]): feature 0 stores the frame index.
  const encoderData = new Float32Array(featureSize * options.frameCount);
  for (let frame = 0; frame < options.frameCount; frame += 1) {
    encoderData[frame] = frame;
  }
  const encoderSession = new (class implements OrtSessionLike {
    async run(): Promise<Record<string, OrtTensorLike>> {
      return { outputs: new MockTensor(encoderData, [1, featureSize, options.frameCount]) };
    }
  })();
  const decoderSession = new GridDecoderSession(options.script);
  const tokenizer = new ParakeetTokenizer(VOCAB, {
    blankId: config.tokenizer.blankTokenId,
  });
  const executor = new OrtNemoRnntExecutor(
    'test-rnnt-grid',
    DEFAULT_NEMO_RNNT_CLASSIFICATION,
    config,
    'wasm',
  ) as unknown as {
    transcribe(
      audio: AudioBufferLike,
      options?: Record<string, unknown>,
      context?: unknown,
    ): Promise<Record<string, unknown>>;
    dispose(): Promise<void>;
    rnntBatchAllowed: boolean;
    loadStatePromise?: Promise<unknown>;
  };
  executor.loadStatePromise = Promise.resolve({
    ort: createMockOrt(),
    tokenizer,
    encoderSession,
    decoderSession,
    preprocessorBackend: 'onnx',
    preprocessor: {
      async process() {
        return {
          features: new Float32Array(config.melBins * options.frameCount),
          frameCount: options.frameCount,
          validLength: options.frameCount,
        };
      },
    },
    warnings: [],
    decoderQuantization: options.decoderQuantization,
  });
  return { executor, decoderSession };
}

const TIMING_KEYS = [
  'preprocessMs', 'encodeMs', 'decodeMs', 'tokenizeMs',
  'totalMs', 'wallMs', 'rtf', 'rtfx',
] as const;

const slimMetrics = (metrics: unknown) => {
  const { decoderGridBatchRuns: _ignored, ...rest } = (metrics ?? {}) as Record<string, unknown>;
  for (const key of TIMING_KEYS) {
    delete rest[key];
  }
  return rest;
};

const tokenShape = (result: Record<string, unknown>) =>
  (result.tokens as Array<Record<string, unknown>>).map((token) => [
    token.text,
    token.startTime,
    token.endTime,
  ]);

describe('nemo-rnnt speculative grid batching', () => {
  it('matches the forced-sequential transcript while collapsing blank runs', async () => {
    const script: RnntScript = (frame, target) =>
      frame === 6 && target === BLANK ? HELLO : BLANK;
    const sequential = createFixture({ frameCount: 12, script });
    const batched = createFixture({ frameCount: 12, script });
    const seqResult = await sequential.executor.transcribe(
      createAudio(),
      { gridBatching: false },
      {} as never,
    );
    const batchResult = await batched.executor.transcribe(
      createAudio(),
      { gridBatching: true },
      {} as never,
    );

    expect(batchResult.utteranceText).toBe(seqResult.utteranceText);
    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    expect(slimMetrics(batchResult.metrics)).toEqual(slimMetrics(seqResult.metrics));
    expect(sequential.decoderSession.calls.length).toBe(13);
    expect(batched.decoderSession.calls.length).toBeLessThan(6);
    expect((batchResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(batched.decoderSession.calls.length);
    expect((seqResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(0);
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it('supports multi-token frames by re-batching from the emitting frame', async () => {
    const script: RnntScript = (frame, target) => {
      if (frame === 4 && target === BLANK) return HELLO;
      if (frame === 4 && target === HELLO) return WORLD;
      return BLANK;
    };
    const sequential = createFixture({ frameCount: 8, script });
    const batched = createFixture({ frameCount: 8, script });
    const seqResult = await sequential.executor.transcribe(
      createAudio(),
      { gridBatching: false },
      {} as never,
    );
    const batchResult = await batched.executor.transcribe(
      createAudio(),
      { gridBatching: true },
      {} as never,
    );

    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    expect(slimMetrics(batchResult.metrics)).toEqual(slimMetrics(seqResult.metrics));
    expect(batched.decoderSession.calls.length).toBeLessThan(
      sequential.decoderSession.calls.length,
    );
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it('latches batching off after the sampling window when rows are emission-dense', async () => {
    const script: RnntScript = () => HELLO;
    const sequential = createFixture({ frameCount: 24, script });
    const batched = createFixture({ frameCount: 24, script });
    const seqResult = await sequential.executor.transcribe(
      createAudio(),
      { gridBatching: false },
      {} as never,
    );
    const batchResult = await batched.executor.transcribe(
      createAudio(),
      { gridBatching: true },
      {} as never,
    );
    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    const gridRuns = batched.decoderSession.calls.filter((call) => call.width > 1);
    expect(gridRuns.length).toBeGreaterThan(4);
    expect(gridRuns.length).toBeLessThanOrEqual(13);
    expect((batchResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(gridRuns.length);
    // The utilization gate blocks further batch entry; the sticky latch
    // stays true because no run failed.
    expect(batched.executor.rnntBatchAllowed).toBe(true);
    expect(batched.decoderSession.calls.filter((call) => call.width === 1).length)
      .toBeGreaterThan(20);
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it('defaults grid batching on for fp32 decoders', async () => {
    const script: RnntScript = (frame, target) =>
      frame === 6 && target === BLANK ? HELLO : BLANK;
    const fixture = createFixture({
      frameCount: 12,
      script,
      decoderQuantization: 'fp32',
    });
    const result = await fixture.executor.transcribe(createAudio(), {}, {} as never);
    expect((result.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBeGreaterThan(0);
    await fixture.executor.dispose();
  });

  it('keeps int8 decoders sequential by default', async () => {
    const script: RnntScript = (frame, target) =>
      frame === 6 && target === BLANK ? HELLO : BLANK;
    const fixture = createFixture({
      frameCount: 12,
      script,
      decoderQuantization: 'int8',
    });
    const result = await fixture.executor.transcribe(createAudio(), {}, {} as never);
    expect((result.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(0);
    expect(fixture.decoderSession.calls.every((call) => call.width === 1)).toBe(true);
    await fixture.executor.dispose();
  });

  it('pins the sequential path when gridBatching is disabled', async () => {
    const script: RnntScript = (frame, target) =>
      frame === 6 && target === BLANK ? HELLO : BLANK;
    const enabled = createFixture({ frameCount: 12, script });
    const disabled = createFixture({ frameCount: 12, script });
    const onResult = await enabled.executor.transcribe(
      createAudio(),
      { gridBatching: true },
      {} as never,
    );
    const offResult = await disabled.executor.transcribe(createAudio(), {}, {} as never);
    expect(tokenShape(offResult)).toEqual(tokenShape(onResult));
    expect((onResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBeGreaterThan(0);
    expect((offResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(0);
    expect(disabled.decoderSession.calls.every((call) => call.width === 1)).toBe(true);
    await enabled.executor.dispose();
    await disabled.executor.dispose();
  });

  it('latches off on a rejected grid run and still decodes sequentially', async () => {
    const script: RnntScript = (frame, target) =>
      frame === 6 && target === BLANK ? HELLO : BLANK;
    const fixture = createFixture({ frameCount: 12, script });
    fixture.decoderSession.rejectRuns = 1;
    const result = await fixture.executor.transcribe(
      createAudio(),
      { gridBatching: true },
      {} as never,
    );
    expect(result.utteranceText).toBe('hello');
    expect(fixture.executor.rnntBatchAllowed).toBe(false);
    expect(fixture.decoderSession.calls.every((call) => call.width === 1)).toBe(true);
    await fixture.executor.dispose();
  });
});

