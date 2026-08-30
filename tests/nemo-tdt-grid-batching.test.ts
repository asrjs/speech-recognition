import { describe, expect, it } from 'vitest';

import { PipelineAbortedError } from '../src/pipeline/composition.js';

import { parseNemoTdtConfig } from '../src/models/nemo-tdt/config.js';
import { DEFAULT_NEMO_TDT_CLASSIFICATION } from '../src/models/nemo-tdt/config.js';
import { OrtNemoTdtExecutor } from '../src/models/nemo-tdt/executor.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/nemo-tdt/ort.js';
import type { NemoTdtModelConfig } from '../src/models/nemo-tdt/types.js';
import type { AudioBufferLike } from '../src/types/index.js';

// Grid-capable mock of the fused Parakeet TDT decoder-joint graph.
// The shipped graph scores every slot of a [1, features, frames] encoder
// batch independently against the fed (target, recurrent state) pair,
// verified byte-exact against single-frame runs for both the fp32 and
// int8 v3 exports. The mock therefore derives each row from the same
// pure function of (frame, target) that a single-frame run observes.

const BLANK = 0;
const HELLO = 1;
const WORLD = 2;
const VOCAB = ['<blk>', '_hello', '_world'];
const ROW_LENGTH = VOCAB.length + 4; // + duration bins

interface ScriptStep {
  readonly token: number;
  readonly step: number;
}

type DecoderScript = (frame: number, target: number) => ScriptStep;

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
  readonly emitted: MockTensor[] = [];
  rejectRuns = 0;
  corruptGrid = false;

  constructor(private readonly script: DecoderScript) {}

  async run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>> {
    if (this.rejectRuns > 0) {
      this.rejectRuns -= 1;
      throw new Error('simulated decoder rejection');
    }
    const encoder = feeds.encoder_outputs as OrtTensorLike<Float32Array>;
    const dims = [...encoder.dims];
    const width = dims[dims.length - 1]!;
    const target = Number((feeds.targets as OrtTensorLike<Int32Array>).data[0] ?? -1);
    // The harness stores the frame index in feature 0 of every frame, so
    // the first value of each gathered row identifies its frame.
    const startFrame = Math.round(encoder.data[0] ?? 0);
    // The executor must feed a feature-major [1, features, width] tensor:
    // element (0, w) at flat offset w identifies frame w. This check
    // catches grid transpositions a layout-blind mock would miss.
    for (let column = 0; column < width; column += 1) {
      const frame = Math.round(encoder.data[column] ?? -1);
      if (frame !== startFrame + column) {
        throw new Error(
          'Grid layout violation at column ' + column + ' of run starting ' + startFrame,
        );
      }
    }
    this.calls.push({ width, startFrame, target });
    const values = new Float32Array(width * ROW_LENGTH + (this.corruptGrid ? 1 : 0));
    for (let row = 0; row < width; row += 1) {
      const step = this.script(startFrame + row, target);
      values[row * ROW_LENGTH + step.token] = 5;
      values[row * ROW_LENGTH + VOCAB.length + step.step] = 5;
    }
    const outputs = new MockTensor(values, [1, width, ROW_LENGTH]);
    const state1 = new MockTensor(new Float32Array(4), [1, 1, 4]);
    const state2 = new MockTensor(new Float32Array(4), [1, 1, 4]);
    this.emitted.push(outputs, state1, state2);
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
  readonly script: DecoderScript;
  readonly config?: Partial<NemoTdtModelConfig>;
  readonly batching: boolean;
}) {
  const config = parseNemoTdtConfig('test-grid', {
    subsamplingFactor: 4,
    frameShiftSeconds: 0.01,
    melBins: 2,
    vocabularySize: VOCAB.length,
    predictionLayers: 1,
    predictionHiddenSize: 4,
    ...options.config,
  });
  // Keep the feature dimension larger than any test frame count so the
  // executor's [1,D,T]-vs-[1,T,D] heuristic unambiguously reads the
  // frame-major layout the mock emits.
  const featureSize = Math.max(16, options.frameCount + 4);
  const encoderData = new Float32Array(featureSize * options.frameCount);
  for (let frame = 0; frame < options.frameCount; frame += 1) {
    encoderData[frame * featureSize] = frame; // frame-major layout
  }
  const encoderSession = new (class implements OrtSessionLike {
    async run(): Promise<Record<string, OrtTensorLike>> {
      return { outputs: new MockTensor(encoderData, [1, options.frameCount, featureSize]) };
    }
  })();
  const decoderSession = new GridDecoderSession(options.script);
  const tokenizer = new ParakeetTokenizer(VOCAB);
  const executor = new OrtNemoTdtExecutor(
    'test-grid',
    DEFAULT_NEMO_TDT_CLASSIFICATION,
    config,
    'wasm',
  ) as unknown as {
    transcribe(
      audio: AudioBufferLike,
      options?: Record<string, unknown>,
    ): Promise<Record<string, unknown>>;
    dispose(): Promise<void>;
    tdtBatchAllowed: boolean;
    loadStatePromise?: Promise<unknown>;
  };
  executor.tdtBatchAllowed = options.batching;
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
  });
  return { executor, decoderSession };
}
const TIMING_KEYS = [
  "preprocessMs", "encodeMs", "decodeMs", "tokenizeMs",
  "totalMs", "wallMs", "rtf", "rtfx",
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

describe('nemo-tdt speculative grid batching', () => {
  it('matches the forced-sequential transcript while collapsing blank runs', async () => {
    // Frames 0-5 are blank, frame 6 emits hello (duration 1), the rest
    // are blank. The sequential loop needs one dispatch per frame; the
    // batched loop covers the leading blank run in a single grid.
    const script: DecoderScript = (frame, target) => {
      if (frame === 6 && target === BLANK) {
        return { token: HELLO, step: 1 };
      }
      return { token: BLANK, step: 0 };
    };
    const sequential = createFixture({ frameCount: 12, script, batching: false });
    const batched = createFixture({ frameCount: 12, script, batching: true });
    const seqResult = await sequential.executor.transcribe(createAudio(), {}, {} as never);
    const batchResult = await batched.executor.transcribe(createAudio(), {}, {} as never);

    expect(batchResult.utteranceText).toBe(seqResult.utteranceText);
    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    expect(slimMetrics(batchResult.metrics)).toEqual(slimMetrics(seqResult.metrics));
    expect(sequential.decoderSession.calls.length).toBe(12);
    expect(batched.decoderSession.calls.length).toBeLessThan(6);
    expect((batchResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBeGreaterThan(0);
    expect((seqResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(0);
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it('reuses duration-skipped rows and re-batches after each emission', async () => {
    // Frame 0 blank with duration 3, frame 3 emits hello with duration
    // 2, frame 5 emits world, frames 6+ blank. The scan must jump 0->3
    // on the same grid run and then restart from the emitting frame.
    const script: DecoderScript = (frame, target) => {
      if (frame === 0 && target === BLANK) return { token: BLANK, step: 3 };
      if (frame === 3 && target === BLANK) return { token: HELLO, step: 2 };
      if (frame === 5 && target === HELLO) return { token: WORLD, step: 0 };
      return { token: BLANK, step: 0 };
    };
    const sequential = createFixture({ frameCount: 10, script, batching: false });
    const batched = createFixture({ frameCount: 10, script, batching: true });
    const seqResult = await sequential.executor.transcribe(createAudio(), {}, {} as never);
    const batchResult = await batched.executor.transcribe(createAudio(), {}, {} as never);

    expect(batchResult.utteranceText).toBe(seqResult.utteranceText);
    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    expect(slimMetrics(batchResult.metrics)).toEqual(slimMetrics(seqResult.metrics));
    // Sequential needs one dispatch per visited step (~9 for this script);
    // the grid path covers the same frames in 6 runs (each emission frame is
    // re-scored by the run that starts from it).
    expect(sequential.decoderSession.calls.length).toBeGreaterThan(6);
    expect(batched.decoderSession.calls.length)
      .toBeLessThan(sequential.decoderSession.calls.length);
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it('supports multi-token frames by re-batching from the emitting frame', async () => {
    // Frames 0-1 blank, frame 2 emits hello then (target=hello) world.
    const script: DecoderScript = (frame, target) => {
      if (frame === 2 && target === BLANK) return { token: HELLO, step: 0 };
      if (frame === 2 && target === HELLO) return { token: WORLD, step: 0 };
      return { token: BLANK, step: 0 };
    };
    const sequential = createFixture({ frameCount: 8, script, batching: false });
    const batched = createFixture({ frameCount: 8, script, batching: true });
    const seqResult = await sequential.executor.transcribe(createAudio(), {}, {} as never);
    const batchResult = await batched.executor.transcribe(createAudio(), {}, {} as never);

    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    expect(batchResult.utteranceText).toBe('_hello_world');
    expect(slimMetrics(batchResult.metrics)).toEqual(slimMetrics(seqResult.metrics));
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });
  it('honours maxSymbolsPerStep inside the batched scan', async () => {
    // Frame 3 emits repeatedly under its own token; the cap of 2 must
    // force frame advancement exactly like the sequential loop.
    const script: DecoderScript = (frame, target) => {
      if (frame === 3) {
        return { token: target === BLANK ? HELLO : WORLD, step: 0 };
      }
      return { token: BLANK, step: 0 };
    };
    const sequential = createFixture({
      frameCount: 8, script, batching: false, config: { maxSymbolsPerStep: 2 },
    });
    const batched = createFixture({
      frameCount: 8, script, batching: true, config: { maxSymbolsPerStep: 2 },
    });
    const seqResult = await sequential.executor.transcribe(createAudio(), {}, {} as never);
    const batchResult = await batched.executor.transcribe(createAudio(), {}, {} as never);

    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    expect(slimMetrics(batchResult.metrics)).toEqual(slimMetrics(seqResult.metrics));
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it('latches off permanently on a rejected or malformed grid run', async () => {
    const script: DecoderScript = () => ({ token: BLANK, step: 0 });
    const rejecting = createFixture({ frameCount: 8, script, batching: true });
    rejecting.decoderSession.rejectRuns = 1;
    const result = await rejecting.executor.transcribe(createAudio(), {}, {} as never);
    expect(result.utteranceText).toBe('');
    expect(rejecting.executor.tdtBatchAllowed).toBe(false);
    await rejecting.executor.dispose();
    const corrupt = createFixture({ frameCount: 8, script, batching: true });
    corrupt.decoderSession.corruptGrid = true;
    await corrupt.executor.transcribe(createAudio(), {}, {} as never);
    expect(corrupt.executor.tdtBatchAllowed).toBe(false);
    // After latching, only single-frame runs remain.
    const gridWidths = corrupt.decoderSession.calls.filter((c) => c.width > 1);
    expect(gridWidths.length).toBe(1);
    await corrupt.executor.dispose();
  });

  it('re-throws aborts without latching batching off', async () => {
    const script: DecoderScript = () => ({ token: BLANK, step: 0 });
    const fixture = createFixture({ frameCount: 8, script, batching: true });
    const original = fixture.decoderSession.run.bind(fixture.decoderSession);
    fixture.decoderSession.run = async (feeds) => {
      if (fixture.decoderSession.calls.length >= 1) {
        throw new PipelineAbortedError('decode');
      }
      return original(feeds);
    };
    await expect(
      fixture.executor.transcribe(createAudio(), {}, {} as never),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(fixture.executor.tdtBatchAllowed).toBe(true);
    await fixture.executor.dispose();
  });

  it("latches batching off after the warmup window when rows are emission-dense", async () => {
    // Every frame emits, so each grid scan stops at its first row and the
    // remaining speculative columns are wasted. Once 24 columns have been
    // sampled below 70% utilization, further grid runs must stop and the
    // rest of the file is decoded sequentially. The transcript is
    // unchanged because the grid emission row equals the sequential step.
    const script: DecoderScript = () => ({ token: HELLO, step: 0 });
    const sequential = createFixture({ frameCount: 24, script, batching: false });
    const batched = createFixture({ frameCount: 24, script, batching: true });
    const seqResult = await sequential.executor.transcribe(createAudio(), {}, {} as never);
    const batchResult = await batched.executor.transcribe(createAudio(), {}, {} as never);
    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    const gridRuns = batched.decoderSession.calls.filter((c) => c.width > 1);
    // One narrow grid run per frame until the gate closes; every call
    // after the sampling window must be single-frame.
    expect(gridRuns.length).toBeGreaterThan(4);
    expect(gridRuns.length).toBeLessThanOrEqual(13);
    expect((batchResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(gridRuns.length);
    const widths = batched.decoderSession.calls.map((c) => c.width);
    // The last frames are covered without any wide speculative grid once
    // the sampling window (24 columns) has been consumed.
    expect(widths.slice(-6).every((w) => w <= 2)).toBe(true);
    expect(widths.filter((w) => w > 2).length).toBeLessThanOrEqual(3);
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it("re-opens the gate after a latched dense phase once the audio goes blank", async () => {
    // Frames 0-11 emit on every visit (the gate latches during the
    // sampling window), frames 12+ are blank. After six sequential blank
    // visits the executor must re-open the sampling window and cross the
    // remaining silence with widening grid runs instead of one dispatch
    // per frame.
    const script: DecoderScript = (frame, target) => {
      if (frame < 12 && target === BLANK) {
        return { token: HELLO, step: 1 };
      }
      return { token: BLANK, step: 0 };
    };
    const sequential = createFixture({ frameCount: 64, script, batching: false });
    const batched = createFixture({ frameCount: 64, script, batching: true });
    const seqResult = await sequential.executor.transcribe(createAudio(), {}, {} as never);
    const batchResult = await batched.executor.transcribe(createAudio(), {}, {} as never);

    expect(tokenShape(batchResult)).toEqual(tokenShape(seqResult));
    const calls = batched.decoderSession.calls;
    const wideRuns = calls.filter((call) => call.width >= 8);
    expect(wideRuns.length).toBeGreaterThan(0);
    // The latched dense phase decodes sequentially (12 dense frames plus
    // the six blank visits before the re-probe), but the 40+ blank tail
    // must be crossed by grids: far fewer dispatches than the 64 the
    // sequential loop needs.
    expect(calls.length).toBeLessThan(40);
    const maxGridWidth = Math.max(...calls.map((call) => call.width));
    expect(maxGridWidth).toBeGreaterThanOrEqual(8);
    await sequential.executor.dispose();
    await batched.executor.dispose();
  });

  it("keeps grid runs active for blank-dominant audio past the sampling window", async () => {
    // A long leading blank run followed by two emissions near the end of
    // the widest speculative grid: the scans examine almost every column
    // (>=70% utilization), so the gate must stay open and the 64-frame
    // file must be covered in few dispatches.
    const script: DecoderScript = (frame, target) => {
      if (frame === 53 && target === BLANK) {
        return { token: HELLO, step: 0 };
      }
      if (frame === 54 && target === HELLO) {
        return { token: WORLD, step: 0 };
      }
      return { token: BLANK, step: 0 };
    };
    const batched = createFixture({ frameCount: 64, script, batching: true });
    const result = await batched.executor.transcribe(createAudio(), {}, {} as never);
    expect(result.utteranceText).toBe('_hello_world');
    const gridRuns = batched.decoderSession.calls.filter((c) => c.width > 1);
    expect(gridRuns.length).toBeGreaterThan(2);
    expect(batched.decoderSession.calls.length).toBeLessThan(16);
    await batched.executor.dispose();
  });

  it("pins the sequential path when gridBatching is disabled", async () => {
    const script: DecoderScript = (frame, target) => {
      if (frame === 6 && target === BLANK) {
        return { token: HELLO, step: 1 };
      }
      return { token: BLANK, step: 0 };
    };
    const enabled = createFixture({ frameCount: 12, script, batching: true });
    const disabled = createFixture({ frameCount: 12, script, batching: true });
    const onResult = await enabled.executor.transcribe(createAudio(), {}, {} as never);
    const offResult = await disabled.executor.transcribe(
      createAudio(),
      { gridBatching: false },
      {} as never,
    );
    expect(tokenShape(offResult)).toEqual(tokenShape(onResult));
    expect((onResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBeGreaterThan(0);
    expect((offResult.metrics as Record<string, unknown>).decoderGridBatchRuns)
      .toBe(0);
    expect(disabled.decoderSession.calls.every((c) => c.width === 1)).toBe(true);
    await enabled.executor.dispose();
    await disabled.executor.dispose();
  });

  it('disposes every decoder tensor it produced', async () => {
    const script: DecoderScript = (frame, target) => {
      if (frame === 4 && target === BLANK) return { token: HELLO, step: 0 };
      return { token: BLANK, step: 0 };
    };
    const fixture = createFixture({ frameCount: 10, script, batching: true });
    await fixture.executor.transcribe(createAudio(), {}, {} as never);
    expect(fixture.decoderSession.emitted.length).toBeGreaterThan(3);
    expect(fixture.decoderSession.emitted.every((tensor) => tensor.disposed)).toBe(true);
    await fixture.executor.dispose();
  });
});