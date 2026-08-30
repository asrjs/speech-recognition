import { describe, expect, it } from 'vitest';

import { GigaAmRnntTokenizer, OrtGigaAmRnntExecutor } from '../src/models/gigaam-rnnt/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/lasr-ctc/ort.js';
import { PipelineAbortedError } from '../src/pipeline/composition.js';

const config = {
  ecosystem: 'gigaam' as const,
  architecture: 'gigaam-rnnt' as const,
  processorArchitecture: 'gigaam-fbank' as const,
  encoderArchitecture: 'gigaam-conformer' as const,
  decoderArchitecture: 'rnnt' as const,
  sampleRate: 16000,
  rawStride: 4,
  nMels: 64,
  featureHopSeconds: 0.01,
  vocabularySize: 4,
  languages: ['ru'],
  tokenizer: { kind: 'sentencepiece' as const, blankTokenId: 3 },
  nFft: 320 as const,
  winLength: 320 as const,
  hopLength: 160 as const,
  featureLayout: 'mel-major' as const,
  predictionHiddenSize: 320,
  predictionRnnLayers: 1,
  maxTokensPerFrame: 2,
};

const BLANK = 3;
const VOCAB = 4;
const HIDDEN = 768;
const DEC_LEN = 320;
const FRAMES = 6;

// Scripted joint decisions keyed by "frame:predictorLabel". Frame 1 carries
// the multi-emission case (token a then token b on the SAME frame) where a
// naive port that skips the emitting frame on re-batch silently diverges.
// Frame 4 would emit three tokens but must be clamped by maxTokensPerFrame.
const EMIT: Record<string, number> = {
  '1:3': 1, '1:1': 2,
  '2:2': 1,
  '3:1': 2,
  '4:2': 1,
  '4:1': 2,
};

function emitFor(frame: number, label: number): number {
  return EMIT[frame + ':' + label] ?? BLANK;
}

class TrackingTensor implements OrtTensorLike {
  disposed = 0;

  constructor(
    readonly type: string,
    readonly data: ArrayBufferView,
    readonly dims: readonly number[],
  ) {}

  dispose(): void {
    this.disposed += 1;
  }
}

class Tensor extends TrackingTensor {
  constructor(type: string, data: ArrayBufferView, dims: readonly number[]) {
    super(type, data, dims);
  }
}

type JoinerMode = 'ok' | 'reject' | 'badshape';

function createExecutor(mode: JoinerMode, forceSequential = false) {
  const tracked: TrackingTensor[] = [];
  let joinerCalls = 0;
  let batchJoinerCalls = 0;
  let decoderCalls = 0;

  const encoder: OrtSessionLike = {
    async run() {
      // hidden-major [1, HIDDEN, FRAMES] where every column equals the frame
      // id, matching the executor's strided gather layout.
      const data = new Float32Array(HIDDEN * FRAMES);
      for (let frame = 0; frame < FRAMES; frame += 1) {
        for (let index = 0; index < HIDDEN; index += 1) data[index * FRAMES + frame] = frame;
      }
      const encoded = new TrackingTensor('float32', data, [1, HIDDEN, FRAMES]);
      const length = new TrackingTensor('int32', new Int32Array([FRAMES]), [1]);
      tracked.push(encoded, length);
      return { encoded, encoded_len: length };
    },
  };

  const decoder: OrtSessionLike = {
    async run(feeds) {
      decoderCalls += 1;
      const label = Number(((feeds.x as TrackingTensor).data as BigInt64Array)[0]);
      const decValues = new Float32Array(DEC_LEN);
      decValues[0] = label;
      const dec = new TrackingTensor('float32', decValues, [1, 1, DEC_LEN]);
      const ho = new TrackingTensor('float32', new Float32Array(320), [1, 1, 320]);
      const co = new TrackingTensor('float32', new Float32Array(320), [1, 1, 320]);
      tracked.push(dec, ho, co);
      return { dec, ho, co };
    },
  };

  const joint: OrtSessionLike = {
    async run(feeds) {
      joinerCalls += 1;
      const enc = feeds.enc as TrackingTensor;
      const dec = feeds.dec as TrackingTensor;
      const rows = enc.dims[0] ?? 1;
      if (rows > 1) batchJoinerCalls += 1;
      if (mode === 'reject' && rows > 1) throw new Error('joiner graph rejects batched shapes');
      const columns = mode === 'badshape' && rows > 1 ? 3 : VOCAB;
      const data = new Float32Array(rows * columns).fill(-5);
      if (columns === VOCAB) {
        const encData = enc.data as Float32Array;
        const decData = dec.data as Float32Array;
        for (let row = 0; row < rows; row += 1) {
          const frameId = Math.round(encData[row * HIDDEN] ?? -1);
          const label = Math.round(decData[row * DEC_LEN] ?? -1);
          data[row * VOCAB + emitFor(frameId, label)] = 5;
        }
      }
      const logits = new TrackingTensor('float32', data, [rows, 1, 1, columns]);
      tracked.push(logits);
      return { joint: logits };
    },
  };

  const ort: OrtModuleLike = {
    env: { wasm: {} },
    Tensor,
    InferenceSession: {
      async create(): Promise<OrtSessionLike> {
        throw new Error('The test injects sessions.');
      },
    },
  };

  const executor = new OrtGigaAmRnntExecutor('gigaam-rnnt-batching', 'wasm', config, undefined);
  (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
    ort,
    encoder,
    decoder,
    joint,
    tokenizer: GigaAmRnntTokenizer.fromText(' 0\na 1\nb 2\n<blk> 3\n'),
    warnings: [],
  });
  if (forceSequential) {
    (executor as unknown as { joinerBatchAllowed: boolean }).joinerBatchAllowed = false;
  }
  return {
    executor,
    tracked,
    getJoinerCalls: () => joinerCalls,
    getBatchJoinerCalls: () => batchJoinerCalls,
    getDecoderCalls: () => decoderCalls,
    batchAllowedFlag: () => (executor as unknown as { joinerBatchAllowed: boolean }).joinerBatchAllowed,
  };
}

const audio = {
  sampleRate: 16000,
  numberOfChannels: 1,
  numberOfFrames: 16000,
  durationSeconds: 1,
  channels: [new Float32Array(16000)],
};

describe('GigaAM RNN-T speculative batched joiner decode', () => {
  it('emits the identical token sequence through batched joint runs', async () => {
    const fixture = createExecutor('ok');
    const result = await fixture.executor.transcribe(audio);
    expect(result.utteranceText).toBe('ababab');
    expect(result.tokens.map((token) => token.startTime)).toEqual([0.04, 0.04, 0.08, 0.12, 0.16, 0.16]);
    expect(fixture.getBatchJoinerCalls()).toBeGreaterThan(0);
    expect(result.metrics?.joinerBatchRuns).toBeGreaterThan(0);
    expect(fixture.tracked.every((tensor) => tensor.disposed >= 1)).toBe(true);
    await fixture.executor.dispose();
  });

  it('matches the forced-sequential transcript, iteration count, and dispatch budget', async () => {
    const batched = createExecutor('ok');
    const sequential = createExecutor('ok', true);
    const batchedResult = await batched.executor.transcribe(audio);
    const sequentialResult = await sequential.executor.transcribe(audio);
    expect(batchedResult.utteranceText).toBe(sequentialResult.utteranceText);
    expect(batchedResult.tokens.map((token) => [token.id, token.startTime, token.endTime]))
      .toEqual(sequentialResult.tokens.map((token) => [token.id, token.startTime, token.endTime]));
    expect(batchedResult.metrics?.decodeIterations).toBe(sequentialResult.metrics?.decodeIterations);
    expect(sequential.getBatchJoinerCalls()).toBe(0);
    expect(batched.getJoinerCalls()).toBeLessThan(sequential.getJoinerCalls());
    // Predictor-state caching: blank re-scoring must not re-run the LSTM.
    // Both paths share the cache, so decoder runs are equal here; the
    // batched path's strict win is the joint dispatch count asserted above.
    expect(batched.getDecoderCalls()).toBeLessThanOrEqual(sequential.getDecoderCalls());
    await batched.executor.dispose();
    await sequential.executor.dispose();
  });

  it('clamps multi-token frames at maxTokensPerFrame on both paths', async () => {
    const batched = createExecutor('ok');
    const sequential = createExecutor('ok', true);
    const batchedResult = await batched.executor.transcribe(audio);
    const sequentialResult = await sequential.executor.transcribe(audio);
    // Frames 1 and 4 carry two scripted emissions; the clamp caps each at two.
    const frame1 = batchedResult.tokens.filter((token) => token.startTime === 0.04);
    const frame4 = batchedResult.tokens.filter((token) => token.startTime === 0.16);
    expect(frame1.length).toBe(2);
    expect(frame4.length).toBe(2);
    expect(batchedResult.utteranceText).toBe(sequentialResult.utteranceText);
    await batched.executor.dispose();
    await sequential.executor.dispose();
  });

  it('falls back permanently when the joint graph rejects batched shapes', async () => {
    const fixture = createExecutor('reject');
    const result = await fixture.executor.transcribe(audio);
    expect(result.utteranceText).toBe('ababab');
    expect(fixture.batchAllowedFlag()).toBe(false);
    expect(result.metrics?.joinerBatchRuns).toBe(0);
    const callsAfterFirstRun = fixture.getJoinerCalls();
    const batchCalls = fixture.getBatchJoinerCalls();
    await fixture.executor.transcribe(audio);
    expect(fixture.getBatchJoinerCalls()).toBe(batchCalls);
    expect(fixture.getJoinerCalls()).toBeGreaterThan(callsAfterFirstRun);
    expect(fixture.tracked.every((tensor) => tensor.disposed >= 1)).toBe(true);
    await fixture.executor.dispose();
  });

  it('falls back permanently when batched logits are not row-parallel', async () => {
    const fixture = createExecutor('badshape');
    const result = await fixture.executor.transcribe(audio);
    expect(result.utteranceText).toBe('ababab');
    expect(fixture.batchAllowedFlag()).toBe(false);
    expect(result.metrics?.joinerBatchRuns).toBe(0);
    await fixture.executor.dispose();
  });

  it('propagates aborts raised inside a batched joint run without latching off', async () => {
    const fixture = createExecutor('ok');
    const signal = { aborted: false };
    const loaded = await (fixture.executor as unknown as { state: Promise<{ joint: OrtSessionLike }> }).state;
    const originalJoint = loaded.joint;
    (loaded as unknown as { joint: OrtSessionLike }).joint = {
      async run(feeds) {
        const rows = (feeds.enc as TrackingTensor).dims[0] ?? 1;
        if (rows > 1) {
          signal.aborted = true;
          throw new PipelineAbortedError('decode');
        }
        return originalJoint.run(feeds);
      },
    };
    await expect(fixture.executor.transcribe(audio, { signal })).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(fixture.batchAllowedFlag()).toBe(true);
    expect(fixture.tracked.every((tensor) => tensor.disposed >= 1)).toBe(true);
    await fixture.executor.dispose();
  });
});
