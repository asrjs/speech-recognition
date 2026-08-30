import { describe, expect, it } from 'vitest';

import { OrtXAsrExecutor, XAsrTokenizer, type XAsrModelConfig } from '../src/models/x-asr/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/lasr-ctc/ort.js';

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

// Each mock encoder run emits FRAMES_PER_CHUNK rows whose single hidden
// value is the global frame id. The scripted joiner turns a frame id into a
// logit row, so batched and sequential decoding must agree by construction.
const FRAME_TOKENS: Record<number, number> = { 1: 1, 8: 2 };
const VOCAB = 4;
const FRAMES_PER_CHUNK = 5;

const config: XAsrModelConfig = {
  ecosystem: 'x-asr',
  architecture: 'zipformer2-streaming-rnnt',
  processorArchitecture: 'kaldi-fbank',
  encoderArchitecture: 'zipformer2',
  decoderArchitecture: 'stateless-rnnt',
  sampleRate: 16000,
  featureDim: 80,
  featureHopSeconds: 0.01,
  rawStride: 1,
  languages: ['zh', 'en'],
  chunkMs: 160,
  graph: {
    encoderStateInputs: [{ name: 'cached', type: 'float32', dims: [1] }],
    encoderFrameSize: FRAMES_PER_CHUNK,
    encoderFrameShift: FRAMES_PER_CHUNK,
    decoderContextSize: 2,
    featureInputName: 'x',
    encoderOutputName: 'encoder_out',
    decoderInputName: 'y',
    decoderOutputName: 'decoder_out',
    joinerEncoderInputName: 'encoder_out',
    joinerDecoderInputName: 'decoder_out',
    joinerOutputName: 'logit',
  },
};

type JoinerMode = 'ok' | 'reject' | 'badshape';

function createExecutor(mode: JoinerMode, forceSequential = false) {
  const tracked: TrackingTensor[] = [];
  let joinerCalls = 0;
  let batchJoinerCalls = 0;
  let encoderRuns = 0;

  const encoder: OrtSessionLike = {
    async run() {
      const base = encoderRuns * FRAMES_PER_CHUNK;
      encoderRuns += 1;
      const values = new Float32Array(FRAMES_PER_CHUNK);
      for (let index = 0; index < FRAMES_PER_CHUNK; index += 1) values[index] = base + index;
      const encoded = new TrackingTensor('float32', values, [1, FRAMES_PER_CHUNK, 1]);
      const cached = new TrackingTensor('float32', new Float32Array([encoderRuns]), [1]);
      tracked.push(encoded, cached);
      return { encoder_out: encoded, new_cached: cached };
    },
  };
  const decoder: OrtSessionLike = {
    async run() {
      const decoderOut = new TrackingTensor('float32', new Float32Array([0.5]), [1, 1]);
      tracked.push(decoderOut);
      return { decoder_out: decoderOut };
    },
  };
  const joiner: OrtSessionLike = {
    async run(feeds) {
      joinerCalls += 1;
      const enc = feeds.encoder_out as TrackingTensor;
      const rows = enc.dims[0] ?? 1;
      if (rows > 1) batchJoinerCalls += 1;
      if (mode === 'reject' && rows > 1) throw new Error('joiner graph rejects batched shapes');
      const columns = mode === 'badshape' && rows > 1 ? 3 : VOCAB;
      const data = new Float32Array(rows * columns).fill(-5);
      if (columns === VOCAB) {
        const encData = enc.data as Float32Array;
        for (let row = 0; row < rows; row += 1) {
          const frameId = Math.round(encData[row] ?? -1);
          data[row * VOCAB + (FRAME_TOKENS[frameId] ?? 0)] = 5;
        }
      }
      const logits = new TrackingTensor('float32', data, [rows, columns]);
      tracked.push(logits);
      return { logit: logits };
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
  const executor = new OrtXAsrExecutor('x-asr-batching', 'wasm', config, undefined);
  (executor as unknown as { source: unknown }).source = {
    kind: 'direct',
    artifacts: { encoderUrl: 'encoder', decoderUrl: 'decoder', joinerUrl: 'joiner', tokenizerUrl: 'tokens' },
  };
  (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
    ort,
    encoder,
    decoder,
    joiner,
    tokenizer: XAsrTokenizer.fromText('<blk> 0\n▁hello 1\n▁world 2\n'),
    graph: {
      featureInputName: 'x',
      encoderOutputName: 'encoder_out',
      encoderFrameSize: FRAMES_PER_CHUNK,
      encoderFrameShift: FRAMES_PER_CHUNK,
      encoderStateInputs: [{ name: 'cached', type: 'float32', dims: [1] }],
      decoderInputName: 'y',
      decoderOutputName: 'decoder_out',
      decoderContextSize: 2,
      decoderIndexType: 'int64',
      joinerEncoderInputName: 'encoder_out',
      joinerDecoderInputName: 'decoder_out',
      joinerOutputName: 'logit',
    },
  });
  if (forceSequential) {
    (executor as unknown as { joinerBatchAllowed: boolean }).joinerBatchAllowed = false;
  }
  return {
    executor,
    tracked,
    getJoinerCalls: () => joinerCalls,
    getBatchJoinerCalls: () => batchJoinerCalls,
    batchAllowedFlag: () => (executor as unknown as { joinerBatchAllowed: boolean }).joinerBatchAllowed,
  };
}

const audioSeconds = 4;
const audio = {
  sampleRate: 16000,
  numberOfChannels: 1,
  numberOfFrames: 16000 * audioSeconds,
  durationSeconds: audioSeconds,
  channels: [new Float32Array(16000 * audioSeconds).fill(0.1)],
};

async function transcribeOnce(executor: OrtXAsrExecutor): Promise<string> {
  const result = await executor.transcribe(audio);
  return result.utteranceText ?? '';
}

describe('X-ASR speculative batched joiner decode', () => {
  it('emits the identical token sequence through batched joiner runs', async () => {
    const { executor, tracked, getJoinerCalls, getBatchJoinerCalls } = createExecutor('ok');
    const text = await transcribeOnce(executor);
    expect(text).toBe('hello world');
    expect(getBatchJoinerCalls()).toBeGreaterThan(0);
    // 80 encoder runs x 5 frames = 400 sequential joiner runs; the
    // speculative path converges to roughly one batch run per chunk plus
    // one re-batch after each emitted token.
    expect(getJoinerCalls()).toBeLessThan(120);
    expect(tracked.every((tensor) => tensor.disposed >= 1)).toBe(true);
    await executor.dispose();
  });

  it('matches the forced-sequential transcript and joiner-idle frame budget', async () => {
    const batched = createExecutor('ok');
    const sequential = createExecutor('ok', true);
    const batchedText = await transcribeOnce(batched.executor);
    const sequentialText = await transcribeOnce(sequential.executor);
    expect(batchedText).toBe(sequentialText);
    expect(batchedText).toBe('hello world');
    expect(sequential.getBatchJoinerCalls()).toBe(0);
    expect(batched.getJoinerCalls()).toBeLessThan(sequential.getJoinerCalls());
    await batched.executor.dispose();
    await sequential.executor.dispose();
  });

  it('falls back permanently when the joiner graph rejects batch shapes', async () => {
    const { executor, getBatchJoinerCalls, batchAllowedFlag } = createExecutor('reject');
    const text = await transcribeOnce(executor);
    expect(text).toBe('hello world');
    expect(batchAllowedFlag()).toBe(false);
    expect(getBatchJoinerCalls()).toBe(1);
    await executor.dispose();
  });

  it('falls back when the batch output is not row-parallel', async () => {
    const { executor, batchAllowedFlag } = createExecutor('badshape');
    const text = await transcribeOnce(executor);
    expect(text).toBe('hello world');
    expect(batchAllowedFlag()).toBe(false);
    await executor.dispose();
  });
});
