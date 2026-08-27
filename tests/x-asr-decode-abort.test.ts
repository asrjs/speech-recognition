import { describe, expect, it } from 'vitest';

import { OrtXAsrExecutor, XAsrTokenizer, type XAsrModelConfig } from '../src/models/x-asr/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/lasr-ctc/ort.js';
import { PipelineAbortedError } from '../src/pipeline/composition.js';

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
    encoderFrameSize: 1,
    encoderFrameShift: 1,
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

function createExecutor() {
  const logitsAndStates: TrackingTensor[] = [];
  let joinerCalls = 0;
  const encoder: OrtSessionLike = {
    async run() {
      const encoded = new TrackingTensor('float32', new Float32Array([0.2]), [1, 1, 1]);
      const cached = new TrackingTensor('float32', new Float32Array([1]), [1]);
      logitsAndStates.push(encoded, cached);
      return { encoder_out: encoded, new_cached: cached };
    },
  };
  const decoder: OrtSessionLike = {
    async run() {
      const decoderOut = new TrackingTensor('float32', new Float32Array([0.1]), [1, 1]);
      logitsAndStates.push(decoderOut);
      return { decoder_out: decoderOut };
    },
  };
  const joiner: OrtSessionLike = {
    async run() {
      joinerCalls += 1;
      const logits = new Float32Array(4).fill(-5);
      logits[joinerCalls === 1 ? 1 : 0] = 5;
      const tensor = new TrackingTensor('float32', logits, [1, 4]);
      logitsAndStates.push(tensor);
      return { logit: tensor };
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
  const executor = new OrtXAsrExecutor('x-asr-abort', 'wasm', config, undefined);
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
      encoderFrameSize: 1,
      encoderFrameShift: 1,
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
  return { executor, logitsAndStates, getJoinerCalls: () => joinerCalls, resetJoiner: () => { joinerCalls = 0; } };
}

const audio = {
  sampleRate: 16000,
  numberOfChannels: 1,
  numberOfFrames: 16000,
  durationSeconds: 1,
  channels: [new Float32Array(16000)],
};

describe('X-ASR streaming decode abort', () => {
  it('stops the streaming step loop on abort without corrupting leftover encoder state', async () => {
    const { executor, logitsAndStates, getJoinerCalls, resetJoiner } = createExecutor();
    const original = new TrackingTensor('float32', new Float32Array([9]), [1]);
    const stream = executor.createStream();
    (stream.encoderStates as OrtTensorLike[]).push(original);
    const signal = { aborted: false };
    let abortAfterFirst = true;
    const originalJoiner = ((await (executor as unknown as { state: Promise<{ joiner: OrtSessionLike }> }).state).joiner);
    const bound = originalJoiner.run.bind(originalJoiner);
    originalJoiner.run = async (feeds) => {
      const result = await bound(feeds);
      if (abortAfterFirst && getJoinerCalls() === 1) signal.aborted = true;
      return result;
    };

    await expect(executor.pushStream(stream, audio.channels[0]!, true, { signal })).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(getJoinerCalls()).toBe(1);
    expect(original.disposed).toBe(0);
    expect(logitsAndStates.every((tensor) => tensor.disposed >= 1)).toBe(true);

    abortAfterFirst = false;
    signal.aborted = false;
    resetJoiner();
    const result = await executor.pushStream(stream, audio.channels[0]!, true);
    expect(result.transcript.isFinal).toBe(true);
    expect(result.transcript.utteranceText).toContain('hello');
    executor.disposeStream(result.state);
    await executor.dispose();
  });

  it('disposes the offline transcribe stream after abort and can transcribe again', async () => {
    const { executor, getJoinerCalls, resetJoiner } = createExecutor();
    const signal = { aborted: false };
    let abortAfterFirst = true;
    const joiner = (await (executor as unknown as { state: Promise<{ joiner: OrtSessionLike }> }).state).joiner;
    const bound = joiner.run.bind(joiner);
    joiner.run = async (feeds) => {
      const result = await bound(feeds);
      if (abortAfterFirst && getJoinerCalls() === 1) signal.aborted = true;
      return result;
    };

    await expect(executor.transcribe(audio, { signal })).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(getJoinerCalls()).toBe(1);

    abortAfterFirst = false;
    signal.aborted = false;
    resetJoiner();
    const result = await executor.transcribe(audio);
    expect(result.isFinal).toBe(true);
    await executor.dispose();
  });
});
