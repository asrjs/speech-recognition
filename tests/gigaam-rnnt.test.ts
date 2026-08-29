import { describe, expect, it } from 'vitest';

import { GigaAmRnntTokenizer, OrtGigaAmRnntExecutor, resolveGigaAmRnntBackends } from '../src/models/gigaam-rnnt/index.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from '../src/models/lasr-ctc/ort.js';
import { PipelineAbortedError } from '../src/pipeline/composition.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';

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
  vocabularySize: 35,
  languages: ['ru'],
  tokenizer: { kind: 'sentencepiece' as const, blankTokenId: 34 },
  nFft: 320 as const,
  winLength: 320 as const,
  hopLength: 160 as const,
  featureLayout: 'mel-major' as const,
  predictionHiddenSize: 320,
  predictionRnnLayers: 1,
  maxTokensPerFrame: 3,
};

describe('GigaAM v3 RNN-T contract', () => {
  it('resolves explicit hybrid component providers while preserving backend defaults', () => {
    expect(resolveGigaAmRnntBackends(undefined, 'wasm')).toEqual({
      ortBackend: 'wasm', encoderBackend: 'wasm', decoderBackend: 'wasm', jointBackend: 'wasm',
    });
    expect(resolveGigaAmRnntBackends({ encoderBackend: 'webgpu', decoderBackend: 'wasm', jointBackend: 'wasm' }, 'wasm')).toEqual({
      ortBackend: 'webgpu', encoderBackend: 'webgpu', decoderBackend: 'wasm', jointBackend: 'wasm',
    });
    expect(resolveGigaAmRnntBackends(undefined, 'webgpu')).toEqual({
      ortBackend: 'webgpu', encoderBackend: 'webgpu', decoderBackend: 'webgpu', jointBackend: 'webgpu',
    });
  });

  it('adds the implicit final blank to the published piece vocabulary', () => {
    const tokenizer = GigaAmRnntTokenizer.fromText('  0\na 2\n<blk> 34\n');
    expect(tokenizer.blankId).toBe(34);
    expect(tokenizer.decode([2, 34])).toBe('a');
  });

  it('remains artifact-gated and discoverable', async () => {
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    expect(runtime.listModelFamilies().find((family) => family.family === 'gigaam-rnnt')?.supports('gigaam-v3-e2e-rnnt')).toBe(true);
    const model = await runtime.loadModel({ family: 'gigaam-rnnt', modelId: 'gigaam-v3-e2e-rnnt', backend: 'wasm' });
    await expect(model.createSession()).rejects.toThrow(/No GigaAM RNN-T artifact source/);
  });

  it('runs encoder, prediction network, and joint with the official feed boundary', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
    }
    const feeds: Record<string, Record<string, unknown>> = {};
    let jointCalls = 0;
    const encoder: OrtSessionLike = {
      async run(input) {
        feeds.encoder = input;
        return {
          encoded: new Tensor('float32', new Float32Array(768), [1, 768, 1]),
          encoded_len: new Tensor('int32', new Int32Array([1]), [1]),
        };
      },
    };
    const decoder: OrtSessionLike = {
      async run(input) {
        feeds.decoder = input;
        return {
          dec: new Tensor('float32', new Float32Array(320), [1, 1, 320]),
          ho: new Tensor('float32', new Float32Array(320), [1, 1, 320]),
          co: new Tensor('float32', new Float32Array(320), [1, 1, 320]),
        };
      },
    };
    const joint: OrtSessionLike = {
      async run(input) {
        jointCalls += 1;
        feeds.joint = input;
        const logits = new Float32Array(35).fill(-5);
        logits[jointCalls === 1 ? 2 : 34] = 5;
        return { joint: new Tensor('float32', logits, [1, 1, 35]) };
      },
    };
    const ort: OrtModuleLike = {
      env: { wasm: {} },
      Tensor,
      InferenceSession: {
        create: async (url) => (url.includes('encoder') ? encoder : url.includes('decoder') ? decoder : joint),
      },
    };
    const executor = new OrtGigaAmRnntExecutor('gigaam-rnnt-test', 'wasm', config, undefined);
    (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
      ort,
      encoder,
      decoder,
      joint,
      tokenizer: GigaAmRnntTokenizer.fromText('  0\na 2\n<blk> 34\n'),
      warnings: [],
    });

    const result = await executor.transcribe({
      sampleRate: 16000,
      numberOfChannels: 1,
      numberOfFrames: 16000,
      durationSeconds: 1,
      channels: [new Float32Array(16000)],
    });
    expect(result.utteranceText).toBe('a');
    expect(Object.keys(feeds.encoder ?? {})).toEqual(['audio_signal', 'length']);
    expect(Object.keys(feeds.decoder ?? {})).toEqual(['x', 'hi', 'ci']);
    expect(Object.keys(feeds.joint ?? {})).toEqual(['enc', 'dec']);
    expect(jointCalls).toBe(2);
    expect(result.metrics?.preprocessMs).toBeGreaterThanOrEqual(0);
    expect(result.metrics?.encodeMs).toBeGreaterThanOrEqual(0);
    expect(result.metrics?.decodeMs).toBeGreaterThanOrEqual(0);
    expect(result.metrics).toMatchObject({ encoderBackend: 'wasm', decoderBackend: 'wasm', jointBackend: 'wasm' });
  });

  it('stops the joint/decoder loop on abort, disposes tensors, and can decode again', async () => {
    class Tensor<TData extends ArrayBufferView = ArrayBufferView> implements OrtTensorLike<TData> {
      disposed = 0;
      constructor(readonly type: string, readonly data: TData, readonly dims: readonly number[]) {}
      dispose(): void {
        this.disposed += 1;
      }
    }
    let jointCalls = 0;
    let decoderCalls = 0;
    const created: Tensor[] = [];
    const encoder: OrtSessionLike = {
      async run() {
        return {
          encoded: new Tensor('float32', new Float32Array(768), [1, 768, 1]),
          encoded_len: new Tensor('int32', new Int32Array([1]), [1]),
        };
      },
    };
    const decoder: OrtSessionLike = {
      async run() {
        decoderCalls += 1;
        return {
          dec: new Tensor('float32', new Float32Array(320), [1, 1, 320]),
          ho: new Tensor('float32', new Float32Array(320), [1, 1, 320]),
          co: new Tensor('float32', new Float32Array(320), [1, 1, 320]),
        };
      },
    };
    const signal = { aborted: false };
    const joint: OrtSessionLike = {
      async run() {
        jointCalls += 1;
        const logits = new Float32Array(35).fill(-5);
        logits[jointCalls === 1 ? 2 : 34] = 5;
        const tensor = new Tensor('float32', logits, [1, 1, 35]);
        created.push(tensor);
        if (jointCalls === 1) signal.aborted = true;
        return { joint: tensor };
      },
    };
    const ort: OrtModuleLike = {
      env: { wasm: {} },
      Tensor,
      InferenceSession: {
        create: async (url) => (url.includes('encoder') ? encoder : url.includes('decoder') ? decoder : joint),
      },
    };
    const executor = new OrtGigaAmRnntExecutor('gigaam-rnnt-abort', 'wasm', config, undefined);
    (executor as unknown as { state: Promise<unknown> }).state = Promise.resolve({
      ort,
      encoder,
      decoder,
      joint,
      tokenizer: GigaAmRnntTokenizer.fromText('  0\na 2\n<blk> 34\n'),
      warnings: [],
    });
    const audio = {
      sampleRate: 16000,
      numberOfChannels: 1,
      numberOfFrames: 16000,
      durationSeconds: 1,
      channels: [new Float32Array(16000)],
    };

    await expect(executor.transcribe(audio, { signal })).rejects.toBeInstanceOf(PipelineAbortedError);
    expect(jointCalls).toBe(1);
    expect(decoderCalls).toBe(1);
    expect(created.every((tensor) => tensor.disposed === 1)).toBe(true);

    signal.aborted = false;
    jointCalls = 0;
    decoderCalls = 0;
    const result = await executor.transcribe(audio);
    expect(result.utteranceText).toBe('a');
    expect(jointCalls).toBe(2);
    await executor.dispose();
  });
});
