import { describe, expect, it } from 'vitest';
import {
  mapXAsrNativeToCanonical,
  DEFAULT_XASR_ENCODER_STATE_OUTPUTS,
  XAsrTokenizer,
  createXAsrModelFamily,
  type XAsrExecutor,
  type XAsrStreamState,
} from '../src/models/x-asr/index.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';

describe('X-ASR artifact-gated family', () => {
  it('defines every official Zipformer2 state output for GPU-resident streaming caches', () => {
    expect(DEFAULT_XASR_ENCODER_STATE_OUTPUTS).toHaveLength(116);
    expect(DEFAULT_XASR_ENCODER_STATE_OUTPUTS[0]).toBe('new_cached_key_0');
    expect(DEFAULT_XASR_ENCODER_STATE_OUTPUTS.at(-1)).toBe('new_processed_lens');
  });

  it('maps transducer-native output through the X-ASR family contract', () => {
    const result = mapXAsrNativeToCanonical(
      {
        utteranceText: 'hello world',
        isFinal: true,
        words: [{ index: 0, text: 'hello world', startTime: 0, endTime: 1 }],
        tokens: [
          { index: 0, id: 1, text: 'hello', startTime: 0, endTime: 0.5 },
          { index: 1, id: 2, text: ' world', startTime: 0.5, endTime: 1 },
        ],
        warnings: [],
      },
      {
        family: 'x-asr',
        ecosystem: 'x-asr',
        processor: 'kaldi-fbank',
        encoder: 'zipformer2',
        decoder: 'stateless-rnnt',
        topology: 'stateless-rnnt',
        task: 'asr',
      },
      {
        detailLevel: 'detailed',
        modelId: 'x-asr-test',
        backendId: 'wasm',
        sampleRate: 16000,
        durationSeconds: 1,
      },
    );

    expect(result.text).toBe('hello world');
    expect(result.meta.modelFamily).toBe('x-asr');
    expect(result.words?.[0]?.tokenIndices).toEqual([0, 1]);
    expect(result.tokens?.map((token) => token.id)).toEqual([1, 2]);
  });

  it('decodes icefall token text without exposing blank/control pieces', () => {
    const tokenizer = XAsrTokenizer.fromText('<blk> 0\n▁hello 1\n▁world 2\n<eps> 3\n');
    expect(tokenizer.decode([0, 1, 2, 3])).toBe('hello world');
    expect(tokenizer.decodeTokenPiece(0)).toBe('');
  });

  it('is discoverable but refuses to initialize without explicit artifacts', async () => {
    const runtime = createBuiltInSpeechRuntime();
    expect(runtime.listModelFamilies().some((family) => family.family === 'x-asr')).toBe(true);
    const family = createXAsrModelFamily();
    const backend = runtime.listBackends().find((item) => item.id === 'wasm');
    expect(backend).toBeDefined();
    const model = await family.createModel(
      {
        modelId: 'X-ASR-zh-en',
        options: {
          config: {
            graph: {
              encoderStateInputs: [],
              encoderFrameSize: 16,
              encoderFrameShift: 16,
              decoderContextSize: 2,
            },
          },
        },
      },
      { backend: backend!, assetProvider: undefined, hooks: {} },
    );
    await expect(model.createSession()).rejects.toThrow(/No X-ASR artifact source/);
    await runtime.dispose();
  });

  it('keeps streaming state in the model executor and releases it on reset', async () => {
    const calls: string[] = [];
    const emptyState = (): XAsrStreamState => ({
      audio: new Float32Array(0),
      features: new Float32Array(0),
      encodedFrames: 0,
      inputFrames: 0,
      tokenIds: [],
      encoderStates: [],
    });
    const executor: XAsrExecutor = {
      async ready() {},
      createStream: emptyState,
      async pushStream(state, audio, final) {
        calls.push(final ? 'final' : `push:${audio.length}`);
        return {
          state,
          transcript: { utteranceText: final ? 'done' : 'partial', isFinal: final, warnings: [] },
        };
      },
      transcribe: async () => ({ utteranceText: '', isFinal: true, warnings: [] }),
      disposeStream() {
        calls.push('dispose-stream');
      },
      dispose() {
        calls.push('dispose-executor');
      },
    };
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    const model = await runtime.loadModel({
      family: 'x-asr',
      modelId: 'X-ASR-zh-en',
      backend: 'wasm',
      options: {
        config: {
          graph: {
            encoderStateInputs: [],
            encoderFrameSize: 16,
            encoderFrameShift: 16,
            decoderContextSize: 2,
          },
        },
      },
    });
    const streamingModel = await createXAsrModelFamily({ dependencies: { executor } }).createModel(
      { modelId: 'X-ASR-zh-en' },
      {
        backend: runtime.listBackends().find((item) => item.id === 'wasm')!,
        assetProvider: undefined,
        hooks: {},
      },
    );
    await model.dispose();
    const transcriber = await streamingModel.createStreamingTranscriber();
    expect((await transcriber.pushAudio(new Float32Array(160))).text).toBe('partial');
    await transcriber.reset();
    await transcriber.finalize();
    expect(calls).toEqual(['push:160', 'dispose-stream', 'final', 'dispose-stream']);
    await streamingModel.dispose();
    await runtime.dispose();
  });

  it('serializes streaming calls and invalidates an in-flight result on reset', async () => {
    const emptyState = (): XAsrStreamState => ({
      audio: new Float32Array(0),
      features: new Float32Array(0),
      encodedFrames: 0,
      inputFrames: 0,
      tokenIds: [],
      encoderStates: [],
    });
    let calls = 0;
    let resolveFirst!: (result: {
      state: XAsrStreamState;
      transcript: XAsrNativeTranscript;
    }) => void;
    const firstResult = new Promise<{ state: XAsrStreamState; transcript: XAsrNativeTranscript }>(
      (resolve) => {
        resolveFirst = resolve;
      },
    );
    const signals: AbortSignal[] = [];
    const executor: XAsrExecutor = {
      async ready() {},
      createStream: emptyState,
      async pushStream(state, _audio, _final, options) {
        calls += 1;
        signals.push(options?.signal as AbortSignal);
        if (calls === 1) return firstResult;
        return { state, transcript: { utteranceText: 'fresh', isFinal: false, warnings: [] } };
      },
      transcribe: async () => ({ utteranceText: '', isFinal: true, warnings: [] }),
      disposeStream() {},
      dispose() {},
    };
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    const model = await createXAsrModelFamily({ dependencies: { executor } }).createModel(
      { modelId: 'X-ASR-zh-en' },
      {
        backend: runtime.listBackends().find((item) => item.id === 'wasm')!,
        assetProvider: undefined,
        hooks: {},
      },
    );
    const transcriber = await model.createStreamingTranscriber();

    const first = transcriber.pushAudio(new Float32Array(160));
    const second = transcriber.pushAudio(new Float32Array(160));
    await Promise.resolve();
    expect(calls).toBe(1);
    await transcriber.reset();
    expect(signals[0]?.aborted).toBe(true);
    resolveFirst({
      state: emptyState(),
      transcript: { utteranceText: 'stale', isFinal: false, warnings: [] },
    });

    expect((await first).text).toBe('');
    expect((await second).text).toBe('');
    expect((await transcriber.pushAudio(new Float32Array(160))).text).toBe('fresh');
    expect(calls).toBe(2);

    await transcriber.dispose();
    await model.dispose();
    await runtime.dispose();
  });
});
