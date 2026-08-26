import { describe, expect, it } from 'vitest';
import { XAsrTokenizer, createXAsrModelFamily, type XAsrExecutor, type XAsrStreamState } from '../src/models/x-asr/index.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';

describe('X-ASR artifact-gated family', () => {
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
    const model = await family.createModel({ modelId: 'X-ASR-zh-en', options: { config: { graph: { encoderStateInputs: [], encoderFrameSize: 16, encoderFrameShift: 16, decoderContextSize: 2 } } } }, { backend: backend!, assetProvider: undefined, hooks: {} });
    await expect(model.createSession()).rejects.toThrow(/No X-ASR artifact source/);
    await runtime.dispose();
  });

  it('keeps streaming state in the model executor and releases it on reset', async () => {
    const calls: string[] = [];
    const emptyState = (): XAsrStreamState => ({ audio: new Float32Array(0), features: new Float32Array(0), encodedFrames: 0, inputFrames: 0, tokenIds: [], encoderStates: [] });
    const executor: XAsrExecutor = {
      async ready() {},
      createStream: emptyState,
      async pushStream(state, audio, final) {
        calls.push(final ? 'final' : `push:${audio.length}`);
        return { state, transcript: { utteranceText: final ? 'done' : 'partial', isFinal: final, warnings: [] } };
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
    const model = await runtime.loadModel({ family: 'x-asr', modelId: 'X-ASR-zh-en', backend: 'wasm', options: { config: { graph: { encoderStateInputs: [], encoderFrameSize: 16, encoderFrameShift: 16, decoderContextSize: 2 } } }, });
    const streamingModel = await createXAsrModelFamily({ dependencies: { executor } }).createModel({ modelId: 'X-ASR-zh-en' }, { backend: runtime.listBackends().find((item) => item.id === 'wasm')!, assetProvider: undefined, hooks: {} });
    await model.dispose();
    const transcriber = await streamingModel.createStreamingTranscriber();
    expect((await transcriber.pushAudio(new Float32Array(160))).text).toBe('partial');
    await transcriber.reset();
    await transcriber.finalize();
    expect(calls).toEqual(['push:160', 'dispose-stream', 'final', 'dispose-stream']);
    await streamingModel.dispose();
    await runtime.dispose();
  });
});
