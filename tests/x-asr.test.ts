import { describe, expect, it } from 'vitest';
import { XAsrTokenizer, createXAsrModelFamily } from '../src/models/x-asr/index.js';
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
});
