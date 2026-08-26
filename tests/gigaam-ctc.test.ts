import { describe, expect, it } from 'vitest';

import { GigaAmJsPreprocessor, GigaAmTokenizer } from '../src/models/gigaam-ctc/index.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';
import { loadSpeechModel } from '../src/runtime/load.js';

describe('GigaAM Multilingual CTC contract', () => {
  it('is discoverable but remains artifact-gated', async () => {
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    expect(runtime.listModelFamilies().find((family) => family.family === 'gigaam-ctc')?.supports('gigaam-multilingual-ctc')).toBe(true);
    await expect(loadSpeechModel({ family: 'gigaam-ctc', modelId: 'gigaam-multilingual-ctc', backend: 'wasm' })).rejects.toThrow(/No GigaAM artifact source/);
    await expect(loadSpeechModel({ modelId: 'gigaam-multilingual-ctc', backend: 'wasm' })).rejects.toThrow(/No GigaAM artifact source/);
  });

  it('uses the published 64-bin, 320/320/160 feature geometry', () => {
    const processor = new GigaAmJsPreprocessor();
    const result = processor.process(new Float32Array(16000));

    expect(result.featureSize).toBe(64);
    expect(result.frameCount).toBe(99);
    expect(result.features.length).toBe(64 * 99);
  });

  it('decodes the character vocabulary and final CTC blank', () => {
    const tokenizer = GigaAmTokenizer.fromText("▁ 0\na 2\n' 1\n<blk> 70\n");

    expect(tokenizer.blankId).toBe(70);
    expect(tokenizer.decode([0, 2, 1, 70])).toBe("a'");
  });
});
