import { describe, expect, it } from 'vitest';

import { GigaAmJsPreprocessor, GigaAmTokenizer } from '../src/models/gigaam-ctc/index.js';

describe('GigaAM Multilingual CTC contract', () => {
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
