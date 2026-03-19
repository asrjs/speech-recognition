import { describe, expect, it } from 'vitest';

import {
  createModelArchitecture,
  type ModelArchitectureDescriptor,
} from '../src/types/architecture.js';

describe('createModelArchitecture', () => {
  it('returns the original descriptor reference unchanged', () => {
    const descriptor: ModelArchitectureDescriptor = {
      processor: { layer: 'processor', module: 'processor-module', implementation: 'processor' },
      encoder: { layer: 'encoder', module: 'encoder-module', implementation: 'encoder' },
      decoder: { layer: 'decoder', module: 'decoder-module', implementation: 'decoder' },
      decoding: { layer: 'decoding', module: 'decoding-module', implementation: 'decoding' },
      tokenizer: { layer: 'tokenizer', module: 'tokenizer-module', implementation: 'tokenizer' },
    };

    expect(createModelArchitecture(descriptor)).toBe(descriptor);
  });

  it('preserves optional descriptor metadata', () => {
    const descriptor: ModelArchitectureDescriptor = {
      processor: {
        layer: 'processor',
        module: 'processor-module',
        implementation: 'processor',
        shared: true,
        notes: ['shared across presets'],
      },
      encoder: {
        layer: 'encoder',
        module: 'encoder-module',
        implementation: 'encoder',
        shared: false,
      },
      decoder: {
        layer: 'decoder',
        module: 'decoder-module',
        implementation: 'decoder',
        notes: ['model specific'],
      },
      decoding: { layer: 'decoding', module: 'decoding-module', implementation: 'decoding' },
      tokenizer: { layer: 'tokenizer', module: 'tokenizer-module', implementation: 'tokenizer' },
    };

    const result = createModelArchitecture(descriptor);

    expect(result).toBe(descriptor);
    expect(result.processor.shared).toBe(true);
    expect(result.processor.notes).toEqual(['shared across presets']);
    expect(result.encoder.shared).toBe(false);
    expect(result.decoder.notes).toEqual(['model specific']);
  });
});
