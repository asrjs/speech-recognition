import { describe, it, expect } from 'vitest';
import {
  createModelArchitecture,
  ModelArchitectureDescriptor,
} from '../src/types/architecture.js';

describe('createModelArchitecture', () => {
  it('should return the provided descriptor object unmodified', () => {
    const descriptor: ModelArchitectureDescriptor = {
      processor: { layer: 'processor', module: 'test-module', implementation: 'test-impl' },
      encoder: { layer: 'encoder', module: 'test-module', implementation: 'test-impl' },
      decoder: { layer: 'decoder', module: 'test-module', implementation: 'test-impl' },
      decoding: { layer: 'decoding', module: 'test-module', implementation: 'test-impl' },
      tokenizer: { layer: 'tokenizer', module: 'test-module', implementation: 'test-impl' },
    };

    const result = createModelArchitecture(descriptor);

    // It should return the exact same object reference
    expect(result).toBe(descriptor);

    // It should have the correct shape
    expect(result).toEqual({
      processor: { layer: 'processor', module: 'test-module', implementation: 'test-impl' },
      encoder: { layer: 'encoder', module: 'test-module', implementation: 'test-impl' },
      decoder: { layer: 'decoder', module: 'test-module', implementation: 'test-impl' },
      decoding: { layer: 'decoding', module: 'test-module', implementation: 'test-impl' },
      tokenizer: { layer: 'tokenizer', module: 'test-module', implementation: 'test-impl' },
    });
  });

  it('should preserve optional fields like shared and notes', () => {
    const descriptor: ModelArchitectureDescriptor = {
      processor: {
        layer: 'processor',
        module: 'test-module',
        implementation: 'test-impl',
        shared: true,
        notes: ['note1'],
      },
      encoder: {
        layer: 'encoder',
        module: 'test-module',
        implementation: 'test-impl',
        shared: false,
      },
      decoder: {
        layer: 'decoder',
        module: 'test-module',
        implementation: 'test-impl',
        notes: ['note2', 'note3'],
      },
      decoding: { layer: 'decoding', module: 'test-module', implementation: 'test-impl' },
      tokenizer: { layer: 'tokenizer', module: 'test-module', implementation: 'test-impl' },
    };

    const result = createModelArchitecture(descriptor);

    expect(result).toBe(descriptor);
    expect(result.processor.shared).toBe(true);
    expect(result.processor.notes).toEqual(['note1']);
    expect(result.encoder.shared).toBe(false);
    expect(result.decoder.notes).toEqual(['note2', 'note3']);
  });
});
