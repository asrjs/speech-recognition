import { describe, it, expect } from 'vitest';
import * as builtins from '../src/builtins.js';

describe('src/builtins.ts exports', () => {
  it('should export registerBuiltInBackends', () => {
    expect(builtins.registerBuiltInBackends).toBeDefined();
    expect(typeof builtins.registerBuiltInBackends).toBe('function');
  });

  it('should export registerBuiltInModelFamilies', () => {
    expect(builtins.registerBuiltInModelFamilies).toBeDefined();
    expect(typeof builtins.registerBuiltInModelFamilies).toBe('function');
  });

  it('should export registerBuiltInPresets', () => {
    expect(builtins.registerBuiltInPresets).toBeDefined();
    expect(typeof builtins.registerBuiltInPresets).toBe('function');
  });

  it('should export createBuiltInSpeechRuntime', () => {
    expect(builtins.createBuiltInSpeechRuntime).toBeDefined();
    expect(typeof builtins.createBuiltInSpeechRuntime).toBe('function');
  });

  it('should export loadBuiltInSpeechModel', () => {
    expect(builtins.loadBuiltInSpeechModel).toBeDefined();
    expect(typeof builtins.loadBuiltInSpeechModel).toBe('function');
  });
});
