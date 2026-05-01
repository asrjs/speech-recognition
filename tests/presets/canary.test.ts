import { describe, expect, it } from 'vitest';
import * as canaryExports from '../../src/presets/canary.js';

describe('canary index exports', () => {
  it('exports expected items from catalog', () => {
    expect(canaryExports.LANGUAGE_NAMES).toBeDefined();
    expect(canaryExports.MODELS).toBeDefined();
    expect(typeof canaryExports.getModelConfig).toBe('function');
    expect(typeof canaryExports.getModelKeyFromRepoId).toBe('function');
    expect(typeof canaryExports.listModels).toBe('function');
    expect(typeof canaryExports.getLanguageName).toBe('function');
  });

  it('exports expected items from compat', () => {
    expect(typeof canaryExports.getCanaryModel).toBe('function');
    expect(typeof canaryExports.CanaryModel).toBe('function');
    expect(typeof canaryExports.transcribeCanary).toBe('function');
  });

  it('exports expected items from factory', () => {
    expect(typeof canaryExports.createCanaryPresetFactory).toBe('function');
  });

  it('exports expected items from manifest', () => {
    expect(canaryExports.CANARY_180M_FLASH_DOCS).toBeDefined();
    expect(canaryExports.DEFAULT_MODEL).toBeDefined();
    expect(typeof canaryExports.resolveCanaryArtifactSource).toBe('function');
  });
});
