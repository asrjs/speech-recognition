import { describe, expect, it } from 'vitest';
import * as parakeetExports from '../../src/presets/parakeet.js';

describe('parakeet index exports', () => {
  it('exports expected items from catalog', () => {
    expect(parakeetExports.MODELS).toBeDefined();
    expect(typeof parakeetExports.getModelConfig).toBe('function');
    expect(typeof parakeetExports.getModelKeyFromRepoId).toBe('function');
    expect(typeof parakeetExports.listModels).toBe('function');
  });

  it('exports expected items from compat', () => {
    expect(typeof parakeetExports.getParakeetModel).toBe('function');
    expect(typeof parakeetExports.ParakeetModel).toBe('function');
    expect(typeof parakeetExports.loadParakeetModelWithFallback).toBe('function');
  });

  it('exports expected items from factory', () => {
    expect(typeof parakeetExports.createParakeetPresetFactory).toBe('function');
  });

  it('exports expected items from manifest', () => {
    expect(parakeetExports.PARAKEET_PRESET_MANIFESTS).toBeDefined();
    expect(typeof parakeetExports.listParakeetPresetManifests).toBe('function');
    expect(typeof parakeetExports.resolveParakeetPresetManifest).toBe('function');
    expect(typeof parakeetExports.resolveParakeetArtifactSource).toBe('function');
  });
});
