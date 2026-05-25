import { describe, expect, it } from 'vitest';
import { resolveMedAsrPresetManifest } from '../src/presets/medasr/manifest.js';

describe('resolveMedAsrPresetManifest', () => {
  it('should resolve preset by exact modelId', () => {
    const manifest = resolveMedAsrPresetManifest('google/medasr');
    expect(manifest).toBeDefined();
    expect(manifest?.modelId).toBe('google/medasr');
  });

  it('should return undefined for unknown modelId', () => {
    const manifest = resolveMedAsrPresetManifest('unknown/model');
    expect(manifest).toBeUndefined();
  });
});
