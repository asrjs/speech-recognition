import { describe, expect, it } from 'vitest';

import { resolveMedAsrPresetManifest } from '../src/presets/medasr/manifest.js';

describe('resolveMedAsrPresetManifest', () => {
  it('resolves MedASR by canonical model ID and aliases', () => {
    const manifest = resolveMedAsrPresetManifest('google/medasr');

    expect(manifest?.modelId).toBe('google/medasr');
    expect(resolveMedAsrPresetManifest('medasr')).toBe(manifest);
    expect(resolveMedAsrPresetManifest('google-medasr')).toBe(manifest);
  });

  it('normalizes model IDs before lookup', () => {
    expect(resolveMedAsrPresetManifest('  GOOGLE/MEDASR  ')?.modelId).toBe('google/medasr');
    expect(resolveMedAsrPresetManifest('  Google-MedASR  ')?.modelId).toBe('google/medasr');
  });

  it('returns undefined for unknown MedASR model IDs', () => {
    expect(resolveMedAsrPresetManifest('unknown/model')).toBeUndefined();
  });
});
