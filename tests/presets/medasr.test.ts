import { describe, expect, it } from 'vitest';
import * as medasrExports from '../../src/presets/medasr.js';

describe('medasr index exports', () => {
  it('exports expected items from factory', () => {
    expect(typeof medasrExports.createMedAsrPresetFactory).toBe('function');
  });

  it('exports expected items from manifest', () => {
    expect(medasrExports.MEDASR_PRESET_MANIFESTS).toBeDefined();
    expect(typeof medasrExports.resolveMedAsrPresetManifest).toBe('function');
  });
});
