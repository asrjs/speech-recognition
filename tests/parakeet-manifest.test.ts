import { describe, expect, it } from 'vitest';

import {
  listParakeetPresetManifests,
  resolveParakeetPresetManifest,
  resolveParakeetArtifactSource,
} from '../src/presets/parakeet/manifest.js';

describe('Parakeet preset manifests', () => {
  describe('listParakeetPresetManifests', () => {
    it('returns a list of preset manifests', () => {
      const manifests = listParakeetPresetManifests();
      expect(manifests.length).toBeGreaterThan(0);
      expect(manifests[0].preset).toBe('parakeet');
    });
  });

  describe('resolveParakeetPresetManifest', () => {
    it('resolves Parakeet by model ID', () => {
      const manifest = resolveParakeetPresetManifest('parakeet-tdt-0.6b-v2');
      expect(manifest?.modelId).toBe('parakeet-tdt-0.6b-v2');
    });

    it('returns undefined for unknown Parakeet model IDs', () => {
      expect(resolveParakeetPresetManifest('unknown/model')).toBeUndefined();
    });
  });

  describe('resolveParakeetArtifactSource', () => {
    it('resolves artifact source for known model IDs', () => {
      const source = resolveParakeetArtifactSource('parakeet-tdt-0.6b-v2');
      expect(source).toBeDefined();
    });

    it('returns undefined for unknown model IDs', () => {
      expect(resolveParakeetArtifactSource('unknown/model')).toBeUndefined();
    });
  });
});
