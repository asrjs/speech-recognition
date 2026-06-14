import { describe, expect, it } from 'vitest';

import {
  listParakeetPresetManifests,
  resolveParakeetArtifactSource,
  resolveParakeetPresetManifest,
} from '../src/presets/parakeet/manifest.js';

describe('Parakeet preset manifests', () => {
  it('lists Parakeet preset manifests', () => {
    const manifests = listParakeetPresetManifests();

    expect(manifests.length).toBeGreaterThan(0);
    expect(manifests.every((manifest) => manifest.preset === 'parakeet')).toBe(true);
  });

  it('resolves manifests and artifact sources for known model ids', () => {
    const manifest = resolveParakeetPresetManifest('parakeet-tdt-0.6b-v2');
    const source = resolveParakeetArtifactSource('parakeet-tdt-0.6b-v2');

    expect(manifest?.modelId).toBe('parakeet-tdt-0.6b-v2');
    expect(source?.kind).toBe('huggingface');
    expect(source?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v2-onnx');
    expect(source?.revision).toBe('main');
  });

  it('returns undefined for unknown Parakeet model ids', () => {
    expect(resolveParakeetPresetManifest('unknown/model')).toBeUndefined();
    expect(resolveParakeetArtifactSource('unknown/model')).toBeUndefined();
  });
});
