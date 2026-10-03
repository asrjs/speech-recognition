import { describe, expect, it } from 'vitest';

import {
  listParakeetPresetManifests,
  resolveParakeetArtifactSource,
  resolveParakeetPresetManifest,
} from '../src/presets/parakeet/manifest.js';
import { createParakeetPresetFactory } from '../src/presets/parakeet/factory.js';
import { getBuiltInModelDescriptor } from '../src/presets/descriptors.js';

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

  it('resolves the audited Orukeet preset contract through the factory', async () => {
    const manifest = resolveParakeetPresetManifest('oruk/orukeet');

    expect(manifest?.preset).toBe('parakeet');
    expect(manifest?.config.vocabularySize).toBe(8193);
    expect(manifest?.config.tokenizer?.blankTokenId).toBe(8192);
    expect(manifest?.config.melBins).toBe(128);
    expect(manifest?.config.languages).toHaveLength(25);
    expect(manifest?.config.languages).toContain('lt');

    // Registration is only real when the preset factory resolves the same
    // audited config; the native CLI passed an explicit runtime config and
    // would have masked an empty merge here.
    const factory = createParakeetPresetFactory();
    expect(factory.supports('oruk/orukeet')).toBe(true);

    const request = await factory.resolveModelRequest(
      { modelId: 'oruk/orukeet' },
      undefined as never,
    );

    expect(request.family).toBe('nemo-tdt');
    expect(request.options.config).toMatchObject({
      vocabularySize: 8193,
      melBins: 128,
      tokenizer: { kind: 'sentencepiece', blankTokenId: 8192 },
    });
    expect(request.options.config.languages).toHaveLength(25);
  });

  it('does not advertise a Hub source for Orukeet', () => {
    // Upstream revision b59c13a7 publishes graphs under onnx/combined-v0.1.0-int8/
    // and a tokenizer.json, so the root-level layout this preset resolves would
    // 404. Orukeet stays local-folder only until a validated export source exists.
    expect(resolveParakeetArtifactSource('oruk/orukeet')).toBeUndefined();
    const descriptor = getBuiltInModelDescriptor('oruk/orukeet');
    expect(descriptor?.loading.supportsHubSource).toBe(false);
    expect(descriptor?.loading.supportsLocalSource).toBe(true);
  });

  it('returns undefined for unknown Parakeet model ids', () => {
    expect(resolveParakeetPresetManifest('unknown/model')).toBeUndefined();
    expect(resolveParakeetArtifactSource('unknown/model')).toBeUndefined();
  });
});
