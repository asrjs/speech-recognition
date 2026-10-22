import { describe, expect, it } from 'vitest';
import { CATALOG_DESCRIPTORS, CATALOG_LANGUAGES, getCatalogArtifactSource } from '../src/presets/catalog-data.generated.js';
import { BUILT_IN_MODEL_DESCRIPTORS } from '../src/presets/descriptors-catalog.js';
import { LANGUAGE_NAMES as parakeetLanguages, getLanguageName as parakeetName } from '../src/presets/parakeet/catalog.js';
import { LANGUAGE_NAMES as canaryLanguages, getLanguageName as canaryName } from '../src/presets/canary/catalog.js';
import { resolveParakeetArtifactSource } from '../src/presets/parakeet/manifest.js';
import { resolveCanaryArtifactSource } from '../src/presets/canary/manifest.js';
import { resolveMedAsrPresetManifest } from '../src/presets/medasr/manifest.js';
import { resolveNemotronArtifactSource } from '../src/presets/nemotron/factory.js';
import { getBuiltInModelLanguageName, listBuiltInModelDescriptors } from '../src/presets/descriptors.js';

describe('lightweight catalog data', () => {
  it('preserves all authoritative descriptor fields, including undefined properties', () => {
    expect(CATALOG_DESCRIPTORS).toStrictEqual(BUILT_IN_MODEL_DESCRIPTORS);
    expect(listBuiltInModelDescriptors()).toBe(CATALOG_DESCRIPTORS);
  });
  it('preserves artifact source options for every synchronous hub helper', () => {
    for (const d of BUILT_IN_MODEL_DESCRIPTORS) {
      const source = d.preset === 'parakeet' ? resolveParakeetArtifactSource(d.modelId)
        : d.preset === 'canary' ? resolveCanaryArtifactSource(d.modelId)
        : d.preset === 'medasr' ? resolveMedAsrPresetManifest(d.modelId)?.source
        : d.preset === 'nemotron' ? resolveNemotronArtifactSource(d.modelId) : undefined;
      expect(getCatalogArtifactSource(d.modelId), d.modelId).toStrictEqual(source);
    }
  });
  it('preserves language lookup precedence, case and unknown-language fallback', () => {
    for (const code of [...Object.keys(parakeetLanguages), ...Object.keys(canaryLanguages), 'auto', 'unknown']) {
      for (const value of [code, code.toUpperCase()]) {
        const p = parakeetName(value);
        const c = canaryName(value);
        const expected = p !== value ? p : c !== value ? c : value.toLowerCase() === 'auto' ? 'Auto-detect' : value;
        expect(getBuiltInModelLanguageName(value)).toBe(expected);
      }
    }
    expect(Object.keys(CATALOG_LANGUAGES).length).toBeGreaterThan(0);
  });
});
