import { describe, expect, it } from 'vitest';

import { materializeNemoTdtHuggingFaceArtifacts } from '../src/models/nemo-tdt/executor-artifacts.js';
import { resolveNemoTdtArtifacts } from '../src/models/nemo-tdt/ort.js';
import { resolveParakeetPresetManifest } from '../src/presets/parakeet/manifest.js';
import type { NemoTdtArtifactSource } from '../src/models/nemo-tdt/types.js';
import type { ResolvedAssetHandle } from '../src/types/index.js';

function createRecordingProvider() {
  const requested: string[] = [];
  const provider = {
    async resolve(request: { readonly filename: string }): Promise<ResolvedAssetHandle> {
      requested.push(request.filename);
      return {
        async getLocator() {
          return `blob:probe/${request.filename}`;
        },
        dispose() {},
      } as ResolvedAssetHandle;
    },
  };
  return { provider, requested };
}

describe('Parakeet Hugging Face artifact sidecars', () => {
  it('resolves the fp32 encoder external-data sidecar from the pinned file list', async () => {
    const manifest = resolveParakeetPresetManifest('parakeet-tdt-0.6b-v3');
    const source = { ...manifest.source, encoderQuant: 'fp32' } as NemoTdtArtifactSource;
    const { provider, requested } = createRecordingProvider();

    const artifacts = resolveNemoTdtArtifacts(source, 'webgpu').artifacts;
    const materialized = await materializeNemoTdtHuggingFaceArtifacts(
      { modelId: manifest.modelId, sourceOptions: source, assetProvider: provider, assetHandles: [] },
      artifacts,
    );

    expect(materialized.encoderFilename).toBe('encoder-model.onnx');
    expect(materialized.encoderDataUrl).toBe('blob:probe/encoder-model.onnx.data');
    expect(requested).toContain('encoder-model.onnx.data');
  });

});
