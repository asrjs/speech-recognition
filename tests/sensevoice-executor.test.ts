import { describe, expect, it } from 'vitest';

import { OrtSenseVoiceExecutor } from '../src/models/sensevoice/executor.js';
import type { SenseVoiceHuggingFaceSource } from '../src/models/sensevoice/types.js';
import type { AssetProvider, ResolvedAssetHandle, RuntimeProgressEvent } from '../src/types/index.js';

function createHandle(filename: string, disposed: string[]): ResolvedAssetHandle {
  return {
    request: { id: filename, filename },
    async *openStream() { yield new Uint8Array(); },
    async readBytes() { return new Uint8Array(); },
    async readText() { return ''; },
    async readJson<T>() { return {} as T; },
    async getLocator(target) { return target === 'url' ? `blob:test/${filename}` : null; },
    dispose() { disposed.push(filename); },
  };
}

describe('SenseVoice artifact executor', () => {
  it('materializes Hugging Face assets through the provider and forwards progress', async () => {
    const requested: string[] = [];
    const disposed: string[] = [];
    const progress: RuntimeProgressEvent[] = [];
    const provider: AssetProvider = {
      canResolve: () => true,
      async resolve(request) {
        requested.push(request.filename ?? '');
        request.onProgress?.({ loaded: 10, total: 20, done: false });
        request.onProgress?.({ loaded: 20, total: 20, done: true });
        return createHandle(request.filename ?? '', disposed);
      },
    };
    const source: SenseVoiceHuggingFaceSource = {
      kind: 'huggingface',
      repoId: 'OpenVoiceOS/sensevoice-small-onnx',
      revision: 'pinned',
      modelFilename: 'model.onnx',
      modelDataFilename: 'model.onnx_data',
      tokenizerFilename: 'vocab.txt',
    };
    const executor = new OrtSenseVoiceExecutor('sensevoice-test', 'wasm', undefined, {
      assetProvider: provider,
      runtimeHooks: { onProgress: (event) => progress.push(event) },
    });
    const materialize = (executor as unknown as {
      materializeHuggingFaceArtifacts(
        source: SenseVoiceHuggingFaceSource,
        artifacts: Record<string, unknown>,
      ): Promise<{ artifacts: Record<string, unknown>; warnings: readonly unknown[] }>;
    }).materializeHuggingFaceArtifacts.bind(executor);

    const result = await materialize(source, {
      modelUrl: 'https://fallback/model.onnx',
      tokenizerUrl: 'https://fallback/vocab.txt',
      modelDataUrl: 'https://fallback/model.onnx_data',
      modelDataFilename: 'model.onnx_data',
    });

    expect(requested).toEqual(['vocab.txt', 'model.onnx', 'model.onnx_data']);
    expect(result.warnings).toEqual([]);
    expect(result.artifacts).toMatchObject({
      modelUrl: 'blob:test/model.onnx',
      tokenizerUrl: 'blob:test/vocab.txt',
      modelDataUrl: 'blob:test/model.onnx_data',
    });
    expect(progress).toHaveLength(6);
    expect(progress.at(-1)).toMatchObject({ phase: 'asset:download', file: 'model.onnx_data', percent: 100, isComplete: true });

    executor.dispose();
    expect(disposed).toEqual(['vocab.txt', 'model.onnx', 'model.onnx_data']);
  });
});
