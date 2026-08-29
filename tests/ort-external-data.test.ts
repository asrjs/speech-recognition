import path from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

import * as nodeCompat from '../src/io/node-compat.js';
import { resolveOrtExternalDataMounts } from '../src/io/ort-external-data.js';

describe('resolveOrtExternalDataMounts', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('passes browser URL locators through unchanged', async () => {
    vi.spyOn(nodeCompat, 'isNodeLikeRuntime').mockReturnValue(false);

    const mounts = await resolveOrtExternalDataMounts({
      backendId: 'wasm',
      sessionModelUrl: 'https://example.test/model.onnx',
      externalDataUrl: 'https://example.test/model.onnx.data',
      externalDataPath: 'model.onnx.data',
    });

    expect(mounts).toEqual([
      { data: 'https://example.test/model.onnx.data', path: 'model.onnx.data' },
    ]);
  });

  it('reads byte buffers for Node-hosted ORT Web WASM', async () => {
    const modelUrl = path.resolve('package.json');
    const externalDataUrl = path.resolve('package-lock.json');

    const mounts = await resolveOrtExternalDataMounts({
      backendId: 'wasm',
      sessionModelUrl: modelUrl,
      externalDataUrl,
      externalDataPath: 'package-lock.json',
    });

    expect(mounts?.[0]?.path).toBe('package-lock.json');
    expect(mounts?.[0]?.data).toBeInstanceOf(Uint8Array);
    expect((mounts?.[0]?.data as Uint8Array).byteLength).toBeGreaterThan(0);
  });

  it('omits mounts for native Node WebGPU when colocated data exists', async () => {
    const modelUrl = path.resolve('package.json');
    const externalDataUrl = path.resolve('package-lock.json');

    const mounts = await resolveOrtExternalDataMounts({
      backendId: 'webgpu',
      sessionModelUrl: modelUrl,
      externalDataUrl,
      externalDataPath: 'package-lock.json',
    });

    expect(mounts).toBeUndefined();
  });

  it('still mounts bytes for Node WebGPU when colocated data is missing', async () => {
    const modelUrl = path.resolve('package.json');

    const mounts = await resolveOrtExternalDataMounts({
      backendId: 'webgpu',
      sessionModelUrl: modelUrl,
      externalDataUrl: path.resolve('README.md'),
      externalDataPath: 'missing.onnx.data',
    });

    expect(mounts?.[0]?.path).toBe('missing.onnx.data');
    expect((mounts?.[0]?.data as Uint8Array).byteLength).toBeGreaterThan(0);
  });
});
