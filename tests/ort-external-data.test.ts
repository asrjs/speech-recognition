import path from 'node:path';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import { pathToFileURL } from 'node:url';

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

  it('reads byte buffers for Node file URL locators', async () => {
    const modelPath = path.resolve('package.json');
    const dataPath = path.resolve('package-lock.json');
    const mounts = await resolveOrtExternalDataMounts({
      backendId: 'wasm',
      sessionModelUrl: pathToFileURL(modelPath).href,
      externalDataUrl: pathToFileURL(dataPath).href,
      externalDataPath: 'package-lock.json',
    });

    expect(mounts?.[0]?.data).toBeInstanceOf(Uint8Array);
    expect((mounts?.[0]?.data as Uint8Array).byteLength).toBeGreaterThan(0);
  });

  it('preserves valid URL locators in Node instead of reading them as filesystem paths', async () => {
    const blobUrl = URL.createObjectURL(new Blob([new Uint8Array([31, 32, 33])]));
    try {
      const locators = [
        'https://example.test/weights.onnx.data',
        blobUrl,
        'data:application/octet-stream;base64,Hh8=',
      ];
      for (const externalDataUrl of locators) {
        const mounts = await resolveOrtExternalDataMounts({
          backendId: 'wasm',
          sessionModelUrl: 'https://example.test/model.onnx',
          externalDataUrl,
          externalDataPath: 'model.onnx.data',
        });
        expect(mounts).toEqual([{ data: externalDataUrl, path: 'model.onnx.data' }]);
      }
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  });

  it('mounts colocated bytes for Node-hosted ORT WebGPU', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'asrjs-ort-webgpu-'));
    const modelUrl = path.join(directory, 'model.onnx');
    const externalDataUrl = path.join(directory, 'model.onnx.data');
    const expected = new Uint8Array([4, 5, 6]);
    try {
      await writeFile(modelUrl, new Uint8Array([1]));
      await writeFile(externalDataUrl, expected);
      const mounts = await resolveOrtExternalDataMounts({
        backendId: 'webgpu',
        sessionModelUrl: modelUrl,
        externalDataUrl,
        externalDataPath: 'model.onnx.data',
      });

      expect(mounts?.[0]?.path).toBe('model.onnx.data');
      expect(Array.from(mounts?.[0]?.data ?? [])).toEqual(Array.from(expected));
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
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
