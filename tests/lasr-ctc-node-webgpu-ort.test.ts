import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { describe, expect, it, vi } from 'vitest';

import { createOrtSession, initOrt } from '../src/models/lasr-ctc/ort.js';

import * as nodeOrtModule from 'onnxruntime-node';

describe('lasr-ctc shared ORT selection in Node-like runtimes', () => {
  it('loads onnxruntime-node for WebGPU backends', async () => {
    const ort = await initOrt('webgpu');
    expect(ort.env).toBe(nodeOrtModule.env);
  });

  it('passes a plain webgpu execution provider to the native session', async () => {
    const create = vi.fn(async () => ({
      run: async () => ({}),
      release: async () => undefined,
    }));
    const ort = {
      env: { wasm: {} },
      Tensor: class {},
      InferenceSession: { create },
    } as unknown as Awaited<ReturnType<typeof initOrt>>;
    await createOrtSession(ort, 'C:/models/model.onnx', {
      backendId: 'webgpu',
      preferredOutputLocation: { logits: 'cpu', state: 'gpu-buffer' },
    });
    expect(create).toHaveBeenCalledTimes(1);
    const [, options] = create.mock.calls[0] as unknown as [string, Record<string, unknown>];
    expect(options.executionProviders).toEqual(['webgpu']);
    expect(options.preferredOutputLocation).toEqual({ logits: 'cpu', state: 'cpu' });
  });

  it('keeps the wasm backend on onnxruntime-web', async () => {
    const ort = await initOrt('wasm');
    expect(ort.env).not.toBe(nodeOrtModule.env);
  });

  it('mounts colocated external data for Node-hosted ORT Web WASM', async () => {
    const create = vi.fn(async () => ({ run: async () => ({}) }));
    const ort = {
      env: { wasm: {} },
      Tensor: class {},
      InferenceSession: { create },
    } as unknown as Awaited<ReturnType<typeof initOrt>>;
    const modelUrl = pathToFileURL(path.resolve('package.json')).href;
    const externalDataUrl = pathToFileURL(path.resolve('package-lock.json')).href;

    await createOrtSession(ort, modelUrl, {
      backendId: 'wasm',
      externalDataUrl,
      externalDataPath: 'package-lock.json',
    });

    const [, options] = create.mock.calls[0] as unknown as [string, Record<string, unknown>];
    const externalData = options.externalData as Array<{ data: Uint8Array; path: string }>;
    expect(externalData[0]?.path).toBe('package-lock.json');
    expect(externalData[0]?.data.byteLength).toBeGreaterThan(0);
  });
});
