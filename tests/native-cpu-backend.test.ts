import { afterEach, describe, expect, it, vi } from 'vitest';

const host = vi.hoisted(() => ({ node: true, importPackage: vi.fn() }));
vi.mock('../src/io/node-compat.js', () => ({
  isNodeLikeRuntime: () => host.node,
  importNodePackage: host.importPackage,
}));

import {
  createNativeCpuBackend,
  probeNativeCpuCapabilities,
} from '../src/inference/backends/native-cpu/index.js';
import { loadNativeCpuOrt } from '../src/inference/backends/native-cpu/ort.js';

afterEach(() => {
  host.node = true;
  host.importPackage.mockReset();
});

describe('native CPU backend', () => {
  it('does not touch the native binding when probed in a browser', async () => {
    host.node = false;
    expect((await probeNativeCpuCapabilities()).available).toBe(false);
    expect(host.importPackage).not.toHaveBeenCalled();
    expect(() => loadNativeCpuOrt()).toThrow('Node-compatible');
  });

  it('honors an explicitly requested browser environment on a Node host', async () => {
    expect((await probeNativeCpuCapabilities({ environment: 'browser' })).available).toBe(false);
    expect(host.importPackage).not.toHaveBeenCalled();
  });

  it('reports the actual engine/provider and requires family opt-in', async () => {
    host.importPackage.mockReturnValue({
      listSupportedBackends: () => [{ name: 'cpu', bundled: true }],
    });
    const backend = createNativeCpuBackend();
    const caps = await backend.probeCapabilities();
    expect(backend.requiresModelOptIn).toBe(true);
    expect(caps).toMatchObject({
      available: true,
      provider: 'onnxruntime-node/cpu',
      experimental: true,
      supportedPrecisions: ['fp32'],
    });
    expect(host.importPackage).toHaveBeenCalledWith('onnxruntime-node');
    const lease = await backend.createExecutionContext({
      modelFamily: 'example',
      modelId: 'example',
    });
    expect(lease.backendId).toBe('native-cpu');
    await lease.dispose();
    await lease.dispose();
  });

  it('fails clearly rather than switching to WASM when the native package is missing', async () => {
    host.importPackage.mockImplementation(() => {
      throw new Error('missing binding');
    });
    expect((await probeNativeCpuCapabilities()).available).toBe(false);
    expect(() => loadNativeCpuOrt()).toThrow('could not be loaded');
    await expect(
      createNativeCpuBackend().createExecutionContext({
        modelFamily: 'example',
        modelId: 'example',
      }),
    ).rejects.toMatchObject({ code: 'backend-unavailable' });
  });

  it('does not advertise a CPU provider that is not bundled', async () => {
    host.importPackage.mockReturnValue({
      listSupportedBackends: () => [{ name: 'cpu', bundled: false }],
    });
    expect((await probeNativeCpuCapabilities()).available).toBe(false);
  });
});
