import {
  createWasmBackend,
  createWebGlBackend,
  createWebGpuBackend,
  createWebNnBackend,
  probeWebGlCapabilities,
  probeWasmCapabilities,
  probeWebGpuCapabilities,
  probeWebNnCapabilities,
} from '@asrjs/speech-recognition';
import { afterEach, describe, expect, it } from 'vitest';

const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');

function restoreGlobal(
  key: 'navigator' | 'document',
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) {
    Object.defineProperty(globalThis, key, descriptor);
    return;
  }

  delete (globalThis as Record<string, unknown>)[key];
}

afterEach(() => {
  restoreGlobal('navigator', originalNavigator);
  restoreGlobal('document', originalDocument);
});

describe('backend probes', () => {
  it('reports WASM as a universal baseline', async () => {
    const caps = await probeWasmCapabilities();

    expect(caps.id).toBe('wasm');
    expect(caps.supportedPrecisions).toContain('fp32');
    expect(caps.fallbackSuitable).toBe(true);
  });

  it('creates an idempotently disposable WASM execution context', async () => {
    const context = await createWasmBackend().createExecutionContext({
      modelFamily: 'test-family',
      modelId: 'test-model',
      precision: 'fp32',
    });

    expect(context.backendId).toBe('wasm');
    expect(context.capabilities.id).toBe('wasm');
    expect(() => context.dispose()).not.toThrow();
    expect(() => context.dispose()).not.toThrow();
  });

  it('rejects unsupported precision at context creation', async () => {
    await expect(
      createWasmBackend().createExecutionContext({
        modelFamily: 'test-family',
        modelId: 'test-model',
        precision: 'fp16',
      }),
    ).rejects.toMatchObject({
      name: 'CapabilityMismatchError',
      code: 'capability-mismatch',
    });
  });

  it('detects WebGPU capabilities from a mocked navigator', async () => {
    const adapterRequests: unknown[] = [];
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          async requestAdapter(options?: unknown) {
            adapterRequests.push(options);
            return {
              features: {
                has(feature: string) {
                  return feature === 'shader-f16';
                },
              },
              info: {
                vendor: 'MockVendor',
                architecture: 'MockArchitecture',
              },
            };
          },
        },
      },
    });

    const caps = await probeWebGpuCapabilities();
    expect(caps.available).toBe(true);
    expect(caps.supportsFp16).toBe(true);
    expect(caps.provider).toBe('MockVendor');
    expect(adapterRequests).toEqual([{ powerPreference: 'high-performance' }]);

    const context = await createWebGpuBackend().createExecutionContext({
      modelFamily: 'test-family',
      modelId: 'test-model',
      precision: 'fp16',
    });
    expect(context.provider).toBe('MockVendor');
    await context.dispose();
  });

  it('falls back to default adapter selection when high-performance selection is unavailable', async () => {
    const adapterRequests: unknown[] = [];
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          async requestAdapter(options?: { readonly powerPreference?: string }) {
            adapterRequests.push(options);
            if (options?.powerPreference === 'high-performance') return null;
            return { info: { vendor: 'FallbackVendor' } };
          },
        },
      },
    });

    const caps = await probeWebGpuCapabilities();

    expect(caps.available).toBe(true);
    expect(caps.provider).toBe('FallbackVendor');
    expect(adapterRequests).toEqual([{ powerPreference: 'high-performance' }, undefined]);
    expect(caps.notes).toContain(
      'High-performance adapter selection returned null; retrying default WebGPU adapter selection.',
    );
  });

  it('falls back when a browser rejects the high-performance adapter option', async () => {
    const adapterRequests: unknown[] = [];
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          async requestAdapter(options?: { readonly powerPreference?: string }) {
            adapterRequests.push(options);
            if (options?.powerPreference === 'high-performance') {
              throw new TypeError('powerPreference is not supported');
            }
            return { info: { vendor: 'RejectedPreferenceFallback' } };
          },
        },
      },
    });

    const caps = await probeWebGpuCapabilities();

    expect(caps.available).toBe(true);
    expect(caps.provider).toBe('RejectedPreferenceFallback');
    expect(adapterRequests).toEqual([{ powerPreference: 'high-performance' }, undefined]);
    expect(caps.notes).toContain(
      'High-performance WebGPU adapter probe failed: TypeError: powerPreference is not supported; retrying default adapter selection.',
    );
  });

  it('detects WebNN from a mocked navigator', async () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        ml: {},
      },
    });

    const caps = await probeWebNnCapabilities();
    expect(caps.available).toBe(true);
    expect(caps.experimental).toBe(true);

    const context = await createWebNnBackend().createExecutionContext({
      modelFamily: 'test-family',
      modelId: 'test-model',
      precision: 'fp16',
    });
    expect(context.backendId).toBe('webnn');
    await context.dispose();
  });

  it('detects WebGL from a mocked document', async () => {
    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: {
        createElement() {
          return {
            getContext(kind: string) {
              return kind === 'webgl' ? {} : null;
            },
          };
        },
      },
    });

    const caps = await probeWebGlCapabilities();
    expect(caps.available).toBe(true);

    const backend = createWebGlBackend();
    expect(backend.id).toBe('webgl');
  });
});
