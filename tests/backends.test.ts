import { createWebGlBackend, probeWebGlCapabilities } from '@asrjs/speech-recognition';
import { probeWasmCapabilities } from '@asrjs/speech-recognition';
import { probeWebGpuCapabilities } from '@asrjs/speech-recognition';
import { probeWebNnCapabilities } from '@asrjs/speech-recognition';
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

  it('detects WebGPU capabilities from a mocked navigator', async () => {
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        gpu: {
          async requestAdapter() {
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
