import { afterEach, describe, expect, it, vi } from 'vitest';
import { resolveWasmThreads } from '../src/inference/backends/wasm-threads.js';

afterEach(() => vi.unstubAllGlobals());

describe('WASM hardware thread cap', () => {
  it.each([
    [1, 2],
    [2, 3],
    [4, 4],
    [8, 6],
    [16, 10],
    [32, 18],
    [7, 5],
  ])('%i hardware threads limits WASM to %i', (hardwareConcurrency, limit) => {
    vi.stubGlobal('navigator', { hardwareConcurrency });
    expect(resolveWasmThreads()).toBe(limit);
    expect(resolveWasmThreads(100)).toBe(limit);
    expect(resolveWasmThreads(1)).toBe(1);
    expect(resolveWasmThreads(undefined, 1)).toBe(1);
  });
  it.each([undefined, NaN, Infinity, 0, -1])(
    'uses safe fallback for invalid count %s',
    (hardwareConcurrency) => {
      vi.stubGlobal('navigator', { hardwareConcurrency });
      expect(resolveWasmThreads()).toBe(4);
      expect(resolveWasmThreads(100)).toBe(4);
    },
  );
  it('does not require a browser and preserves smaller requests', () => {
    vi.stubGlobal('navigator', undefined);
    expect(resolveWasmThreads(2)).toBe(2);
    expect(resolveWasmThreads(2.9)).toBe(2);
    expect(resolveWasmThreads(0)).toBe(1);
  });
});
