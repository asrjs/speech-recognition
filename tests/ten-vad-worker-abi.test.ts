import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('TEN-VAD worker C ABI lifecycle', () => {
  const responses: any[] = [];
  const heap = new Int32Array(4096);
  let pointer = 0;
  const module = {
    HEAP32: heap,
    _malloc: vi.fn(() => (pointer += 4)),
    _free: vi.fn(),
    _ten_vad_create: vi.fn((ptr: number) => {
      heap[ptr >> 2] = 8192;
      return 0;
    }),
    _ten_vad_destroy: vi.fn((ptr: number) => {
      heap[ptr >> 2] = 0;
      return 0;
    }),
    _ten_vad_get_version: () => 0,
  };
  const scope = { onmessage: null as any, postMessage: (message: any) => responses.push(message) };

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    pointer = 0;
    responses.length = 0;
    heap.fill(0);
    module._ten_vad_destroy.mockImplementation((ptr: number) => {
      heap[ptr >> 2] = 0;
      return 0;
    });
    vi.stubGlobal('self', scope);
    vi.stubGlobal('__tenVadAbiModule', module);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, text: async () => '' })),
    );
    vi.spyOn(URL, 'createObjectURL').mockReturnValue(
      'data:text/javascript,export default ()=>Promise.resolve(globalThis.__tenVadAbiModule)',
    );
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    await import('../src/runtime/ten-vad-worker.js');
    await scope.onmessage({
      data: {
        type: 'INIT',
        id: 1,
        payload: {
          hopSize: 256,
          threshold: 0.5,
          scriptUrl: '/test.js',
          wasmUrl: '/test.wasm',
          fallbackScriptUrl: null,
          fallbackWasmUrl: null,
        },
      },
    });
    expect(responses[0].type).toBe('INIT');
    responses.length = 0;
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  for (const type of ['RESET', 'DISPOSE']) {
    it(`${type} passes the handle pointer, not the handle value`, async () => {
      await scope.onmessage({ data: { type, id: 2, payload: null } });
      expect(module._ten_vad_destroy).toHaveBeenCalledWith(4);
      expect(responses[0].type).toBe(type);
    });
    it(`${type} cannot acknowledge a failed native destruction`, async () => {
      module._ten_vad_destroy.mockReturnValue(-1);
      await scope.onmessage({ data: { type, id: 2, payload: null } });
      expect(responses).toEqual([
        { type: 'ERROR', id: 2, payload: 'ten_vad_destroy failed with code -1' },
      ]);
      expect(module._free).not.toHaveBeenCalled();
      expect(module._ten_vad_create).toHaveBeenCalledTimes(1);
    });
  }
});
