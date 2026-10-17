import { describe, expect, it, vi } from 'vitest';
import {
  configureUnifiedOrtEnv,
  createUnifiedOrtSession,
  releaseUnifiedOrtSession,
  resolveUnifiedExecutionProviders,
  resolveUnifiedWasmPaths,
  type OrtModuleLike,
  type OrtSessionLike,
} from '../src/inference/ort/index.js';
import { AssetLoadAbortedError } from '../src/io/abort.js';

function createMockOrt(
  createFn?: (source: string | Uint8Array, options?: Record<string, unknown>) => Promise<OrtSessionLike>,
): OrtModuleLike {
  return {
    env: {
      wasm: {},
    },
    InferenceSession: {
      create: createFn ?? vi.fn().mockResolvedValue({
        run: vi.fn(),
        release: vi.fn().mockResolvedValue(undefined),
      }),
    },
  };
}

describe('Unified ORT Runtime Layer', () => {
  describe('resolveUnifiedWasmPaths', () => {
    it('returns custom wasmPaths verbatim when provided', () => {
      const custom = 'https://assets.local/ort-wasm/';
      expect(resolveUnifiedWasmPaths(custom)).toBe(custom);
    });

    it('falls back to CDN dist URL in Node when no custom path is given', () => {
      const pathDefault = resolveUnifiedWasmPaths();
      expect(pathDefault).toContain('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/');

      const pathCustomVersion = resolveUnifiedWasmPaths(undefined, '1.24.1');
      expect(pathCustomVersion).toContain('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/');
    });
  });

  describe('configureUnifiedOrtEnv', () => {
    it('configures simd, proxy, numThreads, and logSeverityLevel', () => {
      const mockOrt = createMockOrt();
      configureUnifiedOrtEnv(mockOrt, {
        cpuThreads: 4,
        logSeverityLevel: 3,
        proxy: false,
      });

      expect(mockOrt.env.wasm.simd).toBe(true);
      expect(mockOrt.env.wasm.proxy).toBe(false);
      expect(mockOrt.env.logSeverityLevel).toBe(3);
      expect(typeof mockOrt.env.wasm.numThreads).toBe('number');
      expect(mockOrt.env.wasm.numThreads).toBeGreaterThanOrEqual(1);
      expect(mockOrt.env.wasm.wasmPaths).toBeDefined();
    });

    it('does not clobber existing wasmPaths unless explicitly requested', () => {
      const mockOrt = createMockOrt();
      mockOrt.env.wasm.wasmPaths = '/existing/dist/';

      configureUnifiedOrtEnv(mockOrt);
      expect(mockOrt.env.wasm.wasmPaths).toBe('/existing/dist/');

      configureUnifiedOrtEnv(mockOrt, { wasmPaths: '/new/dist/' });
      expect(mockOrt.env.wasm.wasmPaths).toBe('/new/dist/');
    });

    it('sets numThreads = 1 when SharedArrayBuffer is unavailable', () => {
      const originalSab = globalThis.SharedArrayBuffer;
      try {
        // @ts-expect-error simulating environment without SharedArrayBuffer
        delete globalThis.SharedArrayBuffer;

        const mockOrt = createMockOrt();
        configureUnifiedOrtEnv(mockOrt, { cpuThreads: 4 });
        expect(mockOrt.env.wasm.numThreads).toBe(1);
      } finally {
        globalThis.SharedArrayBuffer = originalSab;
      }
    });

    it('configures webgpu profiling when requested', () => {
      const mockOrt = createMockOrt();
      configureUnifiedOrtEnv(mockOrt, { enableProfiling: true });
      expect(mockOrt.env.webgpu?.profiling).toEqual({ mode: 'default' });
    });
  });

  describe('resolveUnifiedExecutionProviders', () => {
    it('resolves for native-cpu to [\"cpu\"]', () => {
      expect(resolveUnifiedExecutionProviders('native-cpu')).toEqual(['cpu']);
    });

    it('resolves for wasm to [\"wasm\"]', () => {
      expect(resolveUnifiedExecutionProviders('wasm')).toEqual(['wasm']);
    });

    it('resolves for webgpu with powerPreference and deviceType defaults', () => {
      const eps = resolveUnifiedExecutionProviders('webgpu');
      expect(eps).toEqual([
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
      ]);
    });

    it('honors custom deviceType, powerPreference, and allowWasmFallback', () => {
      const eps = resolveUnifiedExecutionProviders('webgpu-custom', {
        deviceType: 'cpu',
        powerPreference: 'low-power',
        allowWasmFallback: true,
      });
      expect(eps).toEqual([
        {
          name: 'webgpu',
          deviceType: 'cpu',
          powerPreference: 'low-power',
        },
        'wasm',
      ]);
    });
  });

  describe('createUnifiedOrtSession', () => {
    it('creates primary WebGPU session successfully without fallback', async () => {
      const releaseFn = vi.fn().mockResolvedValue(undefined);
      const mockSession: OrtSessionLike = {
        run: vi.fn(),
        release: releaseFn,
      };
      const createFn = vi.fn().mockResolvedValue(mockSession);
      const mockOrt = createMockOrt(createFn);

      const result = await createUnifiedOrtSession(mockOrt, 'model.onnx', {
        backendId: 'webgpu',
        allowWasmFallback: true,
      });

      expect(result.session).toBe(mockSession);
      expect(result.resolvedBackendId).toBe('webgpu');
      expect(result.fellBackToWasm).toBe(false);
      expect(createFn).toHaveBeenCalledTimes(1);

      const calledOptions = createFn.mock.calls[0]![1] as Record<string, unknown>;
      expect(calledOptions.executionProviders).toEqual([
        {
          name: 'webgpu',
          deviceType: 'gpu',
          powerPreference: 'high-performance',
        },
      ]);
    });

    it('falls back to WASM when WebGPU create fails and allowWasmFallback: true', async () => {
      const releaseFn = vi.fn().mockResolvedValue(undefined);
      const mockSession: OrtSessionLike = {
        run: vi.fn(),
        release: releaseFn,
      };

      const createFn = vi.fn()
        .mockRejectedValueOnce(new Error('WebGPU adapter not supported'))
        .mockResolvedValueOnce(mockSession);

      const mockOrt = createMockOrt(createFn);

      const result = await createUnifiedOrtSession(mockOrt, 'model.onnx', {
        backendId: 'webgpu',
        allowWasmFallback: true,
      });

      expect(result.session).toBe(mockSession);
      expect(result.resolvedBackendId).toBe('wasm');
      expect(result.fellBackToWasm).toBe(true);
      expect(createFn).toHaveBeenCalledTimes(2);

      const fallbackOptions = createFn.mock.calls[1]![1] as Record<string, unknown>;
      expect(fallbackOptions.executionProviders).toEqual(['wasm']);
    });

    it('throws error when WebGPU create fails and allowWasmFallback is false', async () => {
      const createFn = vi.fn().mockRejectedValue(new Error('WebGPU unavailable'));
      const mockOrt = createMockOrt(createFn);

      await expect(
        createUnifiedOrtSession(mockOrt, 'model.onnx', {
          backendId: 'webgpu',
          allowWasmFallback: false,
        }),
      ).rejects.toThrow('WebGPU unavailable');

      expect(createFn).toHaveBeenCalledTimes(1);
    });

    it('supports Uint8Array in-memory model source', async () => {
      const mockSession: OrtSessionLike = { run: vi.fn() };
      const createFn = vi.fn().mockResolvedValue(mockSession);
      const mockOrt = createMockOrt(createFn);

      const buffer = new Uint8Array([8, 7, 6, 5]);
      const result = await createUnifiedOrtSession(mockOrt, buffer, { backendId: 'wasm' });

      expect(result.session).toBe(mockSession);
      expect(result.resolvedBackendId).toBe('wasm');
      expect(createFn).toHaveBeenCalledWith(buffer, expect.any(Object));
    });

    it('mounts external data files into session options', async () => {
      const mockSession: OrtSessionLike = { run: vi.fn() };
      const createFn = vi.fn().mockResolvedValue(mockSession);
      const mockOrt = createMockOrt(createFn);

      await createUnifiedOrtSession(mockOrt, 'model.onnx', {
        backendId: 'wasm',
        externalData: [{ dataUrl: 'http://localhost/model.onnx.data', path: 'model.onnx.data' }],
      });

      const calledOptions = createFn.mock.calls[0]![1] as Record<string, unknown>;
      expect(calledOptions.externalData).toEqual([
        { data: 'http://localhost/model.onnx.data', path: 'model.onnx.data' },
      ]);
    });

    it('aborts early if signal is already aborted', async () => {
      const mockOrt = createMockOrt();
      const abortSignal = { aborted: true };

      await expect(
        createUnifiedOrtSession(mockOrt, 'model.onnx', {
          backendId: 'wasm',
          signal: abortSignal,
        }),
      ).rejects.toThrow(AssetLoadAbortedError);
    });

    it('cleans up and releases session if aborted during session creation', async () => {
      const releaseFn = vi.fn().mockResolvedValue(undefined);
      const abortSignal = { aborted: false };

      const mockOrt = createMockOrt(async () => {
        abortSignal.aborted = true; // aborted concurrently
        return {
          run: vi.fn(),
          release: releaseFn,
        };
      });

      await expect(
        createUnifiedOrtSession(mockOrt, 'model.onnx', {
          backendId: 'wasm',
          signal: abortSignal,
        }),
      ).rejects.toThrow(AssetLoadAbortedError);

      expect(releaseFn).toHaveBeenCalled();
    });
  });

  describe('releaseUnifiedOrtSession', () => {
    it('safely handles null and undefined without throwing', async () => {
      await expect(releaseUnifiedOrtSession(null)).resolves.toBeUndefined();
      await expect(releaseUnifiedOrtSession(undefined)).resolves.toBeUndefined();
    });

    it('calls release on session and awaits completion', async () => {
      const releaseFn = vi.fn().mockResolvedValue(undefined);
      const session: OrtSessionLike = {
        run: vi.fn(),
        release: releaseFn,
      };

      await releaseUnifiedOrtSession(session);
      expect(releaseFn).toHaveBeenCalledTimes(1);
    });

    it('suppresses release errors during teardown', async () => {
      const session: OrtSessionLike = {
        run: vi.fn(),
        release: vi.fn().mockRejectedValue(new Error('GPU context lost')),
      };

      await expect(releaseUnifiedOrtSession(session)).resolves.toBeUndefined();
    });
  });
});
