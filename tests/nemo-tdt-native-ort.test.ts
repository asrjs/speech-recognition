import { describe, expect, it, vi } from 'vitest';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import {
  createOrtSession,
  initOrt,
  resolveNemoTdtArtifacts,
  type OrtModuleLike,
} from '../src/models/nemo-tdt/ort.js';

const native = vi.hoisted(() => ({
  env: { wasm: {} },
  InferenceSession: { create: vi.fn() },
  Tensor: class {},
}));
vi.mock('../src/inference/backends/native-cpu/ort.js', () => ({ loadNativeCpuOrt: () => native }));

const artifacts = {
  encoderUrl: 'file:///encoder.onnx',
  decoderUrl: 'file:///decoder.onnx',
  tokenizerUrl: 'file:///vocab.txt',
};

describe('NeMo TDT explicit native CPU path', () => {
  it('selects the native engine without configuring the WASM environment', async () => {
    expect(await initOrt('native-cpu', { cpuThreads: 3 })).toBe(native);
    expect(native.env.wasm).toEqual({});
  });

  it('resolves both native components and fp32 weights without integer defaults', () => {
    const result = resolveNemoTdtArtifacts(
      { kind: 'huggingface', repoId: 'test/weights' },
      'native-cpu',
    );
    expect(result.ortBackend).toBe('native-cpu');
    expect(result.encoderBackendForOrt).toBe('native-cpu');
    expect(result.decoderBackendForOrt).toBe('native-cpu');
    expect(result.preprocessorBackend).toBe('js');
    expect(result.artifacts.encoderFilename).toBe('encoder-model.onnx');
    expect(result.artifacts.decoderFilename).toBe('decoder_joint-model.onnx');
  });

  it('keeps the legacy browser WebGPU encoder/WASM decoder composition', () => {
    const result = resolveNemoTdtArtifacts({ kind: 'direct', artifacts }, 'webgpu');
    expect(result.encoderBackendForOrt).toBe('webgpu');
    expect(result.decoderBackendForOrt).toBe('wasm');
    expect(result.ortBackend).toBe('webgpu');
  });

  it.each(['wasm', 'webgpu'] as const)(
    'rejects an explicit %s component in a native load',
    (placement) => {
      expect(() =>
        resolveNemoTdtArtifacts(
          {
            kind: 'direct',
            artifacts,
            encoderBackend: placement,
          },
          'native-cpu',
        ),
      ).toThrow('native-cpu');
    },
  );

  it('rejects a native component while the reported model backend is WASM', () => {
    expect(() =>
      resolveNemoTdtArtifacts(
        {
          kind: 'direct',
          artifacts,
          encoderBackend: 'native-cpu',
        },
        'wasm',
      ),
    ).toThrow('native-cpu');
  });

  it('passes CPU provider and per-session thread settings, not WebGPU options', async () => {
    const create = vi.fn(async (_url: string, _options?: Record<string, unknown>) => ({
      run: vi.fn(),
      release: vi.fn(),
    }));
    const ort = { ...native, InferenceSession: { create } } as unknown as OrtModuleLike;
    const file = resolve('tests/fixtures/example.onnx');
    await createOrtSession(ort, pathToFileURL(file).href, {
      backendId: 'native-cpu',
      cpuThreads: 3,
    });
    expect(create).toHaveBeenCalledWith(
      file,
      expect.objectContaining({
        executionProviders: ['cpu'],
        intraOpNumThreads: 3,
        interOpNumThreads: 1,
      }),
    );
    expect(create.mock.calls[0]?.[1]).not.toHaveProperty('preferredOutputLocation');
  });

  it('rejects a remote native graph before ORT can silently choose another engine', async () => {
    const create = vi.fn();
    const ort = { ...native, InferenceSession: { create } } as unknown as OrtModuleLike;
    await expect(
      createOrtSession(ort, 'https://example.test/model.onnx', {
        backendId: 'native-cpu',
      }),
    ).rejects.toThrow('local');
    expect(create).not.toHaveBeenCalled();
  });

  it('rejects a native sidecar locator that differs from its graph-relative location', async () => {
    const create = vi.fn();
    const ort = { ...native, InferenceSession: { create } } as unknown as OrtModuleLike;
    await expect(
      createOrtSession(ort, pathToFileURL(resolve('model/model.onnx')).href, {
        backendId: 'native-cpu',
        externalDataUrl: pathToFileURL(resolve('elsewhere/weights.data')).href,
        externalDataPath: 'weights.data',
      }),
    ).rejects.toThrow('colocated');
    expect(create).not.toHaveBeenCalled();
  });

  it('fails before creating a session for invalid native thread settings', async () => {
    const create = vi.fn();
    const ort = { ...native, InferenceSession: { create } } as unknown as OrtModuleLike;
    await expect(
      createOrtSession(ort, resolve('model.onnx'), {
        backendId: 'native-cpu',
        cpuThreads: 0,
      }),
    ).rejects.toThrow('positive integer');
    expect(create).not.toHaveBeenCalled();
  });
});
