import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  collectSpeechModelLocalEntries,
  createSpeechModelLocalEntries,
  inspectSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '@asrjs/speech-recognition/browser';
import { loadBuiltInSpeechModel } from '../src/runtime/builtins.js';
import { resolveParakeetLocalEntries } from '../src/presets/parakeet/compat.js';
import { PipelineAbortedError } from '../src/pipeline/composition.js';

vi.mock('../src/runtime/builtins.js', () => ({
  loadBuiltInSpeechModel: vi.fn(),
}));

vi.mock('../src/presets/parakeet/compat.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/presets/parakeet/compat.js')>();
  return {
    ...actual,
    resolveParakeetLocalEntries: vi.fn(actual.resolveParakeetLocalEntries),
  };
});

describe('browser local model helpers', () => {
  beforeEach(() => {
    vi.mocked(loadBuiltInSpeechModel).mockReset();
    vi.mocked(resolveParakeetLocalEntries).mockReset();
  });

  it('creates and inspects built-in local speech model entries', () => {
    const entries = createSpeechModelLocalEntries([
      new File(['enc'], 'encoder-model.fp16.onnx'),
      new File(['dec'], 'decoder_joint-model.int8.onnx'),
      new File(['prep'], 'nemo128.onnx'),
      new File(['vocab'], 'vocab.txt'),
    ]);

    const inspection = inspectSpeechModelLocalEntries('parakeet-tdt-0.6b-v2', entries);

    expect(inspection.encoderQuantizations).toEqual(['fp16']);
    expect(inspection.decoderQuantizations).toEqual(['int8']);
    expect(inspection.tokenizerNames).toEqual(['vocab.txt']);
    expect(inspection.preprocessorNames).toEqual(['nemo128']);
  });

  it('rejects local inspection for built-in models without local-folder support', () => {
    expect(() =>
      inspectSpeechModelLocalEntries('google/medasr', createSpeechModelLocalEntries([])),
    ).toThrow('does not support local folder loading');
  });

  it('loads built-in speech models from local entries and disposes owned asset handles', async () => {
    const assetHandleDispose = vi.fn(async () => undefined);
    const loadedHandleDispose = vi.fn(async () => undefined);
    const loadedTranscribe = vi.fn(async () => ({ text: 'hello' }));

    vi.mocked(resolveParakeetLocalEntries).mockResolvedValue({
      config: {
        modelId: 'parakeet-tdt-0.6b-v2',
        encoderBackend: 'webgpu',
        decoderBackend: 'wasm',
        decoderStateOutputLocation: 'gpu-buffer',
        webgpuOptions: { storageBufferCacheMode: 'simple' },
        encoderUrl: 'blob:encoder',
        decoderUrl: 'blob:decoder',
        tokenizerUrl: 'blob:vocab',
        preprocessorBackend: 'js',
        backend: 'webgpu-hybrid',
      },
      assetHandles: [
        {
          dispose: assetHandleDispose,
        },
      ] as any,
      selection: {
        encoderName: 'encoder-model.fp16.onnx',
        decoderName: 'decoder_joint-model.int8.onnx',
        tokenizerName: 'vocab.txt',
        preprocessorName: undefined,
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
      },
    });

    vi.mocked(loadBuiltInSpeechModel).mockResolvedValue({
      runtime: {} as any,
      model: {
        info: {
          modelId: 'parakeet-tdt-0.6b-v2',
        },
      } as any,
      session: {} as any,
      transcribe: loadedTranscribe,
      dispose: loadedHandleDispose,
    });

    const loaded = await loadSpeechModelFromLocalEntries({
      modelId: 'parakeet-tdt-0.6b-v2',
      entries: [],
      backend: 'webgpu-hybrid',
      encoderQuant: 'fp16',
      decoderQuant: 'int8',
      decoderStateOutputLocation: 'gpu-buffer',
      webgpuOptions: { storageBufferCacheMode: 'simple' },
      tokenizerName: 'vocab.txt',
      preprocessorBackend: 'js',
    });

    expect(loaded.selection).toMatchObject({
      encoderName: 'encoder-model.fp16.onnx',
      decoderName: 'decoder_joint-model.int8.onnx',
      tokenizerName: 'vocab.txt',
      encoderQuant: 'fp16',
      decoderQuant: 'int8',
    });
    expect(loadBuiltInSpeechModel).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: 'parakeet-tdt-0.6b-v2',
        preset: 'parakeet',
        backend: 'webgpu',
        options: {
          source: expect.objectContaining({
            kind: 'direct',
            encoderBackend: 'webgpu',
            decoderBackend: 'wasm',
            decoderStateOutputLocation: 'gpu-buffer',
            webgpuOptions: { storageBufferCacheMode: 'simple' },
            artifacts: expect.objectContaining({
              encoderUrl: 'blob:encoder',
              decoderUrl: 'blob:decoder',
              tokenizerUrl: 'blob:vocab',
            }),
            preprocessorBackend: 'js',
          }),
        },
      }),
    );

    const transcript = await loaded.transcribeMonoPcm(new Float32Array([0, 0.25, -0.25]), 16_000, {
      responseFlavor: 'canonical',
    });

    await loaded.dispose();
    await loaded.dispose();

    expect(transcript).toEqual({ text: 'hello' });
    expect(loadedTranscribe).toHaveBeenCalledTimes(1);
    expect(loadedHandleDispose).toHaveBeenCalledTimes(1);
    expect(assetHandleDispose).toHaveBeenCalledTimes(1);
  });

  it('disposes resolved local artifact handles when loading fails', async () => {
    const assetHandleDispose = vi.fn(async () => undefined);

    vi.mocked(resolveParakeetLocalEntries).mockResolvedValue({
      config: {
        modelId: 'parakeet-tdt-0.6b-v2',
        encoderBackend: 'webgpu',
        decoderBackend: 'wasm',
        encoderUrl: 'blob:encoder',
        decoderUrl: 'blob:decoder',
        tokenizerUrl: 'blob:vocab',
        preprocessorBackend: 'js',
        backend: 'webgpu',
      },
      assetHandles: [
        {
          dispose: assetHandleDispose,
        },
      ] as any,
      selection: {
        encoderName: 'encoder-model.fp16.onnx',
        decoderName: 'decoder_joint-model.int8.onnx',
        tokenizerName: 'vocab.txt',
        preprocessorName: undefined,
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
      },
    });
    vi.mocked(loadBuiltInSpeechModel).mockRejectedValue(new Error('Load failed'));

    await expect(
      loadSpeechModelFromLocalEntries({
        modelId: 'parakeet-tdt-0.6b-v2',
        entries: [],
        backend: 'webgpu-hybrid',
      }),
    ).rejects.toThrow('Load failed');

    expect(assetHandleDispose).toHaveBeenCalledTimes(1);
  });

  it('honors options.signal before resolving local entries', async () => {
    const progressEvents: Array<{ phase: string; aborted?: boolean }> = [];

    await expect(
      loadSpeechModelFromLocalEntries({
        modelId: 'parakeet-tdt-0.6b-v2',
        entries: [],
        signal: { aborted: true },
        onProgress(event) {
          progressEvents.push({ phase: event.phase, aborted: event.aborted });
        },
      }),
    ).rejects.toMatchObject({
      name: 'PipelineAbortedError',
    });

    expect(resolveParakeetLocalEntries).not.toHaveBeenCalled();
    expect(loadBuiltInSpeechModel).not.toHaveBeenCalled();
    expect(progressEvents).toEqual([{ phase: 'cancelled', aborted: true }]);
  });

  it('passes signal into session create and disposes local handles on abort', async () => {
    const assetHandleDispose = vi.fn(async () => undefined);
    const signal = { aborted: false };

    vi.mocked(resolveParakeetLocalEntries).mockResolvedValue({
      config: {
        modelId: 'parakeet-tdt-0.6b-v2',
        encoderBackend: 'webgpu',
        decoderBackend: 'wasm',
        encoderUrl: 'blob:encoder',
        decoderUrl: 'blob:decoder',
        tokenizerUrl: 'blob:vocab',
        preprocessorBackend: 'js',
        backend: 'webgpu',
      },
      assetHandles: [
        {
          dispose: assetHandleDispose,
        },
      ] as any,
      selection: {
        encoderName: 'encoder-model.fp16.onnx',
        decoderName: 'decoder_joint-model.int8.onnx',
        tokenizerName: 'vocab.txt',
        preprocessorName: undefined,
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
      },
    });
    vi.mocked(loadBuiltInSpeechModel).mockImplementation(async (loadOptions) => {
      expect(loadOptions.signal).toBe(signal);
      signal.aborted = true;
      throw new PipelineAbortedError('load');
    });

    await expect(
      loadSpeechModelFromLocalEntries({
        modelId: 'parakeet-tdt-0.6b-v2',
        entries: [],
        backend: 'webgpu',
        signal,
      }),
    ).rejects.toBeInstanceOf(PipelineAbortedError);

    expect(assetHandleDispose).toHaveBeenCalledTimes(1);
  });

  it('aborts local file reads and disposes locators created before abort', async () => {
    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation((blob) => `blob:${(blob as Blob).size}:${Math.random()}`);
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const controller = new AbortController();

    try {
      const decoderFile = new File(['dec'], 'decoder_joint-model.int8.onnx');
      const entries = [
        {
          path: 'encoder-model.fp16.onnx',
          basename: 'encoder-model.fp16.onnx',
          file: new File(['enc'], 'encoder-model.fp16.onnx'),
        },
        {
          path: 'decoder_joint-model.int8.onnx',
          basename: 'decoder_joint-model.int8.onnx',
          handle: {
            kind: 'file' as const,
            async getFile() {
              controller.abort();
              return decoderFile;
            },
          },
        },
        {
          path: 'vocab.txt',
          basename: 'vocab.txt',
          file: new File(['vocab'], 'vocab.txt'),
        },
      ];

      await expect(
        loadSpeechModelFromLocalEntries({
          modelId: 'parakeet-tdt-0.6b-v2',
          entries,
          encoderQuant: 'fp16',
          decoderQuant: 'int8',
          preprocessorBackend: 'js',
          signal: controller.signal,
        }),
      ).rejects.toBeInstanceOf(PipelineAbortedError);

      expect(createObjectURL).toHaveBeenCalled();
      expect(revokeObjectURL).toHaveBeenCalled();
      expect(loadBuiltInSpeechModel).not.toHaveBeenCalled();
    } finally {
      createObjectURL.mockRestore();
      revokeObjectURL.mockRestore();
    }
  });

  it('aborts collectSpeechModelLocalEntries during a large folder walk', async () => {
    const signal = { aborted: false };
    let nestedWalked = false;
    const nested = {
      kind: 'directory' as const,
      name: 'weights',
      async *entries() {
        nestedWalked = true;
        yield [
          'nested.bin',
          { kind: 'file' as const, getFile: async () => new File(['x'], 'nested.bin') },
        ];
      },
    };
    const root = {
      kind: 'directory' as const,
      async *entries() {
        yield [
          'encoder-model.onnx',
          { kind: 'file' as const, getFile: async () => new File(['e'], 'encoder-model.onnx') },
        ];
        signal.aborted = true;
        yield ['weights', nested];
      },
    };

    await expect(collectSpeechModelLocalEntries(root, '', signal)).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(nestedWalked).toBe(false);
  });
});
