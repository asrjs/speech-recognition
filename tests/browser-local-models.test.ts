import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  createSpeechModelLocalEntries,
  inspectSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '@asrjs/speech-recognition/browser';
import { loadBuiltInSpeechModel } from '../src/runtime/builtins.js';
import { resolveParakeetLocalEntries } from '../src/presets/parakeet/compat.js';

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
    vi.mocked(resolveParakeetLocalEntries).mockClear();
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
        backend: 'webgpu-hybrid',
        options: {
          source: expect.objectContaining({
            kind: 'direct',
            encoderBackend: 'webgpu',
            decoderBackend: 'wasm',
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

    await loaded.dispose();
    await loaded.dispose();

    expect(loadedHandleDispose).toHaveBeenCalledTimes(1);
    expect(assetHandleDispose).toHaveBeenCalledTimes(1);
  });
});
