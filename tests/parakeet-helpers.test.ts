import {
  ParakeetModel,
  createParakeetLocalEntries,
  DEFAULT_MODEL,
  formatResolvedQuantization,
  getLanguageName,
  getModelConfig,
  getModelKeyFromRepoId,
  getParakeetDefaultWeightSetup,
  inspectParakeetLocalEntries,
  listModels,
  loadModelWithFallback,
  loadParakeetModelFromLocalEntries,
  resolveParakeetArtifactSource,
  resolveParakeetLocalEntries,
  supportsLanguage,
  type GetParakeetModelOptions,
} from '@asrjs/speech-recognition/presets/parakeet';
import { createBuiltInSpeechRuntime } from '@asrjs/speech-recognition/builtins';
import { pickPreferredQuant } from '@asrjs/speech-recognition/io';
import { describe, expect, it, vi } from 'vitest';
import * as huggingface from '../src/runtime/huggingface.js';

describe('Parakeet helpers', () => {
  it('exposes preset model metadata without duplicating implementation families', () => {
    expect(DEFAULT_MODEL).toBe('parakeet-tdt-0.6b-v2');
    expect(listModels()).toContain('parakeet-tdt-0.6b-v3');
    expect(getModelConfig('parakeet-tdt-0.6b-v3')?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v3-onnx');
    expect(getModelConfig('parakeet-tdt-0.6b-v3')?.vocabSize).toBe(8193);
    expect(getModelKeyFromRepoId('ysdede/parakeet-tdt-0.6b-v2-onnx')).toBe('parakeet-tdt-0.6b-v2');
    expect(supportsLanguage('parakeet-tdt-0.6b-v3', 'ja')).toBe(true);
    expect(getLanguageName('zh')).toBe('Chinese');
  });

  it('picks preferred quantization based on backend and component role', () => {
    expect(pickPreferredQuant(['fp16', 'fp32', 'int8'], 'webgpu', 'encoder')).toBe('fp16');
    expect(pickPreferredQuant(['fp32', 'int8'], 'wasm', 'encoder')).toBe('int8');
    expect(pickPreferredQuant(['fp16', 'fp32', 'int8'], 'webgpu', 'decoder')).toBe('int8');
  });

  it('exposes optimized Parakeet default weights with an fp16-to-fp32 encoder fallback on webgpu', () => {
    expect(getParakeetDefaultWeightSetup('parakeet-tdt-0.6b-v3', 'webgpu-hybrid')).toEqual({
      encoderDefault: 'fp16',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['fp16', 'fp32', 'int8'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    });
    expect(getParakeetDefaultWeightSetup('parakeet-tdt-0.6b-v3', 'wasm')).toEqual({
      encoderDefault: 'int8',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['int8', 'fp32', 'fp16'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    });
  });

  it('formats resolved quantization for UI messaging', () => {
    expect(
      formatResolvedQuantization({
        encoder: 'fp16',
        decoder: 'int8',
      }),
    ).toBe('Resolved quantization: encoder=fp16, decoder=int8');
  });

  it('uses JS mel as the advertised Parakeet preset default preprocessor backend', () => {
    expect(resolveParakeetArtifactSource('parakeet-tdt-0.6b-v2')?.preprocessorBackend).toBe('js');
    expect(resolveParakeetArtifactSource('parakeet-tdt-0.6b-v3')?.preprocessorBackend).toBe('js');
  });

  it('creates a built-in runtime with registered backends and presets', () => {
    const runtime = createBuiltInSpeechRuntime();
    expect(runtime.listBackends().map((backend) => backend.id)).toEqual(
      expect.arrayContaining(['webgpu', 'wasm', 'webnn', 'webgl']),
    );
    expect(runtime.listModelFamilies().map((family) => family.family)).toEqual(
      expect.arrayContaining(['nemo-tdt', 'hf-ctc', 'whisper-seq2seq']),
    );
    expect(runtime.listPresets().map((preset) => preset.preset)).toEqual(
      expect.arrayContaining(['parakeet', 'medasr', 'whisper']),
    );
  });

  it('retries fp16 loads as fp32 when the first compile fails', async () => {
    const getParakeetModelFn = vi
      .fn<(modelId: string, options: GetParakeetModelOptions) => Promise<any>>()
      .mockResolvedValueOnce({
        urls: {
          encoderUrl: 'blob:encoder-fp16',
          decoderUrl: 'blob:decoder-int8',
          tokenizerUrl: 'blob:vocab',
        },
        filenames: {
          encoder: 'encoder-model.fp16.onnx',
          decoder: 'decoder_joint-model.int8.onnx',
        },
        quantisation: {
          encoder: 'fp16',
          decoder: 'int8',
        },
        modelConfig: getModelConfig('parakeet-tdt-0.6b-v2'),
        preprocessorBackend: 'js',
      })
      .mockResolvedValueOnce({
        urls: {
          encoderUrl: 'blob:encoder-fp32',
          decoderUrl: 'blob:decoder-int8',
          tokenizerUrl: 'blob:vocab',
        },
        filenames: {
          encoder: 'encoder-model.onnx',
          decoder: 'decoder_joint-model.int8.onnx',
        },
        quantisation: {
          encoder: 'fp32',
          decoder: 'int8',
        },
        modelConfig: getModelConfig('parakeet-tdt-0.6b-v2'),
        preprocessorBackend: 'js',
      });

    const fromUrlsFn = vi
      .fn<(config: any) => Promise<any>>()
      .mockRejectedValueOnce(new Error('fp16 compile failed'))
      .mockResolvedValueOnce({ transcribe: vi.fn() });

    const result = await loadModelWithFallback({
      repoIdOrModelKey: 'parakeet-tdt-0.6b-v2',
      options: {
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
        backend: 'webgpu-hybrid',
      },
      getParakeetModelFn,
      fromUrlsFn,
    });

    expect(result.retryUsed).toBe(true);
    expect(getParakeetModelFn).toHaveBeenCalledTimes(2);
    expect(fromUrlsFn).toHaveBeenCalledTimes(2);
    expect(fromUrlsFn.mock.calls[1]?.[0]?.filenames?.encoder).toBe('encoder-model.onnx');
  });

  it('inspects local Parakeet entries and derives local artifact choices', () => {
    const file = new File(['token-a\ntoken-b\n'], 'vocab.txt', { type: 'text/plain' });
    const entries = createParakeetLocalEntries([
      new File(['enc'], 'encoder-model.fp16.onnx'),
      new File(['dec'], 'decoder_joint-model.int8.onnx'),
      new File(['prep'], 'nemo128.onnx'),
      file,
    ]);

    const inspection = inspectParakeetLocalEntries(entries);

    expect(inspection.encoderQuantizations).toEqual(['fp16']);
    expect(inspection.decoderQuantizations).toEqual(['int8']);
    expect(inspection.tokenizerNames).toEqual(['vocab.txt']);
    expect(inspection.preprocessorNames).toEqual(['nemo128']);
  });

  it('treats vocab.txt as the canonical Parakeet tokenizer file over other local text files', () => {
    const entries = createParakeetLocalEntries([
      new File(['enc'], 'encoder-model.fp16.onnx'),
      new File(['dec'], 'decoder_joint-model.int8.onnx'),
      new File(['prep'], 'nemo128.onnx'),
      new File(['canonical'], 'vocab.txt'),
      new File(['legacy'], 'tokens.txt'),
      new File(['notes'], 'readme.txt'),
    ]);

    const inspection = inspectParakeetLocalEntries(entries);

    expect(inspection.tokenizerNames[0]).toBe('vocab.txt');
    expect(inspection.tokenizerNames).toContain('vocab.txt');
  });

  it('resolves local entries into the same artifact contract as hub loading', async () => {
    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation((blob) => `blob:${(blob as Blob).size}:${Math.random()}`);
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    try {
      const entries = createParakeetLocalEntries([
        new File(['enc'], 'encoder-model.fp16.onnx'),
        new File(['enc-data'], 'encoder-model.fp16.onnx.data'),
        new File(['dec'], 'decoder_joint-model.int8.onnx'),
        new File(['prep'], 'nemo128.onnx'),
        new File(['vocab'], 'vocab.txt'),
      ]);

      const resolved = await resolveParakeetLocalEntries(entries, {
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
        preprocessorBackend: 'js',
      });

      expect(resolved.selection).toMatchObject({
        encoderName: 'encoder-model.fp16.onnx',
        decoderName: 'decoder_joint-model.int8.onnx',
        tokenizerName: 'vocab.txt',
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
      });
      expect(resolved.config.encoderUrl.startsWith('blob:')).toBe(true);
      expect(resolved.config.decoderUrl.startsWith('blob:')).toBe(true);
      expect(resolved.config.tokenizerUrl.startsWith('blob:')).toBe(true);
      expect(resolved.config.preprocessorUrl?.startsWith('blob:')).toBe(true);
      expect(resolved.config.encoderDataUrl?.startsWith('blob:')).toBe(true);
      expect(revokeObjectURL).not.toHaveBeenCalled();
    } finally {
      createObjectURL.mockRestore();
      revokeObjectURL.mockRestore();
    }
  });

  it('routes local entry loading through the Parakeet local model wrapper', async () => {
    const fromLocalEntries = vi
      .spyOn(ParakeetModel, 'fromResolvedLocalArtifacts')
      .mockResolvedValue({ dispose: vi.fn() } as unknown as ParakeetModel);

    const createObjectURL = vi
      .spyOn(URL, 'createObjectURL')
      .mockImplementation((blob) => `blob:${(blob as Blob).size}:${Math.random()}`);

    try {
      const entries = createParakeetLocalEntries([
        new File(['enc'], 'encoder-model.onnx'),
        new File(['dec'], 'decoder_joint-model.int8.onnx'),
        new File(['prep'], 'nemo128.onnx'),
        new File(['vocab'], 'vocab.txt'),
      ]);

      const result = await loadParakeetModelFromLocalEntries(entries, {
        encoderQuant: 'fp32',
        decoderQuant: 'int8',
        preprocessorBackend: 'js',
      });

      expect(fromLocalEntries).toHaveBeenCalledTimes(1);
      expect(result.selection.encoderName).toBe('encoder-model.onnx');
      expect(result.model).toBeTruthy();
    } finally {
      fromLocalEntries.mockRestore();
      createObjectURL.mockRestore();
    }
  });

  it('still resolves the ONNX preprocessor artifact for hub loading when JS preprocessing is requested', async () => {
    const fetchModelFiles = vi
      .spyOn(huggingface, 'fetchModelFiles')
      .mockResolvedValue([
        'encoder-model.fp16.onnx',
        'decoder_joint-model.int8.onnx',
        'vocab.txt',
        'nemo128.onnx',
      ]);
    const getModelFile = vi
      .spyOn(huggingface, 'getModelFile')
      .mockImplementation(async (_repoId, filename) => `https://example.test/${filename}`);

    try {
      const { getParakeetModel } = await import('../src/presets/parakeet.js');
      const resolved = await getParakeetModel('parakeet-tdt-0.6b-v3', {
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
        preprocessorBackend: 'js',
      });

      expect(resolved.preprocessorBackend).toBe('js');
      expect(resolved.urls.preprocessorUrl).toBe('https://example.test/nemo128.onnx');
    } finally {
      fetchModelFiles.mockRestore();
      getModelFile.mockRestore();
    }
  });

  it('uses the optimized Parakeet default weights for hub loading when quantization is not specified', async () => {
    const fetchModelFiles = vi
      .spyOn(huggingface, 'fetchModelFiles')
      .mockResolvedValue([
        'encoder-model.fp16.onnx',
        'encoder-model.int8.onnx',
        'encoder-model.onnx',
        'decoder_joint-model.int8.onnx',
        'decoder_joint-model.onnx',
        'vocab.txt',
        'nemo128.onnx',
      ]);
    const getModelFile = vi
      .spyOn(huggingface, 'getModelFile')
      .mockImplementation(async (_repoId, filename) => `https://example.test/${filename}`);

    try {
      const { getParakeetModel } = await import('../src/presets/parakeet.js');
      const webgpuResolved = await getParakeetModel('parakeet-tdt-0.6b-v3', {
        backend: 'webgpu-hybrid',
      });
      const wasmResolved = await getParakeetModel('parakeet-tdt-0.6b-v3', {
        backend: 'wasm',
      });

      expect(webgpuResolved.quantisation).toEqual({
        encoder: 'fp16',
        decoder: 'int8',
      });
      expect(wasmResolved.quantisation).toEqual({
        encoder: 'int8',
        decoder: 'int8',
      });
    } finally {
      fetchModelFiles.mockRestore();
      getModelFile.mockRestore();
    }
  });
});
