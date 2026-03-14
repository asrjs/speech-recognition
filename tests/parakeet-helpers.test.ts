import {
  createBuiltInSpeechRuntime,
  DEFAULT_MODEL,
  formatResolvedQuantization,
  getLanguageName,
  getModelConfig,
  getModelKeyFromRepoId,
  listModels,
  loadModelWithFallback,
  pickPreferredQuant,
  supportsLanguage,
  type GetParakeetModelOptions
} from 'asr.js';
import { describe, expect, it, vi } from 'vitest';

describe('Parakeet helpers', () => {
  it('exposes preset model metadata without duplicating implementation families', () => {
    expect(DEFAULT_MODEL).toBe('parakeet-tdt-0.6b-v2');
    expect(listModels()).toContain('parakeet-tdt-0.6b-v3');
    expect(getModelConfig('parakeet-tdt-0.6b-v3')?.repoId).toBe('ysdede/parakeet-tdt-0.6b-v3-onnx');
    expect(getModelKeyFromRepoId('ysdede/parakeet-tdt-0.6b-v2-onnx')).toBe('parakeet-tdt-0.6b-v2');
    expect(supportsLanguage('parakeet-tdt-0.6b-v3', 'ja')).toBe(true);
    expect(getLanguageName('zh')).toBe('Chinese');
  });

  it('picks preferred quantization based on backend and component role', () => {
    expect(pickPreferredQuant(['fp16', 'fp32', 'int8'], 'webgpu', 'encoder')).toBe('fp16');
    expect(pickPreferredQuant(['fp32', 'int8'], 'wasm', 'encoder')).toBe('int8');
    expect(pickPreferredQuant(['fp16', 'fp32', 'int8'], 'webgpu', 'decoder')).toBe('int8');
  });

  it('formats resolved quantization for UI messaging', () => {
    expect(formatResolvedQuantization({
      encoder: 'fp16',
      decoder: 'int8'
    })).toBe('Resolved quantization: encoder=fp16, decoder=int8');
  });

  it('creates a built-in runtime with registered backends and presets', () => {
    const runtime = createBuiltInSpeechRuntime();
    expect(runtime.listBackends().map((backend) => backend.id)).toEqual(
      expect.arrayContaining(['webgpu', 'wasm', 'webnn', 'webgl'])
    );
    expect(runtime.listModelFamilies().map((family) => family.family)).toEqual(
      expect.arrayContaining(['parakeet', 'medasr', 'whisper'])
    );
  });

  it('retries fp16 loads as fp32 when the first compile fails', async () => {
    const getParakeetModelFn = vi
      .fn<(modelId: string, options: GetParakeetModelOptions) => Promise<any>>()
      .mockResolvedValueOnce({
        urls: {
          encoderUrl: 'blob:encoder-fp16',
          decoderUrl: 'blob:decoder-int8',
          tokenizerUrl: 'blob:vocab'
        },
        filenames: {
          encoder: 'encoder-model.fp16.onnx',
          decoder: 'decoder_joint-model.int8.onnx'
        },
        quantisation: {
          encoder: 'fp16',
          decoder: 'int8'
        },
        modelConfig: getModelConfig('parakeet-tdt-0.6b-v2'),
        preprocessorBackend: 'js'
      })
      .mockResolvedValueOnce({
        urls: {
          encoderUrl: 'blob:encoder-fp32',
          decoderUrl: 'blob:decoder-int8',
          tokenizerUrl: 'blob:vocab'
        },
        filenames: {
          encoder: 'encoder-model.onnx',
          decoder: 'decoder_joint-model.int8.onnx'
        },
        quantisation: {
          encoder: 'fp32',
          decoder: 'int8'
        },
        modelConfig: getModelConfig('parakeet-tdt-0.6b-v2'),
        preprocessorBackend: 'js'
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
        backend: 'webgpu-hybrid'
      },
      getParakeetModelFn,
      fromUrlsFn
    });

    expect(result.retryUsed).toBe(true);
    expect(getParakeetModelFn).toHaveBeenCalledTimes(2);
    expect(fromUrlsFn).toHaveBeenCalledTimes(2);
    expect(fromUrlsFn.mock.calls[1]?.[0]?.filenames?.encoder).toBe('encoder-model.onnx');
  });
});
