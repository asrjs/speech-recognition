import {
  CanaryModel,
  DEFAULT_MODEL,
  getLanguageName,
  getModelConfig,
  getModelKeyFromRepoId,
  listModels,
  transcribeCanary,
} from '@asrjs/speech-recognition/presets/canary';
import { describe, expect, it, vi } from 'vitest';
import type { DefaultSpeechRuntime } from '../src/runtime/session.js';
import * as huggingface from '../src/runtime/huggingface.js';

describe('Canary helpers', () => {
  it('exposes preset model metadata and language information', () => {
    expect(DEFAULT_MODEL).toBe('nvidia/canary-180m-flash');
    expect(listModels()).toContain('nvidia/canary-180m-flash');
    expect(getModelConfig('nvidia/canary-180m-flash')?.repoId).toBe(
      'ysdede/canary-180m-flash-onnx',
    );
    expect(getModelConfig('nvidia/canary-180m-flash')?.featuresSize).toBe(128);
    expect(getModelKeyFromRepoId('ysdede/canary-180m-flash-onnx')).toBe('nvidia/canary-180m-flash');
    expect(getLanguageName('fr')).toBe('French');
  });

  it('does not resolve an ONNX preprocessor artifact when JS preprocessing is requested', async () => {
    const fetchModelFiles = vi
      .spyOn(huggingface, 'fetchModelFiles')
      .mockResolvedValue([
        'encoder-model.fp16.onnx',
        'encoder-model.fp16.onnx.data',
        'decoder-model.int8.onnx',
        'tokenizer.json',
        'config.json',
        'nemo128.onnx',
      ]);
    const getModelFile = vi
      .spyOn(huggingface, 'getModelFile')
      .mockImplementation(async (_repoId, filename) => `https://example.test/${filename}`);

    try {
      const { getCanaryModel: getCanaryModelFromSource } = await import('../src/presets/canary.js');
      const resolved = await getCanaryModelFromSource('nvidia/canary-180m-flash', {
        encoderQuant: 'fp16',
        decoderQuant: 'int8',
        preprocessorBackend: 'js',
        backend: 'wasm',
      });

      expect(resolved.preprocessorBackend).toBe('js');
      expect(resolved.urls.preprocessorUrl).toBeUndefined();
      expect(resolved.urls.encoderDataUrl).toBe(
        'https://example.test/encoder-model.fp16.onnx.data',
      );
      expect(resolved.filenames.encoder).toBe('encoder-model.fp16.onnx');
      expect(resolved.filenames.decoder).toBe('decoder-model.int8.onnx');
    } finally {
      fetchModelFiles.mockRestore();
      getModelFile.mockRestore();
    }
  });

  it('fails fast when ONNX preprocessing is requested without a nemo128 artifact', async () => {
    const fetchModelFiles = vi
      .spyOn(huggingface, 'fetchModelFiles')
      .mockResolvedValue([
        'encoder-model.onnx',
        'decoder-model.onnx',
        'tokenizer.json',
        'config.json',
      ]);

    try {
      const { getCanaryModel: getCanaryModelFromSource } = await import('../src/presets/canary.js');
      await expect(
        getCanaryModelFromSource('nvidia/canary-180m-flash', {
          encoderQuant: 'fp32',
          decoderQuant: 'fp32',
          preprocessorBackend: 'onnx',
          backend: 'wasm',
        }),
      ).rejects.toThrow("preprocessorBackend='js'");
    } finally {
      fetchModelFiles.mockRestore();
    }
  });

  it('maps NeMo-style aliases through the direct Canary wrapper and returns a legacy-friendly result', async () => {
    const session = {
      transcribe: vi.fn().mockResolvedValue({
        canonical: {
          text: 'Hallo Welt',
          language: 'de',
          words: [{ text: 'Hallo', startTime: 0, endTime: 0.4 }],
          segments: [{ text: 'Hallo Welt', startTime: 0, endTime: 0.8 }],
          warnings: [],
          metrics: { totalMs: 12, audioDurationSec: 0.8, rtfx: 66.67 },
          meta: {
            nativeAvailable: true,
            isFinal: true,
            backendId: 'wasm',
          },
        },
        native: {
          utteranceText: 'Hallo Welt',
          isFinal: true,
          language: 'de',
          warnings: [],
          prompt: {
            settings: {
              sourceLanguage: 'de',
              targetLanguage: 'de',
              decoderContext: '',
              emotion: '<|emo:undefined|>',
              punctuate: false,
              inverseTextNormalization: false,
              timestamps: true,
              diarize: false,
            },
            ids: [1, 2, 3],
            pieces: ['<|startofcontext|>'],
          },
        },
      }),
      dispose: vi.fn(),
    };
    const model = {
      info: {
        modelId: 'nvidia/canary-180m-flash',
        family: 'nemo-aed',
        preset: 'canary',
        backendId: 'wasm',
        classification: {
          ecosystem: 'nemo',
          processor: 'nemo-mel',
          encoder: 'fastconformer',
          decoder: 'transformer-decoder',
          topology: 'aed',
          family: 'canary',
          task: 'multitask-asr-translation',
        },
      },
      backend: { id: 'wasm' },
      createSession: vi.fn().mockResolvedValue(session),
      dispose: vi.fn(),
    };
    const runtime = {
      loadModel: vi.fn().mockResolvedValue(model),
    } as unknown as DefaultSpeechRuntime;

    const canary = await CanaryModel.fromPretrained('nvidia/canary-180m-flash', {
      runtime,
      backend: 'wasm',
      preprocessorBackend: 'js',
    });
    const result = await canary.transcribe(new Float32Array(1600), 16000, {
      source_lang: 'de',
      task: 'asr',
      pnc: 'no',
      timestamp: 'yes',
      returnPromptIds: true,
    });

    expect((runtime as any).loadModel).toHaveBeenCalledWith(
      expect.objectContaining({
        preset: 'canary',
        modelId: 'nvidia/canary-180m-flash',
        backend: 'wasm',
        options: {
          source: expect.objectContaining({
            kind: 'huggingface',
            repoId: 'ysdede/canary-180m-flash-onnx',
            preprocessorBackend: 'js',
          }),
        },
      }),
    );
    expect(session.transcribe).toHaveBeenCalledWith(
      expect.objectContaining({
        sampleRate: 16000,
        numberOfChannels: 1,
      }),
      expect.objectContaining({
        sourceLanguage: 'de',
        targetLanguage: undefined,
        task: 'asr',
        pnc: 'no',
        timestamp: 'yes',
        returnPromptIds: true,
        responseFlavor: 'canonical+native',
        detail: 'detailed',
      }),
    );
    expect(result.text).toBe('Hallo Welt');
    expect(result.language).toBe('de');
    expect(result.timestamp.word?.[0]?.text).toBe('Hallo');
    expect(result.native.prompt?.settings.punctuate).toBe(false);

    await canary.dispose();
    expect(session.dispose).toHaveBeenCalledTimes(1);
    expect(model.dispose).toHaveBeenCalledTimes(1);
  });

  it('disposes the loaded Canary model after one-shot transcription', async () => {
    const dispose = vi.fn();
    const transcribe = vi.fn().mockResolvedValue({
      text: 'hello',
      language: 'en',
      timestamp: {},
      warnings: [],
      canonical: {
        text: 'hello',
        warnings: [],
        meta: {
          nativeAvailable: true,
          isFinal: true,
          backendId: 'wasm',
        },
      },
      native: {
        utteranceText: 'hello',
        isFinal: true,
      },
    });
    const fromPretrained = vi
      .spyOn(CanaryModel, 'fromPretrained')
      .mockResolvedValue({ transcribe, dispose } as unknown as CanaryModel);

    try {
      const result = await transcribeCanary(new Float32Array(800), 16000, {
        modelId: 'nvidia/canary-180m-flash',
        transcribeOptions: {
          source_lang: 'en',
          target_lang: 'fr',
        },
      });

      expect(result.text).toBe('hello');
      expect(transcribe).toHaveBeenCalledWith(
        expect.any(Float32Array),
        16000,
        expect.objectContaining({
          source_lang: 'en',
          target_lang: 'fr',
        }),
      );
      expect(dispose).toHaveBeenCalledTimes(1);
    } finally {
      fromPretrained.mockRestore();
    }
  });
});
