import { beforeEach, describe, expect, it, vi } from 'vitest';
import {
  collectSpeechModelLocalEntries,
  createSpeechModelLocalEntries,
  inspectSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '@asrjs/speech-recognition/browser';
import { loadBuiltInSpeechModel } from '../src/runtime/builtins.js';
import { resolveParakeetLocalEntries } from '../src/presets/parakeet/compat.js';
import { resolveBuiltInLocalModelAdapter } from '../src/runtime/local-adapter-registry.js';
import { PipelineAbortedError } from '../src/pipeline/composition.js';
import { createQwen3AsrPresetFactory } from '../src/presets/qwen3-asr-factory.js';
import { createQwen3AsrModelFamily, type Qwen3AsrExecutor } from '../src/models/qwen-asr/index.js';

function qwenEntries(names: readonly string[]) {
  return names.map((name) => ({
    path: name,
    basename: name.split('/').pop() ?? name,
    file: new File(['artifact'], name.split('/').pop() ?? name),
  }));
}

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

  it('resolves the official Qwen graph set, including its shared fp16 external data', async () => {
    const entries = qwenEntries([
      'audio-encoder-dynamic.onnx',
      'decoder-prefill-fp16.onnx',
      'decoder-step-fp16.onnx',
      'decoder-fp16.onnx.data',
      'tokenizer/tokenizer.json',
    ]);
    const adapter = resolveBuiltInLocalModelAdapter('qwen3-asr-0.6b').adapter;

    expect(adapter.inspectEntries(entries)).toMatchObject({
      encoderQuantizations: ['fp32'],
      decoderQuantizations: ['fp16'],
      tokenizerNames: ['tokenizer.json'],
    });

    const resolved = await adapter.resolveEntries({
      modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
      entries,
      backend: 'wasm',
      decoderQuant: 'fp16',
      cpuThreads: 2,
    } as never);
    try {
      expect(resolved.selection).toMatchObject({
        encoderName: 'audio-encoder-dynamic.onnx',
        decoderName: 'decoder-step-fp16.onnx',
        tokenizerName: 'tokenizer.json',
        encoderQuant: 'fp32',
        decoderQuant: 'fp16',
      });
      expect(resolved.assetHandles).toHaveLength(5);
      expect(resolved.builtInLoadOptions).toMatchObject({
        source: {
          kind: 'direct',
          encoderBackend: undefined,
          decoderBackend: undefined,
          cpuThreads: 2,
          artifacts: {
            encoderUrl: expect.stringMatching(/^blob:/),
            decoderUrl: expect.stringMatching(/^blob:/),
            decoderStepUrl: expect.stringMatching(/^blob:/),
            tokenizerUrl: expect.stringMatching(/^blob:/),
            decoderPrefillDataUrl: expect.stringMatching(/^blob:/),
            decoderStepDataUrl: expect.any(String),
            decoderPrefillDataPath: 'decoder-fp16.onnx.data',
            decoderStepDataPath: 'decoder-fp16.onnx.data',
          },
        },
      });
      const options = resolved.builtInLoadOptions as {
        source: { artifacts: { decoderPrefillDataUrl: string; decoderStepDataUrl: string } };
      };
      expect(options.source.artifacts.decoderStepDataUrl).toBe(
        options.source.artifacts.decoderPrefillDataUrl,
      );
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('rejects the old three-graph Qwen export with a specific compatibility error', async () => {
    const entries = qwenEntries([
      'audio_front_model.onnx',
      'decoder_step_model.onnx',
      'embed_tokens_model.onnx',
      'tokenizer.json',
    ]);

    await expect(
      resolveBuiltInLocalModelAdapter('qwen3-asr-0.6b').adapter.resolveEntries({
        modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
        entries,
      } as never),
    ).rejects.toThrow(/legacy audio_front\/decoder_step\/embed_tokens.*incompatible/i);
  });

  it('uses the declared fp32 decoder default and requires its matching sidecars', async () => {
    const entries = qwenEntries([
      'audio-encoder-dynamic.onnx',
      'decoder-prefill.onnx',
      'decoder-prefill.onnx.data',
      'decoder-step.onnx',
      'decoder-step.onnx.data',
      'tokenizer/tokenizer.json',
    ]);
    const adapter = resolveBuiltInLocalModelAdapter('qwen3-asr-0.6b').adapter;

    expect(adapter.inspectEntries(entries).decoderQuantizations).toEqual(['fp32']);
    const resolved = await adapter.resolveEntries({
      modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
      entries,
      backend: 'wasm',
    } as never);
    try {
      expect(resolved.selection.decoderQuant).toBe('fp32');
      expect(resolved.assetHandles).toHaveLength(6);
      expect(resolved.builtInLoadOptions).toMatchObject({
        source: {
          kind: 'direct',
          artifacts: {
            decoderPrefillDataPath: 'decoder-prefill.onnx.data',
            decoderStepDataPath: 'decoder-step.onnx.data',
          },
        },
      });
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('selects fp16 by default when the local folder has only the fp16 decoder pair', async () => {
    const entries = qwenEntries([
      'audio-encoder-dynamic.onnx',
      'decoder-prefill-fp16.onnx',
      'decoder-step-fp16.onnx',
      'decoder-fp16.onnx.data',
      'tokenizer/tokenizer.json',
    ]);
    const adapter = resolveBuiltInLocalModelAdapter('qwen3-asr-0.6b').adapter;
    const resolved = await adapter.resolveEntries({
      modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
      entries,
    } as never);
    try {
      expect(resolved.selection.decoderQuant).toBe('fp16');
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }

    await expect(adapter.resolveEntries({
      modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
      entries,
      decoderQuant: 'fp32',
    } as never)).rejects.toThrow(/requested Qwen fp32 decoder pair/i);
  });

  it('loads through the public local-entry API and passes arbitrary PCM into the Qwen family', async () => {
    const entries = qwenEntries([
      'audio-encoder-dynamic.onnx',
      'decoder-prefill-fp16.onnx',
      'decoder-step-fp16.onnx',
      'decoder-fp16.onnx.data',
      'tokenizer/tokenizer.json',
    ]);
    const receivedAudio: Array<{ sampleRate: number; numberOfFrames: number; channels: readonly Float32Array[] }> = [];
    const executor: Qwen3AsrExecutor = {
      async transcribe(audio) {
        receivedAudio.push(audio);
        return {
          utteranceText: 'arbitrary clip',
          language: 'English',
          isFinal: true,
          segments: [{ index: 0, text: 'arbitrary clip', startTime: 0, endTime: audio.durationSeconds }],
        };
      },
      dispose() {},
    };

    vi.mocked(loadBuiltInSpeechModel).mockImplementation(async (loadOptions) => {
      const preset = createQwen3AsrPresetFactory();
      const resolvedRequest = await preset.resolveModelRequest(
        {
          modelId: loadOptions.modelId,
          options: loadOptions.options as never,
        } as never,
        {} as never,
      );
      const family = createQwen3AsrModelFamily({ dependencies: { executor } });
      const model = await family.createModel(resolvedRequest as never, {
        runtime: {} as never,
        backend: { id: loadOptions.backend ?? 'wasm' } as never,
        hooks: {},
      } as never);
      const session = await model.createSession();
      return {
        runtime: {} as never,
        model,
        session,
        transcribe: session.transcribe.bind(session),
        dispose: () => model.dispose(),
      } as never;
    });

    const loaded = await loadSpeechModelFromLocalEntries({
      modelId: 'qwen3-asr-0.6b',
      entries,
      backend: 'wasm',
      decoderQuant: 'fp16',
    });
    const pcm = Float32Array.from([0.1, -0.2, 0.3, -0.4, 0.5]);
    try {
      const result = await loaded.transcribeMonoPcm(pcm, 22_050);
      expect(result.text).toBe('arbitrary clip');
      expect(receivedAudio).toHaveLength(1);
      expect(receivedAudio[0]).toMatchObject({ sampleRate: 22_050, numberOfFrames: pcm.length });
      expect(Array.from(receivedAudio[0]!.channels[0]!)).toEqual(Array.from(pcm));
      expect(loadBuiltInSpeechModel).toHaveBeenCalledWith(
        expect.objectContaining({
          modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
          preset: 'qwen3-asr',
          backend: 'wasm',
          options: { source: { kind: 'direct', artifacts: expect.objectContaining({
            encoderUrl: expect.stringMatching(/^blob:/),
            decoderUrl: expect.stringMatching(/^blob:/),
            decoderStepUrl: expect.stringMatching(/^blob:/),
            decoderPrefillDataUrl: expect.stringMatching(/^blob:/),
            decoderStepDataUrl: expect.stringMatching(/^blob:/),
            decoderPrefillDataPath: 'decoder-fp16.onnx.data',
            decoderStepDataPath: 'decoder-fp16.onnx.data',
          }) } },
        }),
      );
    } finally {
      await loaded.dispose();
    }
  });

  it('rejects flat legacy Qwen preset options before model construction', async () => {
    const preset = createQwen3AsrPresetFactory();
    await expect(
      preset.resolveModelRequest(
        { modelId: 'Qwen/Qwen3-ASR-0.6B-hf', options: { frontUrl: 'blob:legacy' } } as never,
        {} as never,
      ),
    ).rejects.toThrow(/source.kind="direct".*legacy frontUrl/i);

    const directOptions = { source: { kind: 'direct', artifacts: {} } };
    await expect(
      preset.resolveModelRequest(
        { modelId: 'Qwen/Qwen3-ASR-0.6B-hf', options: directOptions } as never,
        {} as never,
      ),
    ).resolves.toMatchObject({
      family: 'qwen-asr',
      resolvedPreset: 'qwen3-asr',
      options: directOptions,
    });
  });

  it('rejects local inspection for built-in models without local-folder support', () => {
    expect(() =>
      inspectSpeechModelLocalEntries('google/medasr', createSpeechModelLocalEntries([])),
    ).toThrow('does not support local folder loading');
  });

  it('inspects and resolves Nemotron local folders with encoder, decoder and joint graphs', async () => {
    const entries = createSpeechModelLocalEntries([
      new File(['enc'], 'encoder_320ms.fp16.onnx'),
      new File(['encData'], 'encoder_320ms.fp16.onnx.data'),
      new File(['dec'], 'decoder.onnx'),
      new File(['joint'], 'joint.onnx'),
      new File(['tokens'], 'tokens.txt'),
    ]);

    expect(
      inspectSpeechModelLocalEntries('nemotron-3.5-asr-streaming-0.6b', entries),
    ).toMatchObject({
      encoderQuantizations: ['fp16'],
      decoderQuantizations: ['fp32'],
      tokenizerNames: ['tokens.txt'],
    });

    const resolved = await resolveBuiltInLocalModelAdapter(
      'nemotron-3.5-asr-streaming-0.6b',
    ).adapter.resolveEntries({
      modelId: 'nemotron-3.5-asr-streaming-0.6b',
      entries,
      preprocessorBackend: 'js',
    } as never);

    try {
      expect(resolved.preset).toBe('nemotron');
      expect(resolved.selection).toMatchObject({
        encoderName: 'encoder_320ms.fp16.onnx',
        decoderName: 'decoder.onnx',
        tokenizerName: 'tokens.txt',
      });
      const builtInLoadOptions = resolved.builtInLoadOptions;
      if (
        typeof builtInLoadOptions !== 'object' ||
        builtInLoadOptions === null ||
        !('source' in builtInLoadOptions)
      ) {
        throw new Error('Expected a direct local source');
      }

      const source = builtInLoadOptions.source;
      if (typeof source !== 'object' || source === null || !('artifacts' in source)) {
        throw new Error('Expected direct local artifacts');
      }

      expect(Object.keys(source.artifacts).sort()).toEqual([
        'decoderDataUrl',
        'decoderFilename',
        'decoderUrl',
        'encoderDataUrl',
        'encoderFilename',
        'encoderUrl',
        'jointDataUrl',
        'jointFilename',
        'jointUrl',
        'tokenizerUrl',
      ]);
      expect(resolved.assetHandles.length).toBe(5);
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('fails closed when a Nemotron local folder is missing its joint graph', async () => {
    const entries = createSpeechModelLocalEntries([
      new File(['enc'], 'encoder_320ms.onnx'),
      new File(['dec'], 'decoder.onnx'),
      new File(['tokens'], 'tokens.txt'),
    ]);

    await expect(
      resolveBuiltInLocalModelAdapter('nemotron-3.5-asr-streaming-0.6b').adapter.resolveEntries({
        modelId: 'nemotron-3.5-asr-streaming-0.6b',
        entries,
      } as never),
    ).rejects.toThrow('Missing joint file: joint.onnx');
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
