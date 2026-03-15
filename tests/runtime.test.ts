import { createSpeechRuntime } from '@asrjs/speech-recognition';
import type {
  BackendCapabilities,
  ExecutionBackend,
  TranscriptResult,
  TranscriptionEnvelope,
} from '@asrjs/speech-recognition';
import { createHfCtcModelFamily, type HfCtcNativeTranscript } from '@asrjs/speech-recognition/models/hf-ctc-common';
import { createNemoTdtModelFamily, type NemoTdtNativeTranscript } from '@asrjs/speech-recognition/models/nemo-tdt';
import {
  createWhisperSeq2SeqModelFamily,
  type WhisperNativeTranscript,
} from '@asrjs/speech-recognition/models/whisper-seq2seq';
import { createMedAsrPresetFactory } from '@asrjs/speech-recognition/presets/medasr';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';
import { createWhisperPresetFactory } from '@asrjs/speech-recognition/presets/whisper';
import { describe, expect, expectTypeOf, it } from 'vitest';

function createStaticBackend(capabilities: BackendCapabilities): ExecutionBackend {
  return {
    id: capabilities.id,
    displayName: capabilities.displayName,
    async probeCapabilities() {
      return capabilities;
    },
    async createExecutionContext() {
      return {
        backendId: capabilities.id,
        capabilities,
        dispose() {
          return undefined;
        },
      };
    },
  };
}

describe('DefaultSpeechRuntime', () => {
  it('selects the best backend by capability and preference', async () => {
    const runtime = createSpeechRuntime();

    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerBackend(
      createStaticBackend({
        id: 'webgpu',
        displayName: 'WebGPU',
        available: true,
        priority: 100,
        environments: ['browser'],
        acceleration: ['gpu'],
        supportedPrecisions: ['fp32', 'fp16', 'int8'],
        supportsFp16: true,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );

    const backend = await runtime.selectBackend({
      preferredBackendIds: ['webgpu', 'wasm'],
      requiredPrecision: 'fp16',
      allowExperimental: false,
    });

    expect(backend.id).toBe('webgpu');
  });

  it('loads an architecture-based model family from classification metadata', async () => {
    const runtime = createSpeechRuntime();
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerModelFamily(createNemoTdtModelFamily());

    const model = await runtime.loadModel({
      family: 'nemo-tdt',
      modelId: 'nemo-fastconformer-tdt-scaffold',
      classification: {
        ecosystem: 'nemo',
        encoder: 'fastconformer',
        decoder: 'tdt',
        task: 'asr',
      },
    });
    const session = await model.createSession();

    const canonical = await session.transcribe(new Float32Array(16000), {
      detail: 'detailed',
      responseFlavor: 'canonical',
    });
    const envelope = await session.transcribe(new Float32Array(16000), {
      detail: 'detailed',
      responseFlavor: 'canonical+native',
      returnFrameIndices: true,
    });

    expect(canonical.meta.backendId).toBe('wasm');
    expect(canonical.text.length).toBeGreaterThan(0);
    expect(canonical.tokens?.length).toBeGreaterThan(0);
    expect(model.info.family).toBe('nemo-tdt');
    expect(model.info.classification.processor).toBe('nemo-mel');
    expect(model.info.classification.decoder).toBe('tdt');
    expect(model.info.classification.topology).toBe('tdt');
    expect(model.info.architecture?.processor.module).toBe('audio');
    expect(model.info.architecture?.decoding.module).toBe('inference');
    expect(model.info.architecture?.tokenizer.module).toBe('inference');
    expect(envelope.canonical.meta.nativeAvailable).toBe(true);
    expect(envelope.native?.warnings?.[0]?.code).toBe('nemo-tdt.stubbed-decoder');

    expectTypeOf(canonical).toMatchTypeOf<TranscriptResult>();
    expectTypeOf(envelope).toMatchTypeOf<TranscriptionEnvelope<NemoTdtNativeTranscript>>();
  });

  it('allows branded presets to stay thin over architecture-based implementations', async () => {
    const runtime = createSpeechRuntime();
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerModelFamily(createNemoTdtModelFamily());
    runtime.registerPreset(createParakeetPresetFactory());

    const model = await runtime.loadModel({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
    });

    expect(model.info.family).toBe('nemo-tdt');
    expect(model.info.preset).toBe('parakeet');
    expect(model.info.classification.processor).toBe('nemo-mel');
    expect(model.info.classification.family).toBe('parakeet');
    expect(model.info.classification.decoder).toBe('tdt');
  });

  it('can inject the built-in Parakeet artifact source without moving brand logic into nemo-tdt', async () => {
    const runtime = createSpeechRuntime();
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerModelFamily(createNemoTdtModelFamily());
    runtime.registerPreset(
      createParakeetPresetFactory({
        useManifestSource: true,
      }),
    );

    const model = await runtime.loadModel({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
    });

    expect(model.loadOptions?.source?.kind).toBe('huggingface');
    if (model.loadOptions?.source?.kind === 'huggingface') {
      expect(model.loadOptions.source.repoId).toBe('ysdede/parakeet-tdt-0.6b-v3-onnx');
    }
  });

  it('loads an HF-style CTC family without forcing it through NeMo abstractions', async () => {
    const runtime = createSpeechRuntime();
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerModelFamily(createHfCtcModelFamily());

    const model = await runtime.loadModel({
      family: 'hf-ctc',
      modelId: 'wav2vec2-conformer-medical-scaffold',
      classification: {
        ecosystem: 'hf',
        processor: 'wav2vec2-conv',
        decoder: 'ctc',
        topology: 'ctc',
        task: 'asr',
      },
    });
    const session = await model.createSession();
    const envelope = await session.transcribe(new Float32Array(16000), {
      detail: 'detailed',
      responseFlavor: 'canonical+native',
      returnTokenIds: true,
    });

    expect(model.info.family).toBe('hf-ctc');
    expect(model.info.classification.processor).toBe('wav2vec2-conv');
    expect(model.info.architecture?.decoder.module).toBe('inference');
    expect(envelope.native?.warnings?.[0]?.code).toBe('hf-ctc.stubbed-decoder');
    expectTypeOf(envelope).toMatchTypeOf<TranscriptionEnvelope<HfCtcNativeTranscript>>();
  });

  it('keeps branded presets downstream for MedASR and Whisper', async () => {
    const runtime = createSpeechRuntime();
    runtime.registerBackend(
      createStaticBackend({
        id: 'wasm',
        displayName: 'WASM',
        available: true,
        priority: 60,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        notes: [],
      }),
    );
    runtime.registerModelFamily(createHfCtcModelFamily());
    runtime.registerPreset(createMedAsrPresetFactory());
    runtime.registerPreset(createWhisperPresetFactory());
    runtime.registerModelFamily(createWhisperSeq2SeqModelFamily());

    const medasr = await runtime.loadModel({
      preset: 'medasr',
      modelId: 'google/medasr',
    });
    const whisper = await runtime.loadModel({
      preset: 'whisper',
      modelId: 'openai/whisper-base',
    });
    const whisperSession = await whisper.createSession();
    const whisperEnvelope = await whisperSession.transcribe(new Float32Array(16000), {
      detail: 'segments',
      responseFlavor: 'canonical+native',
    });

    expect(medasr.info.family).toBe('hf-ctc');
    expect(medasr.info.preset).toBe('medasr');
    expect(medasr.info.classification.family).toBe('medasr');
    expect(medasr.info.classification.decoder).toBe('ctc');
    expect(whisper.info.family).toBe('whisper-seq2seq');
    expect(whisper.info.preset).toBe('whisper');
    expect(whisper.info.classification.family).toBe('whisper');
    expect(whisper.info.classification.topology).toBe('aed');
    expect(whisper.info.architecture?.encoder.module).toBe('inference');
    expect(whisperEnvelope.native?.warnings?.[0]?.code).toBe('whisper-seq2seq.stubbed-decoder');
    expectTypeOf(whisperEnvelope).toMatchTypeOf<TranscriptionEnvelope<WhisperNativeTranscript>>();
  });
});
