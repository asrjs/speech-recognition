import {
  buildBuiltInHubLoadOptions,
  buildBuiltInTranscriptionOptions,
  detectBuiltInModelQuantizationsFromFiles,
  getBuiltInModelDescriptor,
  getBuiltInModelLanguageName,
  listBuiltInModelDescriptors,
  listBuiltInModelOptions,
  listApprovedSpeechModelManifests,
  resolveApprovedSpeechExecutionPlan,
  resolveBuiltInModelComponentBackends,
  type BuiltInModelDescriptor,
} from '@asrjs/speech-recognition/presets';
import { describe, expect, expectTypeOf, it } from 'vitest';
import { resolveBuiltInLocalModelAdapter } from '../src/runtime/local-adapter-registry.js';

describe('built-in preset descriptors', () => {
  it('lists built-in models with consolidated metadata', () => {
    const descriptors = listBuiltInModelDescriptors();
    const options = listBuiltInModelOptions();

    expectTypeOf(descriptors).toEqualTypeOf<readonly BuiltInModelDescriptor[]>();
    expect(descriptors.length).toBeGreaterThan(0);
    expect(descriptors.map((descriptor) => descriptor.modelId)).toEqual(
      expect.arrayContaining([
        'parakeet-tdt-0.6b-v2',
        'parakeet-tdt-0.6b-v3',
        'nvidia/canary-180m-flash',
        'google/medasr',
        'facebook/wav2vec2-base-960h',
        'gigaam-multilingual-ctc',
        'gigaam-v3-e2e-rnnt',
        'FunAudioLLM/SenseVoiceSmall',
      ]),
    );
    expect(options.find((option) => option.key === 'nvidia/canary-180m-flash')?.preset).toBe(
      'canary',
    );
  });

  it('declares local experimental GigaAM and SenseVoice capabilities conservatively', () => {
    const ctc = getBuiltInModelDescriptor('gigaam-multilingual-ctc');
    const rnnt = getBuiltInModelDescriptor('gigaam-v3-e2e-rnnt');
    const senseVoice = getBuiltInModelDescriptor('FunAudioLLM/SenseVoiceSmall');

    for (const descriptor of [ctc, rnnt, senseVoice]) {
      expect(descriptor?.loading.supportsHubSource).toBe(false);
      expect(descriptor?.loading.supportsLocalSource).toBe(true);
      expect(descriptor?.warmup.mode).toBe('non-empty');
      expect(descriptor?.capabilities.supportsTranslation).toBe(false);
    }
    expect(rnnt?.languages).toEqual(['ru']);
    expect(rnnt?.inference.defaultMergeStrategy).toBe('concat');
    expect(senseVoice?.inference.maxInputDurationSec).toBe(30);
    expect(senseVoice?.inference.maxWindowDurationSec).toBe(30);
    expect(senseVoice?.inference.autoWindowThresholdSec).toBe(30);
  });

  it('exposes Canary capabilities, controls, and loading defaults', () => {
    const descriptor = getBuiltInModelDescriptor('canary-180m-flash');

    expect(descriptor?.preset).toBe('canary');
    expect(descriptor?.capabilities.supportsTranslation).toBe(true);
    expect(descriptor?.capabilities.supportsPunctuationCapitalization).toBe(true);
    expect(descriptor?.loading.encoderArtifactBaseName).toBe('encoder-model');
    expect(descriptor?.loading.decoderArtifactBaseName).toBe('decoder-model');
    expect(descriptor?.loading.availableEncoderBackends).toEqual(['webgpu', 'wasm']);
    expect(descriptor?.loading.availableDecoderBackends).toEqual(['webgpu', 'wasm']);
    expect(descriptor?.loading.defaultEncoderBackend).toBe('webgpu');
    expect(descriptor?.loading.defaultDecoderBackend).toBe('wasm');
    expect(descriptor?.loading.availableEncoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.availableDecoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.defaultEncoderQuantization).toBe('fp32');
    expect(descriptor?.loading.defaultDecoderQuantization).toBe('int8');
    expect(descriptor?.controls.map((control) => control.key)).toEqual([
      'task',
      'targetLanguage',
      'punctuate',
      'timestamps',
    ]);
  });

  it('lists the custom Whisper Turbo artifact as its own WebGPU local-capable model', () => {
    const turbo = getBuiltInModelDescriptor('ysdede/whisper-large-v3-turbo-onnx-4graph');
    const publicLarge = getBuiltInModelDescriptor('openai/whisper-large-v3');

    expect(turbo).toMatchObject({
      modelId: 'ysdede/whisper-large-v3-turbo-onnx-4graph',
      preset: 'whisper',
      loading: {
        supportsHubSource: true,
        supportsLocalSource: true,
        availableEncoderBackends: ['webgpu'],
        availableDecoderBackends: ['webgpu'],
        availableEncoderQuantizations: ['fp16'],
        availableDecoderQuantizations: ['fp16'],
      },
    });
    expect(publicLarge?.modelId).toBe('openai/whisper-large-v3');
    expect(
      resolveBuiltInLocalModelAdapter('ysdede/whisper-large-v3-turbo-onnx-4graph').preset,
    ).toBe('whisper');
  });

  it('exposes split quantization metadata for Parakeet too', () => {
    const descriptor = getBuiltInModelDescriptor('parakeet-tdt-0.6b-v2');

    expect(descriptor?.loading.availableEncoderBackends).toEqual(['webgpu', 'wasm']);
    expect(descriptor?.loading.availableDecoderBackends).toEqual(['wasm']);
    expect(descriptor?.loading.defaultEncoderBackend).toBe('webgpu');
    expect(descriptor?.loading.defaultDecoderBackend).toBe('wasm');
    expect(descriptor?.loading.availableEncoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.availableDecoderQuantizations).toEqual(['fp16', 'int8', 'fp32']);
    expect(descriptor?.loading.defaultEncoderQuantization).toBe('fp16');
    expect(descriptor?.loading.defaultDecoderQuantization).toBe('int8');
    expect(descriptor?.loading.defaultRevision).toBe('main');
  });

  it('builds preset-aware hub load requests from a single generic helper', () => {
    const parakeet = buildBuiltInHubLoadOptions({
      modelId: 'parakeet-tdt-0.6b-v3',
      revision: 'feat/fp16-canonical-v3',
      backend: 'webgpu-hybrid',
      encoderQuant: 'fp16',
      decoderQuant: 'int8',
      preprocessorName: 'nemo128',
      preprocessorBackend: 'js',
    });
    const canary = buildBuiltInHubLoadOptions({
      modelId: 'nvidia/canary-180m-flash',
      backend: 'wasm',
      encoderQuant: 'fp32',
      decoderQuant: 'int8',
      preprocessorBackend: 'js',
    });

    expect(parakeet).toMatchObject({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
      options: {
        source: {
          repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
          encoderQuant: 'fp16',
          decoderQuant: 'int8',
          preprocessorBackend: 'js',
        },
      },
    });
    expect(canary).toMatchObject({
      preset: 'canary',
      modelId: 'nvidia/canary-180m-flash',
      options: {
        source: {
          repoId: 'ysdede/canary-180m-flash-onnx',
          encoderBackend: 'wasm',
          decoderBackend: 'wasm',
          encoderQuant: 'fp32',
          decoderQuant: 'int8',
          preprocessorBackend: 'js',
        },
      },
    });
    const qwen = buildBuiltInHubLoadOptions({ modelId: 'qwen3-asr-0.6b', backend: 'wasm', cpuThreads: 4 });
    expect(qwen.options).toMatchObject({ source: {
      kind: 'huggingface', repoId: 'ysdede/qwen3-asr-0.6b-onnx',
      revision: 'ac0bedcc0f1614c0b3982940c14bdb169ff2365d', encoderBackend: 'wasm',
      decoderBackend: 'wasm', encoderQuant: 'fp32', decoderQuant: 'fp16', cpuThreads: 4,
    } });
    const moonshine = buildBuiltInHubLoadOptions({ modelId: 'moonshine-streaming-tiny', backend: 'wasm', cpuThreads: 2 });
    expect(moonshine.options).toMatchObject({ encoderBackend: 'wasm', cpuThreads: 2 });
  });

  it('builds preset-aware transcription options with Canary prompt semantics', () => {
    const canaryAsr = buildBuiltInTranscriptionOptions('nvidia/canary-180m-flash', {
      sourceLanguage: 'de',
      task: 'asr',
      punctuate: false,
      timestamps: true,
      enableProfiling: true,
    });
    const canaryTranslate = buildBuiltInTranscriptionOptions('nvidia/canary-180m-flash', {
      sourceLanguage: 'fr',
      targetLanguage: 'en',
      task: 'translation',
      punctuate: true,
      timestamps: false,
    });
    const parakeet = buildBuiltInTranscriptionOptions('parakeet-tdt-0.6b-v2', {
      frameStride: 2,
      timestamps: true,
      returnConfidences: true,
      enableProfiling: true,
    });

    expect(canaryAsr).toEqual({
      sourceLanguage: 'de',
      targetLanguage: 'de',
      task: 'asr',
      pnc: 'no',
      timestamp: 'yes',
      enableProfiling: true,
    });
    expect(canaryTranslate).toEqual({
      sourceLanguage: 'fr',
      targetLanguage: 'en',
      task: 'translation',
      pnc: 'yes',
      timestamp: 'no',
      enableProfiling: undefined,
    });
    expect(parakeet).toEqual({
      returnTimestamps: true,
      returnConfidences: true,
      frameStride: 2,
      enableProfiling: true,
    });
  });

  it('exposes Wav2Vec2 CTC metadata with raw-waveform loading defaults', () => {
    const descriptor = getBuiltInModelDescriptor('wav2vec2-base-960h');

    expect(descriptor?.preset).toBe('wav2vec2');
    expect(descriptor?.classification).toMatchObject({
      ecosystem: 'meta',
      processor: 'wav2vec2-conv',
      encoder: 'wav2vec2-conformer',
      decoder: 'ctc',
      topology: 'ctc',
      family: 'wav2vec2',
      task: 'asr',
    });
    expect(descriptor?.languages).toEqual(['en']);
    expect(descriptor?.capabilities.supportsWordTimestamps).toBe(true);
    expect(descriptor?.loading.supportsHubSource).toBe(true);
    expect(descriptor?.loading.supportsLocalSource).toBe(true);
    expect(descriptor?.loading.defaultEncoderBackend).toBe('wasm');
    expect(descriptor?.loading.defaultDecoderBackend).toBe('wasm');
    expect(descriptor?.loading.encoderArtifactBaseName).toBe('model');
  });

  it('builds Wav2Vec2 transcription defaults through the generic helper', () => {
    expect(
      buildBuiltInTranscriptionOptions('facebook/wav2vec2-base-960h', {
        timestamps: true,
        returnConfidences: true,
      }),
    ).toEqual({
      returnTimestamps: true,
      returnConfidences: true,
      frameStride: undefined,
      enableProfiling: undefined,
    });
  });

  it('resolves human-readable language names across preset families', () => {
    expect(getBuiltInModelLanguageName('zh')).toBe('Chinese');
    expect(getBuiltInModelLanguageName('fr')).toBe('French');
    expect(getBuiltInModelLanguageName('auto')).toBe('Auto-detect');
  });

  it('resolves model-specific component backend defaults and restrictions', () => {
    expect(resolveBuiltInModelComponentBackends('parakeet-tdt-0.6b-v2')).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    });

    expect(
      resolveBuiltInModelComponentBackends('parakeet-tdt-0.6b-v2', {
        encoderBackend: 'wasm',
        decoderBackend: 'wasm',
      }),
    ).toEqual({
      encoderBackend: 'wasm',
      decoderBackend: 'wasm',
    });

    expect(
      resolveBuiltInModelComponentBackends('parakeet-tdt-0.6b-v2', {
        backend: 'webgpu-hybrid',
      }),
    ).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    });

    expect(
      resolveBuiltInModelComponentBackends('nvidia/canary-180m-flash', {
        backend: 'webgpu-strict',
      }),
    ).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'webgpu',
    });

    expect(() => resolveBuiltInModelComponentBackends('unknown-model-id')).toThrow(
      'Unknown built-in model "unknown-model-id".',
    );
  });

  it('selects the receipt-backed TDT v3 profile only for its exact immutable artifact identity', () => {
    const manifest = listApprovedSpeechModelManifests().find((item) => item.modelId === 'parakeet-tdt-0.6b-v3');
    expect(manifest?.artifacts?.encoder.sha256).toBe('a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717');
    const plan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'fp16', 'int8'], wasmPrecisions: ['fp32', 'int8'] },
      artifactIdentity: {
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'f88260fa0777fe0868dda6df85d1a98f012a4a7a',
        encoderSha256: 'a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717',
        decoderSha256: 'eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70',
        configSha256: '562aedb0b804acbeb4e04215de190846c37e0bb239b696376df69d555785373a',
        vocabularySha256: 'ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742',
      },
    });
    expect(plan).toMatchObject({
      status: 'approved', profileId: 'webgpu-fp16-wasm-int8-v1',
      encoderBackend: 'webgpu', decoderBackend: 'wasm', encoderQuant: 'fp16', decoderQuant: 'int8',
      jointBackend: 'wasm', jointQuant: 'int8',
    });
  });

  it('reports automatic departures from the approved profile as warnings', () => {
    const plan = (runtime: Parameters<typeof resolveApprovedSpeechExecutionPlan>[1]['runtime'], extra = {}) =>
      resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', { runtime, backend: 'webgpu-hybrid', ...extra });

    expect(plan({ webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'fp16', 'int8'] }).warnings).toBeUndefined();

    const noF16 = plan({ webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'int8'] });
    expect(noF16.encoderQuant).toBe('fp32');
    expect(noF16.warnings?.map((warning) => warning.code)).toEqual(['plan.encoder-precision-fallback']);
    expect(noF16.warnings?.[0]?.message).toContain('larger download');

    const noGpu = plan({ webgpu: false, wasm: true, wasmPrecisions: ['fp32', 'int8'] });
    expect(noGpu.encoderBackend).toBe('wasm');
    expect(noGpu.warnings?.map((warning) => warning.code)).toContain('plan.encoder-backend-fallback');

    // Explicit caller choices are not reported as fallbacks.
    expect(plan({ webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'int8'] }, { encoderQuant: 'fp32' }).warnings).toBeUndefined();
    expect(plan({ webgpu: true, wasm: true }, { backend: 'wasm' }).warnings).toBeUndefined();
  });

  it('does not confer approval from model name or artifact filenames alone', () => {
    expect(resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'fp16', 'int8'], wasmPrecisions: ['fp32', 'int8'] },
    }).status).toBe('unverified');
    expect(resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: true, wasm: true },
      artifactIdentity: {
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'main',
        encoderSha256: 'a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717',
        decoderSha256: 'eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70',
        configSha256: '562aedb0b804acbeb4e04215de190846c37e0bb239b696376df69d555785373a',
        vocabularySha256: 'ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742',
      },
    }).status).toBe('unverified');
    expect(resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: true, wasm: true },
      artifactIdentity: {
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'f88260fa0777fe0868dda6df85d1a98f012a4a7a',
        encoderSha256: 'a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717',
        decoderSha256: 'eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70',
        configSha256: '0000000000000000000000000000000000000000000000000000000000000000',
        vocabularySha256: 'ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742',
      },
    }).status).toBe('unverified');
  });

  it('chooses a runtime-supported fallback and marks the unreceipted profile unverified', () => {
    expect(resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: false, wasm: true, wasmPrecisions: ['fp32', 'int8'] },
    })).toMatchObject({ status: 'unverified', encoderBackend: 'wasm', decoderBackend: 'wasm' });
    expect(() => resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: false, wasm: true, wasmPrecisions: ['fp32', 'int8'] }, encoderBackend: 'webgpu',
    })).toThrow('Requested execution plan is unavailable');
    expect(resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'int8'], wasmPrecisions: ['fp32', 'int8'] },
      artifactIdentity: {
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        revision: 'f88260fa0777fe0868dda6df85d1a98f012a4a7a',
        encoderSha256: 'a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717',
        decoderSha256: 'eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70',
        configSha256: '562aedb0b804acbeb4e04215de190846c37e0bb239b696376df69d555785373a',
        vocabularySha256: 'ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742',
      },
    })).toMatchObject({ status: 'unverified', encoderQuant: 'fp32' });
    expect(() => resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: true, wasm: true, webgpuPrecisions: ['fp32', 'int8'], wasmPrecisions: ['fp32', 'int8'] },
      encoderQuant: 'fp16',
    })).toThrow('Requested execution plan is unavailable');
  });

  it('does not treat generic WASM runtime precision flags as ONNX weight-quantization support', () => {
    const plan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
      runtime: { webgpu: false, wasm: true },
      decoderBackend: 'wasm',
      decoderQuant: 'fp16',
    });
    expect(plan).toMatchObject({ status: 'unverified', decoderBackend: 'wasm', decoderQuant: 'fp16' });
  });

  it('preserves the published Qwen FP16 WASM decoder default with generic WASM hints', () => {
    expect(resolveApprovedSpeechExecutionPlan('qwen3-asr-0.6b', {
      runtime: { webgpu: false, wasm: true, wasmPrecisions: ['fp32', 'int8'] },
    })).toMatchObject({ encoderBackend: 'wasm', decoderBackend: 'wasm', encoderQuant: 'fp32', decoderQuant: 'fp16' });
  });

  it('detects quantization variants from repo filenames using built-in artifact names', () => {
    const detected = detectBuiltInModelQuantizationsFromFiles('nvidia/canary-180m-flash', [
      'encoder-model.onnx',
      'encoder-model.fp16.onnx',
      'decoder-model.int8.onnx',
      'decoder-model.onnx',
      'tokenizer.json',
    ]);

    expect(detected.encoderArtifactBaseName).toBe('encoder-model');
    expect(detected.decoderArtifactBaseName).toBe('decoder-model');
    expect(detected.encoder).toEqual(['fp16', 'fp32']);
    expect(detected.decoder).toEqual(['int8', 'fp32']);
  });

  it('falls back to declared quantizations when no repo files match', () => {
    const detected = detectBuiltInModelQuantizationsFromFiles('nvidia/canary-180m-flash', []);

    expect(detected.encoderArtifactBaseName).toBe('encoder-model');
    expect(detected.decoderArtifactBaseName).toBe('decoder-model');
    expect(detected.encoder).toEqual(['fp32']);
    expect(detected.decoder).toEqual(['fp32']);
  });

  it('throws when detecting quantizations for an unknown model id', () => {
    expect(() => detectBuiltInModelQuantizationsFromFiles('unknown-model-id', [])).toThrow(
      'Unknown built-in model "unknown-model-id".',
    );
  });
});
