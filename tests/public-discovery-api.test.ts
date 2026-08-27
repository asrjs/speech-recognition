import {
  buildSpeechModelLoadOptions,
  buildSpeechTranscriptionOptions,
  createExperimentalArtifactMissingError,
  getSpeechModelDescriptor,
  getSpeechModelLanguageName,
  getExperimentalSpeechFamily,
  listExperimentalSpeechFamilies,
  listSpeechModelOptions,
  listSpeechModels,
  resolveSpeechModelComponentBackends,
} from '@asrjs/speech-recognition';
import type {
  ExperimentalSpeechAudioContract,
  ExperimentalSpeechFamilyDescriptor,
} from '@asrjs/speech-recognition';
import { describe, expect, expectTypeOf, it } from 'vitest';

describe('root speech model discovery helpers', () => {
  it('exposes built-in model discovery through the root entry point', () => {
    const descriptors = listSpeechModels();
    const options = listSpeechModelOptions();
    const canary = getSpeechModelDescriptor('canary-180m-flash');

    expect(descriptors.map((descriptor) => descriptor.modelId)).toEqual(
      expect.arrayContaining([
        'parakeet-tdt-0.6b-v2',
        'parakeet-tdt-0.6b-v3',
        'nvidia/canary-180m-flash',
        'google/medasr',
      ]),
    );
    expect(options.find((option) => option.key === 'nvidia/canary-180m-flash')?.preset).toBe(
      'canary',
    );
    expect(canary?.preset).toBe('canary');
    expect(getSpeechModelLanguageName('auto')).toBe('Auto-detect');
  });

  it('builds load and transcription options from the root entry point', () => {
    const loadOptions = buildSpeechModelLoadOptions({
      modelId: 'parakeet-tdt-0.6b-v3',
      backend: 'webgpu-hybrid',
      encoderQuant: 'fp16',
      decoderQuant: 'int8',
      preprocessorName: 'nemo128',
      preprocessorBackend: 'js',
    });
    const transcribeOptions = buildSpeechTranscriptionOptions('nvidia/canary-180m-flash', {
      sourceLanguage: 'de',
      task: 'asr',
      punctuate: false,
      timestamps: true,
    });

    expect(loadOptions).toMatchObject({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
      options: {
        source: {
          encoderQuant: 'fp16',
          decoderQuant: 'int8',
          preprocessorBackend: 'js',
        },
      },
    });
    expect(transcribeOptions).toEqual({
      sourceLanguage: 'de',
      targetLanguage: 'de',
      task: 'asr',
      pnc: 'no',
      timestamp: 'yes',
      enableProfiling: undefined,
    });
  });

  it('resolves model-specific component backend defaults from the root entry point', () => {
    expect(
      resolveSpeechModelComponentBackends('parakeet-tdt-0.6b-v2', {
        backend: 'webgpu-hybrid',
      }),
    ).toEqual({
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    });
  });

  it('lists experimental families without claiming verified presets', () => {
    const experimental = listExperimentalSpeechFamilies();
    const presetIds = new Set(listSpeechModels().map((descriptor) => descriptor.modelId));

    expect(experimental.map((entry) => entry.family)).toEqual([
      'gigaam-ctc',
      'gigaam-rnnt',
      'sensevoice',
      'x-asr',
      'qwen-asr',
    ]);
    for (const entry of experimental) {
      expect(entry.status).toBe('experimental');
      expect(entry.verifiedPreset).toBe(false);
      expect(entry.publicHostedWeights).toBe(false);
      expect(entry.locator).toBe('local-onnx-dir');
      expect(presetIds.has(entry.modelIdHint)).toBe(false);
    }
    expect(getExperimentalSpeechFamily('qwen-asr')?.notes).toContain('audio-encoder-dynamic.onnx');
    expect(getSpeechModelDescriptor('Qwen/Qwen3-ASR-0.6B')).toBeNull();

    const qwen = getExperimentalSpeechFamily('qwen-asr');
    expect(qwen?.audioContract).toBe('short-clip-speech-llm');
    expect(qwen?.languages).toEqual(['multilingual']);
    expect(qwen?.limitations.some((item) => /short-clip/i.test(item))).toBe(true);
    expect(qwen?.limitations.some((item) => /not encoder-cache streaming/i.test(item))).toBe(true);

    const rnnt = getExperimentalSpeechFamily('gigaam-rnnt');
    expect(rnnt?.modelIdHint).toBe('gigaam-v3-e2e-rnnt');
    expect(rnnt?.notes).toMatch(/Russian-only/i);
    expect(rnnt?.notes).toContain('example.wav');
    expect(rnnt?.languages).toEqual(['ru']);
    expect(rnnt?.audioContract).toBe('offline-rnnt');
    expect(rnnt?.limitations.some((item) => /Russian-only/i.test(item))).toBe(true);
    expect(getExperimentalSpeechFamily('gigaam-v3-e2e-rnnt')?.family).toBe('gigaam-rnnt');
    expect(getSpeechModelDescriptor('gigaam-v3-e2e-rnnt')).toBeNull();

    const xasr = getExperimentalSpeechFamily('x-asr');
    expect(xasr?.audioContract).toBe('encoder-cache-streaming');
    expect(xasr?.languages).toEqual(['zh', 'en']);
  });

  it('documents audioContract and limitations on clone-safe public types', () => {
    const listed = listExperimentalSpeechFamilies();
    const clonedList = structuredClone(listed);
    expect(clonedList).toEqual([...listed]);
    expect(clonedList).not.toBe(listed);

    const qwen = getExperimentalSpeechFamily('qwen-asr');
    expect(qwen).not.toBeNull();
    expect(structuredClone(qwen)).toEqual(qwen);

    const contract: ExperimentalSpeechAudioContract = 'short-clip-speech-llm';
    const descriptor: ExperimentalSpeechFamilyDescriptor = qwen!;
    expect(descriptor.audioContract).toBe(contract);
    expect(descriptor.limitations.length).toBeGreaterThan(0);
    expectTypeOf(descriptor.audioContract).toEqualTypeOf<ExperimentalSpeechAudioContract>();
    expectTypeOf(descriptor.limitations).toEqualTypeOf<readonly string[]>();

    (listed[0] as { limitations: string[] }).limitations.push('mutated-by-caller');
    expect(listExperimentalSpeechFamilies()[0].limitations).not.toContain('mutated-by-caller');

    const missing = createExperimentalArtifactMissingError('qwen-asr');
    expect(structuredClone(missing.details)).toEqual(missing.details);
    expect(missing.details?.audioContract).toBe('short-clip-speech-llm');
  });

  it('throws an actionable error when experimental artifacts are missing', async () => {
    const {
      ExperimentalArtifactMissingError,
      isExperimentalArtifactMissingError,
      loadSpeechModel,
    } = await import('@asrjs/speech-recognition');

    await expect(
      loadSpeechModel({ family: 'gigaam-rnnt', modelId: 'gigaam-v3-e2e-rnnt', backend: 'wasm' }),
    ).rejects.toMatchObject({
      name: 'ExperimentalArtifactMissingError',
      code: 'experimental-artifact-missing',
    });

    try {
      await loadSpeechModel({ modelId: 'Qwen/Qwen3-ASR-0.6B', backend: 'wasm' });
      expect.unreachable('Qwen load without source should fail');
    } catch (error) {
      expect(isExperimentalArtifactMissingError(error)).toBe(true);
      expect(error).toBeInstanceOf(ExperimentalArtifactMissingError);
      expect((error as ExperimentalArtifactMissingError).code).toBe('experimental-artifact-missing');
      expect((error as Error).message).toContain('listExperimentalSpeechFamilies()');
      expect((error as Error).message).toContain('short-clip');
      expect((error as Error).message).toContain('local ONNX');
      expect((error as { details?: { audioContract?: string } }).details?.audioContract).toBe(
        'short-clip-speech-llm',
      );
    }
  });
});
