import {
  buildSpeechModelLoadOptions,
  buildSpeechTranscriptionOptions,
  getSpeechModelDescriptor,
  getSpeechModelLanguageName,
  listSpeechModelOptions,
  listSpeechModels,
  resolveSpeechModelComponentBackends,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

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
});
