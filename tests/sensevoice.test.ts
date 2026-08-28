import { describe, expect, it } from 'vitest';
import {
  SenseVoiceJsPreprocessor,
  SenseVoiceTokenizer,
  SenseVoiceSession,
  createSenseVoicePrompt,
  resolveSenseVoiceLanguage,
} from '../src/models/sensevoice/index.js';
import { createBuiltInSpeechRuntime } from '../src/runtime/builtins.js';
import { loadSpeechModel } from '../src/runtime/load.js';

describe('SenseVoice prompt contract', () => {
  it('is discoverable as a built-in model family without a fake preset', () => {
    const runtime = createBuiltInSpeechRuntime({ useManifestSources: false });
    const family = runtime
      .listModelFamilies()
      .find((candidate) => candidate.family === 'sensevoice');
    expect(family?.supports('OpenVoiceOS/SenseVoiceSmall')).toBe(true);
    expect(family?.supports('nvidia/parakeet-tdt-0.6b-v3')).toBe(false);
  });

  it('fails artifact-free loading instead of returning a scaffold transcript', async () => {
    await expect(
      loadSpeechModel({ family: 'sensevoice', modelId: 'SenseVoiceSmall', backend: 'wasm' }),
    ).rejects.toThrow(/No SenseVoice artifact source/);
  });

  it('maps supported languages and ITN IDs to the ONNX prompt contract', () => {
    expect(createSenseVoicePrompt({ language: 'en' })).toEqual({
      language: 'en',
      languageId: 4,
      textnorm: 'withitn',
      textnormId: 14,
    });
    expect(createSenseVoicePrompt({ language: 'ja', useItn: false })).toMatchObject({
      languageId: 11,
      textnormId: 15,
    });
  });

  it('falls back safely for unsupported language values', () => {
    expect(resolveSenseVoiceLanguage('tr')).toBe('auto');
    expect(resolveSenseVoiceLanguage(undefined)).toBe('auto');
  });
});

describe('SenseVoice tokenizer and frontend', () => {
  it('exposes a batch session with the same response-flavor contract as single transcription', async () => {
    const calls: number[] = [];
    const session = new SenseVoiceSession(
      'sensevoice-test',
      { family: 'sensevoice' },
      {} as never,
      'wasm',
      {
        async transcribeBatch(audio) {
          calls.push(audio.length);
          return audio.map(() => ({ utteranceText: 'ok', isFinal: true }));
        },
        async transcribe() {
          return { utteranceText: 'ok', isFinal: true };
        },
        dispose() {},
      },
    );
    const result = await session.transcribeBatch([
      {
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: 160,
        durationSeconds: 0.01,
        channels: [new Float32Array(160)],
      },
      {
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: 160,
        durationSeconds: 0.01,
        channels: [new Float32Array(160)],
      },
    ]);
    expect(calls).toEqual([2]);
    expect(result.map((item) => item.text)).toEqual(['ok', 'ok']);

    const native = await session.transcribeBatch([], { responseFlavor: 'native' });
    expect(native).toEqual([]);
    expect(calls).toEqual([2]);

    const envelope = await session.transcribeBatch(
      [
        {
          sampleRate: 16000,
          numberOfChannels: 1,
          numberOfFrames: 160,
          durationSeconds: 0.01,
          channels: [new Float32Array(160)],
        },
      ],
      { responseFlavor: 'canonical+native' },
    );
    expect(envelope[0]?.canonical.text).toBe('ok');
    expect(envelope[0]?.native?.utteranceText).toBe('ok');
    await session.dispose();
  });

  it('decodes SentencePiece pieces while dropping CTC blank and prompt tags', () => {
    const tokenizer = SenseVoiceTokenizer.fromText('<blk> 0\n<|en|> 1\n▁hello 2\n▁world 3\n');
    expect(tokenizer.blankId).toBe(0);
    expect(tokenizer.decode([1, 2, 3, 0])).toBe('hello world');
  });

  it('emits raw 80-bin time-major fbank frames', () => {
    const processor = new SenseVoiceJsPreprocessor();
    const audio = new Float32Array(16000);
    for (let index = 0; index < audio.length; index += 1) {
      audio[index] = Math.sin((2 * Math.PI * 220 * index) / 16000) * 0.1;
    }
    const result = processor.process(audio);
    expect(result.featureSize).toBe(80);
    expect(result.frameCount).toBeGreaterThan(0);
    expect(result.validFrameCount).toBe(result.frameCount);
    expect(result.features.length).toBe(result.frameCount * 80);
  });
});
