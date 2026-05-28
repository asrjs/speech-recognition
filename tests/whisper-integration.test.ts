import { describe, expect, it, vi } from 'vitest';
import {
  WHISPER_PRESET_MANIFESTS,
  resolveWhisperArtifactSource,
  resolveWhisperPresetManifest,
} from '../src/presets/whisper/manifest.js';
import { createWhisperPresetFactory } from '../src/presets/whisper/factory.js';
import {
  WhisperTokenizer,
  createWhisperSeq2SeqModelFamily,
} from '../src/models/whisper-seq2seq/index.js';
import { WhisperMelProcessor } from '../src/audio/whisper-mel.js';
import { PcmAudioBuffer } from '../src/audio/index.js';

describe('Whisper preset manifests', () => {
  it('includes onnx-community sources for all presets', () => {
    for (const manifest of WHISPER_PRESET_MANIFESTS) {
      expect(manifest.source).toBeDefined();
      expect(manifest.source?.kind).toBe('huggingface');
      if (manifest.source?.kind === 'huggingface') {
        expect(manifest.source.repoId).toMatch(/^onnx-community\/whisper-/);
      }
    }
  });

  it('default preset is whisper-base', async () => {
    const factory = createWhisperPresetFactory({ useManifestSource: true });
    const request = await factory.resolveModelRequest(
      { preset: 'whisper', options: {} },
      {} as never,
    );
    expect(request.modelId).toBe('onnx-community/whisper-base');
    expect(request.options?.source).toBeDefined();
  });

  it('resolves artifact source for known aliases', () => {
    expect(resolveWhisperArtifactSource('whisper-tiny')).toBeDefined();
    expect(resolveWhisperArtifactSource('whisper-base')).toBeDefined();
    expect(resolveWhisperArtifactSource('whisper-small')).toBeDefined();
    expect(resolveWhisperArtifactSource('whisper-large-v3-turbo')).toBeDefined();
  });

  it('manifest config matches expected model sizes', () => {
    const tiny = resolveWhisperPresetManifest('whisper-tiny');
    expect(tiny?.config.melBins).toBe(80);

    const base = resolveWhisperPresetManifest('whisper-base');
    expect(base?.config.melBins).toBe(80);

    const small = resolveWhisperPresetManifest('whisper-small');
    expect(small?.config.melBins).toBe(80);

    const large = resolveWhisperPresetManifest('whisper-large-v3-turbo');
    expect(large?.config.melBins).toBe(128);
    expect(large?.config.vocabularySize).toBe(51866);
  });
});

describe('Whisper tokenizer', () => {
  const mockTokenizerJson = {
    model: {
      type: 'BPE',
      vocab: {
        '!': 0,
        '"': 1,
        '<|endoftext|>': 50257,
      },
      merges: [],
    },
    added_tokens: [
      { id: 50257, content: '<|endoftext|>', special: true },
      { id: 50258, content: '<|startoftranscript|>', special: true },
      { id: 50259, content: '<|en|>', special: true },
      { id: 50260, content: '<|zh|>', special: true },
      { id: 50268, content: '<|tr|>', special: true },
      { id: 50358, content: '<|translate|>', special: true },
      { id: 50359, content: '<|transcribe|>', special: true },
      { id: 50363, content: '<|notimestamps|>', special: true },
      { id: 50364, content: '<|0.00|>', special: true },
      { id: 51864, content: '<|30.00|>', special: true },
    ],
  };

  it('loads from JSON and maps special tokens', () => {
    const tokenizer = new WhisperTokenizer(mockTokenizerJson);
    expect(tokenizer.getTokenId('<|startoftranscript|>')).toBe(50258);
    expect(tokenizer.getTokenId('<|tr|>')).toBe(50268);
    expect(tokenizer.getTokenId('<|transcribe|>')).toBe(50359);
    expect(tokenizer.getTokenId('<|notimestamps|>')).toBe(50363);
  });

  it('detects timestamp tokens correctly', () => {
    const tokenizer = new WhisperTokenizer(mockTokenizerJson);
    expect(tokenizer.isTimestampTokenId(50364)).toBe(true);
    expect(tokenizer.isTimestampTokenId(51864)).toBe(true);
    expect(tokenizer.isTimestampTokenId(50363)).toBe(false);
    expect(tokenizer.isTimestampTokenId(50258)).toBe(false);
  });

  it('converts timestamp token IDs to seconds', () => {
    const tokenizer = new WhisperTokenizer(mockTokenizerJson);
    expect(tokenizer.timestampTokenIdToSeconds(50364)).toBe(0);
    expect(tokenizer.timestampTokenIdToSeconds(51864)).toBe(30);
  });

  it('decodes with basic GPT-2 cleanup', () => {
    const vocab: Record<string, number> = {};
    for (let i = 0; i < 128; i++) {
      vocab[String.fromCharCode(i)] = i;
    }
    vocab.hello = 200;
    vocab['\u0120world'] = 201;

    const tokenizer = new WhisperTokenizer({
      model: { type: 'BPE', vocab, merges: [] },
      added_tokens: [],
    });

    expect(tokenizer.decode([200, 201])).toBe('hello world');
  });
});

describe('Whisper mel processor', () => {
  it('processes 1 second of 16kHz mono audio', () => {
    const sampleRate = 16000;
    const durationSec = 1.0;
    const samples = new Float32Array(sampleRate * durationSec);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const processor = new WhisperMelProcessor({ nMels: 80 });
    const result = processor.process(samples);

    expect(result.nMels).toBe(80);
    expect(result.frameCount).toBeGreaterThan(0);
    expect(result.features.length).toBe(80 * result.frameCount);
  });

  it('pads features to target frame count', () => {
    const sampleRate = 16000;
    const samples = new Float32Array(sampleRate * 2);
    const processor = new WhisperMelProcessor({ nMels: 80 });
    const result = processor.process(samples);

    const padded = WhisperMelProcessor.padToFrames(result, 1500);
    expect(padded.length).toBe(80 * 1500);
  });

  it('returns empty result for empty audio', () => {
    const processor = new WhisperMelProcessor({ nMels: 80 });
    const result = processor.process(new Float32Array(0));
    expect(result.frameCount).toBe(0);
    expect(result.features.length).toBe(0);
  });
});

describe('Whisper model family', () => {
  it('creates a model factory that supports whisper ids', () => {
    const family = createWhisperSeq2SeqModelFamily();
    expect(family.supports('whisper-base')).toBe(true);
    expect(family.supports('openai/whisper-tiny')).toBe(true);
    expect(family.supports('parakeet-tdt-0.6b')).toBe(false);
  });

  it('stub session returns expected scaffold output when no source is provided', async () => {
    const family = createWhisperSeq2SeqModelFamily();
    const model = await family.createModel(
      {
        family: 'whisper-seq2seq',
        modelId: 'whisper-base',
        classification: {},
        options: {},
      },
      {
        backend: { id: 'wasm' },
        hooks: { logger: { info: vi.fn() } },
      } as never,
    );

    const session = await model.createSession();
    const audio = PcmAudioBuffer.fromMono(new Float32Array(16000), 16000);
    const result = await session.transcribe(audio, { responseFlavor: 'native' });

    expect(result.utteranceText).toBe('Whisper seq2seq scaffold');
    expect(result.warnings?.some((w) => w.code === 'whisper-seq2seq.stubbed-decoder')).toBe(true);
    expect(result.segments?.length).toBe(3);

    session.dispose();
    await model.dispose();
  });
});

describe('Whisper regression', () => {
  it('stub output must not accidentally match real transcript format', async () => {
    const family = createWhisperSeq2SeqModelFamily();
    const model = await family.createModel(
      {
        family: 'whisper-seq2seq',
        modelId: 'whisper-base',
        classification: {},
        options: {},
      },
      {
        backend: { id: 'wasm' },
        hooks: { logger: { info: vi.fn() } },
      } as never,
    );

    const session = await model.createSession();
    const audio = PcmAudioBuffer.fromMono(new Float32Array(16000), 16000);
    const result = await session.transcribe(audio, { responseFlavor: 'native' });

    // This test ensures the old stub output is detectable.
    // When real ONNX is wired, this assertion should be updated.
    const isStubOutput = result.utteranceText === 'Whisper seq2seq scaffold';
    expect(isStubOutput).toBe(true);

    session.dispose();
    await model.dispose();
  });
});
