import { describe, expect, it } from 'vitest';
import { canaryBuiltInLocalModelAdapter } from '../src/presets/canary/local-adapter.js';
import { resolveBuiltInLocalModelAdapter } from '../src/runtime/local-adapter-registry.js';
import { CANARY_1B_V2_LANGUAGES, CANARY_1B_V2_VOCABULARY_SIZE } from '../src/presets/canary/manifest.js';

const MODEL_ID = 'nvidia/canary-1b-v2';
const sidecar = 'encoder-model.onnx.data';
const REQUIRED_PROMPT_TOKENS = [
  '<|startofcontext|>', '<|startoftranscript|>', '<|emo:undefined|>', '<|pnc|>', '<|nopnc|>',
  '<|itn|>', '<|noitn|>', '<|timestamp|>', '<|notimestamp|>', '<|diarize|>', '<|nodiarize|>',
  '<|endoftext|>', '<|nospeech|>', '<pad>', ...CANARY_1B_V2_LANGUAGES.map((language) => `<|${language}|>`),
];

function v2SpecialTokens() {
  const tokens = Object.fromEntries(REQUIRED_PROMPT_TOKENS.map((token, index) => [token, index + 4]));
  tokens['<pad>'] = 0;
  tokens['<|startoftranscript|>'] = 1;
  tokens['<|endoftext|>'] = 2;
  return tokens;
}

function bundle(overrides: { config?: Record<string, unknown>; tokenizer?: Record<string, unknown>; sidecars?: string[] } = {}) {
  const config = {
    model_id: MODEL_ID,
    family: 'nemo-aed',
    preset: 'canary',
    classification: { ecosystem: 'nemo', family: 'canary' },
    sample_rate: 16000,
    mel_bins: 128,
    frame_shift_seconds: 0.01,
    subsampling_factor: 8,
    encoder_hidden_size: 1024,
    decoder_hidden_size: 1024,
    encoder_output_size: 1024,
    max_target_positions: 1024,
    languages: [...CANARY_1B_V2_LANGUAGES],
    prompt_format: 'canary2',
    prompt_defaults: [{ source_lang: 'en', target_lang: 'en', decoder_context: '', emotion: '<|emo:undefined|>', pnc: true, itn: false, timestamp: false, diarize: false }],
    preprocessor_valid_length_mode: 'onnx',
    preprocessor_normalization: 'per_feature',
    external_data: { encoder: sidecar, decoder: null },
    ...overrides.config,
  };
  const vocabSize = CANARY_1B_V2_VOCABULARY_SIZE;
  const specialTokens = v2SpecialTokens();
  const pieces = Array.from({ length: vocabSize }, (_, index) => `p${index}`);
  for (const [token, id] of Object.entries(specialTokens)) pieces[id] = token;
  const tokenizer = {
    kind: 'canary-aggregate-tokenizer',
    version: 1,
    prompt_format: 'canary2',
    vocab_size: vocabSize,
    langs: ['canary'],
    language_codes: [...CANARY_1B_V2_LANGUAGES],
    bos_id: 1,
    eos_id: 2,
    pad_id: 0,
    special_tokens: specialTokens,
    subtokenizers: { canary: { offset: 0, size: vocabSize, pieces } },
    ...overrides.tokenizer,
  };
  return [
    ['encoder-model.onnx', 'graph'],
    ['decoder-model.onnx', 'graph'],
    ['tokenizer.json', JSON.stringify(tokenizer)],
    ['config.json', JSON.stringify(config)],
    ...(overrides.sidecars ?? [sidecar]).map((name) => [name, 'external data'] as [string, string]),
  ].map(([name, contents]) => ({ path: `bundle/${name}`, basename: name, file: new File([contents], name) }));
}

function aggregate180mBundle() {
  const modelId = 'nvidia/canary-180m-flash';
  const languages = ['en', 'de', 'es', 'fr'];
  const tokenIds: Record<string, number> = {
    '<pad>': 2, '<|endoftext|>': 3, '<|startoftranscript|>': 4, '<|pnc|>': 5, '<|nopnc|>': 6,
    '<|startofcontext|>': 7, '<|itn|>': 8, '<|noitn|>': 9, '<|timestamp|>': 10,
    '<|notimestamp|>': 11, '<|diarize|>': 12, '<|nodiarize|>': 13, '<|emo:undefined|>': 16,
    '<|en|>': 62, '<|de|>': 64, '<|es|>': 66, '<|fr|>': 76, '<|nospeech|>': 14,
  };
  const specialTokens = tokenIds;
  const vocabSize = 5248;
  const specials = Array.from({ length: 80 }, (_, index) => `<special:${index}>`);
  for (const [token, id] of Object.entries(specialTokens)) specials[id] = token;
  const perLanguage = (vocabSize - specials.length) / languages.length;
  const subtokenizers: Record<string, { offset: number; size: number; pieces: string[] }> = {
    spl_tokens: { offset: 0, size: specials.length, pieces: specials },
  };
  let offset = specials.length;
  for (const language of languages) {
    subtokenizers[language] = {
      offset,
      size: perLanguage,
      pieces: Array.from({ length: perLanguage }, (_, index) => `${language}-${index}`),
    };
    offset += perLanguage;
  }
  const config = {
    model_id: modelId, family: 'nemo-aed', preset: 'canary',
    classification: { ecosystem: 'nemo', family: 'canary' }, sample_rate: 16000, mel_bins: 128,
    frame_shift_seconds: 0.01, subsampling_factor: 8, encoder_hidden_size: 1024,
    decoder_hidden_size: 1024, encoder_output_size: 1024, max_target_positions: 1024,
    languages, prompt_format: 'canary2',
    prompt_defaults: [{ source_lang: 'en', target_lang: 'en', decoder_context: '', emotion: '<|emo:undefined|>', pnc: true, itn: false, timestamp: false, diarize: false }],
    preprocessor_valid_length_mode: 'centered', preprocessor_normalization: 'per_feature',
    external_data: { encoder: null, decoder: null },
  };
  const tokenizer = {
    kind: 'canary-aggregate-tokenizer', version: 1, prompt_format: 'canary2', vocab_size: vocabSize,
    langs: ['spl_tokens', ...languages], language_codes: languages, bos_id: 4, eos_id: 3, pad_id: 2,
    special_tokens: specialTokens, subtokenizers,
  };
  return [
    ['encoder-model.onnx', 'graph'], ['decoder-model.onnx', 'graph'],
    ['tokenizer.json', JSON.stringify(tokenizer)], ['config.json', JSON.stringify(config)],
  ].map(([name, contents]) => ({ path: `canary-180m/${name}`, basename: name, file: new File([contents], name) }));
}

function legacyCanary1bBundle() {
  const modelId = 'nvidia/canary-1b';
  const languages = ['en', 'de', 'es', 'fr'];
  const vocabSize = 4128;
  const specialTokens: Record<string, number> = {
    '<pad>': 1,
    '<|endoftext|>': 2,
    '<|startoftranscript|>': 3,
    '<|en|>': 4,
    '<|es|>': 5,
    '<|de|>': 6,
    '<|fr|>': 7,
    '<|transcribe|>': 8,
    '<|translate|>': 9,
    '<|nopnc|>': 10,
    '<|pnc|>': 11,
    '<|nospeech|>': 12,
    ...Object.fromEntries(Array.from({ length: 18 }, (_, i) => [`<|spltoken${i}|>`, 13 + i])),
  };
  const specialPieces = Array.from({ length: 32 }, (_, index) => `reserved-${index}`);
  for (const [token, id] of Object.entries(specialTokens)) specialPieces[id] = token;
  const subtokenizers: Record<string, { offset: number; size: number; pieces: string[] }> = {
    spl_tokens: { offset: 0, size: 32, pieces: specialPieces },
  };
  let offset = 32;
  for (const language of languages) {
    subtokenizers[language] = {
      offset,
      size: 1024,
      pieces: Array.from({ length: 1024 }, (_, index) => `${language}-${index}`),
    };
    offset += 1024;
  }
  const config = {
    model_id: modelId,
    family: 'nemo-aed',
    preset: 'canary',
    classification: { ecosystem: 'nemo', family: 'canary' },
    sample_rate: 16000,
    mel_bins: 128,
    frame_shift_seconds: 0.01,
    subsampling_factor: 8,
    encoder_hidden_size: 1024,
    decoder_hidden_size: 1024,
    encoder_output_size: 1024,
    max_target_positions: 512,
    languages,
    prompt_format: 'canary',
    prompt_defaults: [{ source_lang: 'en', target_lang: 'en', decoder_context: '', task: 'asr', pnc: true }],
    preprocessor_valid_length_mode: 'onnx',
    preprocessor_normalization: 'per_feature',
    external_data: { encoder: null, decoder: null },
  };
  const tokenizer = {
    kind: 'canary-aggregate-tokenizer',
    version: 1,
    prompt_format: 'canary',
    vocab_size: vocabSize,
    langs: ['spl_tokens', ...languages],
    language_codes: languages,
    bos_id: 3,
    eos_id: 2,
    pad_id: 1,
    special_tokens: specialTokens,
    subtokenizers,
  };
  return [
    ['encoder-model.onnx', 'graph'],
    ['decoder-model.onnx', 'graph'],
    ['tokenizer.json', JSON.stringify(tokenizer)],
    ['config.json', JSON.stringify(config)],
  ].map(([name, contents]) => ({ path: `canary-1b/${name}`, basename: name, file: new File([contents], name) }));
}

describe('Canary local adapter', () => {
  it('accepts the 180M aggregate tokenizer split into contiguous per-language ranges', async () => {
    const resolved = await canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: 'nvidia/canary-180m-flash', entries: aggregate180mBundle() } as never);
    await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    expect((resolved.builtInLoadOptions as any).config).toMatchObject({ vocabularySize: 5248, languages: ['en', 'de', 'es', 'fr'] });
  });

  it('registers the preset and resolves a metadata-matched bundle to direct local artifacts', async () => {
    expect(resolveBuiltInLocalModelAdapter(MODEL_ID).adapter).toBe(canaryBuiltInLocalModelAdapter);
    const resolved = await canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle() } as never);
    try {
      expect(resolved.selection).toEqual({ encoderName: 'encoder-model.onnx', decoderName: 'decoder-model.onnx', tokenizerName: 'tokenizer.json', encoderQuant: 'fp32', decoderQuant: 'fp32' });
      const options = resolved.builtInLoadOptions as any;
      expect(options.config).toMatchObject({ vocabularySize: CANARY_1B_V2_VOCABULARY_SIZE, melBins: 128, preprocessorValidLengthMode: 'onnx', preprocessorNormalization: 'per_feature', languages: CANARY_1B_V2_LANGUAGES });
      expect(options.source).toMatchObject({ kind: 'direct', preprocessorBackend: 'js', artifacts: {
        encoderUrl: expect.stringMatching(/^blob:/), decoderUrl: expect.stringMatching(/^blob:/),
        tokenizerUrl: expect.stringMatching(/^blob:/), configUrl: expect.stringMatching(/^blob:/),
        encoderDataUrl: expect.stringMatching(/^blob:/), encoderFilename: 'encoder-model.onnx', decoderFilename: 'decoder-model.onnx',
      } });
      expect(options.source.artifacts.decoderDataUrl).toBeUndefined();
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('accepts the original Canary prompt format and its 4,128-token checkpoint contract', async () => {
    const resolved = await canaryBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'nvidia/canary-1b',
      entries: legacyCanary1bBundle(),
    } as never);
    try {
      const config = (resolved.builtInLoadOptions as any).config;
      expect(config).toMatchObject({
        vocabularySize: 4128,
        languages: ['en', 'de', 'es', 'fr'],
        encoderHiddenSize: 1024,
        decoderHiddenSize: 1024,
        encoderOutputSize: 1024,
        maxTargetPositions: 512,
        promptFormat: 'canary',
        promptDefaults: [{ task: 'asr', sourceLanguage: 'en', targetLanguage: 'en' }],
      });
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('rejects a prompt format that differs from the tokenizer and selected preset', async () => {
    const entries = await Promise.all(legacyCanary1bBundle().map(async (entry) => {
      if (entry.basename !== 'config.json') return entry;
      const metadata = JSON.parse(await (entry.file as File).text());
      metadata.prompt_format = 'canary2';
      return {
        ...entry,
        file: new File([JSON.stringify(metadata)], entry.basename),
      };
    }));
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'nvidia/canary-1b', entries,
    } as never)).rejects.toThrow(/prompt format must match the selected preset/);
  });

  it('allows inline graphs when metadata declares no sidecars and requires an optional ONNX frontend when selected', async () => {
    const entries = bundle({ config: { external_data: { encoder: null, decoder: null } }, sidecars: [] });
    const resolved = await canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries } as never);
    try {
      expect((resolved.builtInLoadOptions as any).source.artifacts).not.toHaveProperty('encoderDataUrl');
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries, preprocessorBackend: 'onnx' } as never)).rejects.toThrow(/nemo128.onnx is missing/);
  });

  it('round-trips the inspected logical nemo128 preprocessor selection into the local ONNX artifact', async () => {
    const entries = [...bundle({ config: { external_data: { encoder: null, decoder: null } }, sidecars: [] }), {
      path: 'bundle/nemo128.onnx', basename: 'nemo128.onnx', file: new File(['frontend'], 'nemo128.onnx'),
    }];
    expect(canaryBuiltInLocalModelAdapter.inspectEntries(entries).preprocessorNames).toEqual(['nemo128']);
    const resolved = await canaryBuiltInLocalModelAdapter.resolveEntries({
      modelId: MODEL_ID, entries, preprocessorBackend: 'onnx', preprocessorName: 'nemo128',
    } as never);
    try {
      expect(resolved.selection.preprocessorName).toBe('nemo128.onnx');
      expect((resolved.builtInLoadOptions as any).source.artifacts.preprocessorUrl).toMatch(/^blob:/);
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('rejects wrong identity, vocab, frontend dimensions, unsafe or missing declared sidecars, and duplicate files', async () => {
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ config: { model_id: 'nvidia/canary-1b' } }) } as never)).rejects.toThrow(/model_id/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ tokenizer: { vocab_size: 5248 } }) } as never)).rejects.toThrow(/vocabulary/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ config: { mel_bins: 80 } }) } as never)).rejects.toThrow(/mel-bin/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ config: { frame_shift_seconds: Number.POSITIVE_INFINITY } }) } as never)).rejects.toThrow(/runtime dimensions/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ config: { subsampling_factor: 8.5 } }) } as never)).rejects.toThrow(/runtime dimensions/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ tokenizer: { special_tokens: { '<|en|>': 3 } } }) } as never)).rejects.toThrow(/required prompt tokens/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ tokenizer: { special_tokens: { ...v2SpecialTokens(), '<|startofcontext|>': 5 } } }) } as never)).rejects.toThrow(/does not point to its token piece/);
    const omittedSpecial = await Promise.all(bundle().map(async (entry) => {
      if (entry.basename !== 'tokenizer.json') return entry;
      const payload = JSON.parse(await (entry.file as File).text());
      payload.subtokenizers.canary.pieces[CANARY_1B_V2_VOCABULARY_SIZE - 1] = '<|unlisted|>';
      return { ...entry, file: new File([JSON.stringify(payload)], entry.basename) };
    }));
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: omittedSpecial } as never)).rejects.toThrow(/does not cover every special token/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ tokenizer: { language_codes: [...CANARY_1B_V2_LANGUAGES, 'fr'] } }) } as never)).rejects.toThrow(/language metadata/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ config: { external_data: { encoder: '../outside.data', decoder: null } } }) } as never)).rejects.toThrow(/external-data path/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle({ sidecars: [] }) } as never)).rejects.toThrow(/Missing declared external data/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: bundle(), encoderQuant: 'int8' } as never)).rejects.toThrow(/fp32 graphs only/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: [...bundle(), { path: '../escape', basename: 'escape', file: new File(['x'], 'escape') }] } as never)).rejects.toThrow(/Unsafe local Canary artifact path/);
    const splitRoot = bundle().map((entry) => entry.basename === 'config.json' ? { ...entry, path: `other/${entry.basename}` } : entry);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: splitRoot } as never)).rejects.toThrow(/share one bundle directory/);
    const splitSidecar = bundle().map((entry) => entry.basename === sidecar ? { ...entry, path: `other/${entry.basename}` } : entry);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: splitSidecar } as never)).rejects.toThrow(/bundle directory/);
    const gapRange = await Promise.all(bundle().map(async (entry) => {
      if (entry.basename !== 'tokenizer.json') return entry;
      const payload = JSON.parse(await (entry.file as File).text());
      payload.subtokenizers.canary.offset = 1;
      return { ...entry, file: new File([JSON.stringify(payload)], entry.basename) };
    }));
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: gapRange } as never)).rejects.toThrow(/cover the full vocabulary/);
    await expect(canaryBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries: [...bundle(), ...bundle()] } as never)).rejects.toThrow(/Multiple local Canary artifacts/);
  });
});
