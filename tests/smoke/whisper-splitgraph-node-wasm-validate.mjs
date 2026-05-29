#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import { initWhisperOrt } from '../../dist/models/whisper-seq2seq/ort.js';
import { parseWhisperManifest } from '../../dist/models/whisper-seq2seq/manifest.js';
import { parseWhisperGenerationConfig, parseWhisperModelConfig } from '../../dist/models/whisper-seq2seq/generation-config.js';
import { WhisperTokenizer } from '../../dist/models/whisper-seq2seq/tokenizer.js';
import { WhisperMelProcessor } from '../../dist/audio/whisper-mel.js';
import { WhisperTimestampLogitProcessor } from '../../dist/models/whisper-seq2seq/processors.js';
import { processSplitGraphAlignment } from '../../dist/models/whisper-seq2seq/executor.js';

const AUDIO_EXTENSIONS = new Set(['.wav']);
const GRAPH_FILES = {
  encoder: 'encoder_model.onnx',
  decoder_init: 'decoder_init.onnx',
  decoder_step: 'decoder_step.onnx',
  decoder_align: 'decoder_align.onnx',
};

function parseArgs(argv) {
  const args = {
    modelDir: process.env.WHISPER_VARIANT_DIR ?? '/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph',
    fixtures: process.env.WHISPER_FIXTURES ?? 'tests/fixtures',
    variants: ['fp32', 'fp16', 'q8'],
    report: process.env.WHISPER_REPORT ?? 'docs/reports/whisper-large-v3-turbo-variant-validation.md',
    maxNewTokens: Number(process.env.WHISPER_MAX_NEW_TOKENS ?? 64),
    align: process.env.WHISPER_VALIDATE_ALIGN !== '0',
    strict: process.env.WHISPER_VALIDATE_STRICT !== '0',
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--model-dir') { args.modelDir = next; i++; }
    else if (arg === '--fixtures') { args.fixtures = next; i++; }
    else if (arg === '--report') { args.report = next; i++; }
    else if (arg === '--max-new-tokens') { args.maxNewTokens = Number(next); i++; }
    else if (arg === '--variants') {
      args.variants = [];
      while (argv[i + 1] && !argv[i + 1].startsWith('--')) args.variants.push(argv[++i]);
    } else if (arg === '--no-align') {
      args.align = false;
    } else if (arg === '--no-strict') {
      args.strict = false;
    }
  }
  return args;
}

function inferLanguageFromFilename(filename) {
  const withoutAudioExt = filename.replace(/\.[^.]+$/, '');
  const suffix = withoutAudioExt.split('.').at(-1);
  if (suffix === 'tr' || suffix === 'en') return suffix;
  return 'en';
}

function discoverFixtures(fixturesDir) {
  return fs.readdirSync(fixturesDir)
    .filter((name) => AUDIO_EXTENSIONS.has(path.extname(name).toLowerCase()))
    .sort()
    .map((name) => {
      const filePath = path.join(fixturesDir, name);
      const sidecar = filePath.replace(/\.[^.]+$/, '.json');
      let referenceText = '';
      if (fs.existsSync(sidecar)) {
        referenceText = JSON.parse(fs.readFileSync(sidecar, 'utf8')).text ?? '';
      }
      return {
        filename: name,
        path: filePath,
        language: inferLanguageFromFilename(name),
        referenceText,
        sizeBytes: fs.statSync(filePath).size,
      };
    });
}

function readWavMono(pathname) {
  const buf = fs.readFileSync(pathname);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`Not a RIFF/WAVE file: ${pathname}`);
  }
  let offset = 12;
  let fmt = null;
  let data = null;
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (id === 'fmt ') {
      fmt = {
        format: buf.readUInt16LE(start),
        channels: buf.readUInt16LE(start + 2),
        sampleRate: buf.readUInt32LE(start + 4),
        bitsPerSample: buf.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      data = buf.subarray(start, start + size);
    }
    offset = start + size + (size % 2);
  }
  if (!fmt || !data) throw new Error(`WAV missing fmt/data chunk: ${pathname}`);
  const bytes = fmt.bitsPerSample / 8;
  const frames = Math.floor(data.length / bytes / fmt.channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let ch = 0; ch < fmt.channels; ch++) {
      const p = (i * fmt.channels + ch) * bytes;
      if (fmt.format === 3 && fmt.bitsPerSample === 32) sum += data.readFloatLE(p);
      else if (fmt.bitsPerSample === 16) sum += data.readInt16LE(p) / 32768;
      else if (fmt.bitsPerSample === 24) sum += data.readIntLE(p, 3) / 8388608;
      else if (fmt.bitsPerSample === 32) sum += data.readInt32LE(p) / 2147483648;
      else throw new Error(`Unsupported WAV bit depth: ${fmt.bitsPerSample}`);
    }
    out[i] = sum / fmt.channels;
  }
  return { samples: fmt.sampleRate === 16000 ? out : resampleLinear(out, fmt.sampleRate, 16000), sampleRate: 16000, originalSampleRate: fmt.sampleRate };
}

function resampleLinear(input, sourceRate, targetRate) {
  const ratio = targetRate / sourceRate;
  const output = new Float32Array(Math.max(1, Math.floor(input.length * ratio)));
  for (let i = 0; i < output.length; i++) {
    const x = i / ratio;
    const x0 = Math.floor(x);
    const x1 = Math.min(input.length - 1, x0 + 1);
    const t = x - x0;
    output[i] = (input[x0] ?? 0) * (1 - t) + (input[x1] ?? 0) * t;
  }
  return output;
}

function argmax(arr) {
  let idx = 0;
  let val = arr[0] ?? -Infinity;
  for (let i = 1; i < arr.length; i++) {
    const next = arr[i] ?? -Infinity;
    if (next > val) { idx = i; val = next; }
  }
  return idx;
}

function float32ToFloat16Bits(values) {
  const out = new Uint16Array(values.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < values.length; i++) {
    f32[0] = values[i] ?? 0;
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    let mantissa = x & 0x7fffff;
    let exp = (x >>> 23) & 0xff;
    if (exp === 0xff) {
      out[i] = sign | (mantissa ? 0x7e00 : 0x7c00);
    } else if (exp > 142) {
      out[i] = sign | 0x7c00;
    } else if (exp < 113) {
      if (exp < 103) out[i] = sign;
      else {
        mantissa |= 0x800000;
        const shift = 125 - exp;
        out[i] = sign | ((mantissa + (1 << (shift - 1))) >> shift);
      }
    } else {
      exp = exp - 112;
      mantissa = mantissa + 0x1000;
      if (mantissa & 0x800000) { mantissa = 0; exp++; }
      out[i] = sign | (exp << 10) | (mantissa >> 13);
    }
  }
  return out;
}

function float16BitsToFloat32(values) {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const h = values[i] ?? 0;
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >>> 10) & 0x1f;
    const frac = h & 0x03ff;
    if (exp === 0) {
      out[i] = frac === 0 ? sign * 0 : sign * 2 ** -14 * (frac / 1024);
    } else if (exp === 0x1f) {
      out[i] = frac === 0 ? sign * Infinity : NaN;
    } else {
      out[i] = sign * 2 ** (exp - 15) * (1 + frac / 1024);
    }
  }
  return out;
}

function tensorDataAsFloat32(tensor) {
  return tensor.type === 'float16' ? float16BitsToFloat32(tensor.data) : tensor.data;
}

function readJsonIfExists(file) {
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : {};
}

function graphExternalData(variantDir, manifestRaw, graphName) {
  const graph = manifestRaw.artifacts?.[graphName];
  const entries = Array.isArray(graph?.externalData) ? graph.externalData : [];
  return entries.map((entry) => ({
    path: String(entry.path ?? entry.file),
    data: path.join(variantDir, String(entry.file ?? entry.path)),
  }));
}

async function createSession(ort, variantDir, manifestRaw, graphName, runtimeBackend) {
  const modelPath = path.join(variantDir, GRAPH_FILES[graphName]);
  const sessionOptions = runtimeBackend === 'node-cpu'
    ? {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all',
      }
    : {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'parallel',
        enableCpuMemArena: true,
        enableMemPattern: true,
        externalData: graphExternalData(variantDir, manifestRaw, graphName),
      };
  if (sessionOptions.externalData?.length === 0) delete sessionOptions.externalData;
  return ort.InferenceSession.create(modelPath, sessionOptions);
}

function buildPromptIds(tokenizer, language) {
  return [
    tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
    tokenizer.getTokenId(`<|${language}|>`) ?? (language === 'tr' ? 50268 : 50259),
    tokenizer.getTokenId('<|transcribe|>') ?? 50360,
    tokenizer.getTokenId('<|notimestamps|>') ?? 50364,
  ];
}

async function loadVariant(ortWasm, modelDir, variant) {
  const variantDir = path.join(modelDir, variant);
  const runtimeBackend = variant === 'q8' ? 'wasm' : 'node-cpu';
  const ort = runtimeBackend === 'node-cpu' ? await import('onnxruntime-node') : ortWasm;
  const manifestRaw = readJsonIfExists(path.join(variantDir, 'manifest.json'));
  const manifest = parseWhisperManifest(manifestRaw);
  const generationRaw = readJsonIfExists(path.join(variantDir, 'generation_config.json'));
  const configRaw = readJsonIfExists(path.join(variantDir, 'config.json'));
  const generationConfig = parseWhisperGenerationConfig(generationRaw);
  const modelConfig = parseWhisperModelConfig(configRaw);
  const tokenizer = await WhisperTokenizer.fromUrl(`file://${path.join(variantDir, 'tokenizer.json')}`);
  const t0 = performance.now();
  return { variant, runtimeBackend, ort, variantDir, manifestRaw, manifest, generationRaw, generationConfig, modelConfig, tokenizer, loadMs: performance.now() - t0 };
}

function makeTensor(ort, type, data, dims) {
  return new ort.Tensor(type, data, dims);
}

function releaseSession(session) {
  if (session && typeof session.release === 'function') session.release();
}

async function runFixture(variantState, fixture, maxNewTokens, enableAlign) {
  const { samples, originalSampleRate } = readWavMono(fixture.path);
  const numMelBins = variantState.manifestRaw.num_mel_bins ?? variantState.modelConfig.numMelBins ?? 128;
  const manifestMaxSrcPos = variantState.manifestRaw.max_source_positions ?? 1500;
  const inputFrames = manifestMaxSrcPos <= 1500 ? manifestMaxSrcPos * 2 : manifestMaxSrcPos;
  const melProc = new WhisperMelProcessor({ nMels: numMelBins });
  const mel = WhisperMelProcessor.padToFrames(melProc.process(samples), inputFrames);
  const ort = variantState.ort;
  const encoderInput = variantState.variant === 'fp16'
    ? makeTensor(ort, 'float16', float32ToFloat16Bits(mel), [1, numMelBins, inputFrames])
    : makeTensor(ort, 'float32', mel, [1, numMelBins, inputFrames]);

  const language = fixture.language;
  const promptIds = buildPromptIds(variantState.tokenizer, language);
  const eosId = variantState.tokenizer.getTokenId('<|endoftext|>') ?? 50257;
  const noTsId = variantState.generationConfig.noTimestampsTokenId ?? variantState.tokenizer.getTokenId('<|notimestamps|>') ?? 50364;
  const timestampBegin = variantState.tokenizer.getTokenId('<|0.00|>') ?? 50365;
  const controls = {
    language,
    task: 'transcribe',
    no_timestamps: true,
    max_new_tokens: maxNewTokens,
    suppress_tokens: variantState.generationConfig.suppressTokens ?? [],
    begin_suppress_tokens: variantState.generationConfig.beginSuppressTokens ?? [],
    temperature: 0,
    decoding: 'greedy',
    num_beams: 1,
  };
  const processor = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId,
    noTimestampsTokenId: noTsId,
    timestampBegin,
    suppressTokens: controls.suppress_tokens,
    beginSuppressTokens: controls.begin_suppress_tokens,
  });

  const t0 = performance.now();
  const encoderSession = await createSession(ort, variantState.variantDir, variantState.manifestRaw, 'encoder', variantState.runtimeBackend);
  const encOuts = await encoderSession.run({ input_features: encoderInput });
  releaseSession(encoderSession);
  const enc = encOuts[Object.keys(encOuts)[0]];
  const encoderDims = Array.from(enc.dims ?? []);

  const promptTensor = makeTensor(ort, 'int64', new BigInt64Array(promptIds.map(BigInt)), [1, promptIds.length]);
  const initSession = await createSession(ort, variantState.variantDir, variantState.manifestRaw, 'decoder_init', variantState.runtimeBackend);
  const initOut = await initSession.run({ input_ids: promptTensor, encoder_hidden_states: enc });
  releaseSession(initSession);
  const initKeys = Object.keys(initOut);
  const logitsKey = initKeys.find((k) => k.includes('logits')) ?? initKeys[0];
  const initLogitsTensor = initOut[logitsKey];
  const vocabSize = initLogitsTensor.dims.at(-1);
  const initLogitsData = tensorDataAsFloat32(initLogitsTensor);
  const firstLogits = initLogitsData.slice(initLogitsData.length - vocabSize);
  processor.process(firstLogits, promptIds, promptIds.length);
  let nextToken = argmax(firstLogits);
  const tokens = [nextToken];
  let eosReached = nextToken === eosId;

  const pastKv = {};
  for (const key of initKeys) {
    if (key.startsWith('present')) pastKv[key.replace(/^present\./, 'past_key_values.')] = initOut[key];
  }

  const stepSession = await createSession(ort, variantState.variantDir, variantState.manifestRaw, 'decoder_step', variantState.runtimeBackend);
  for (let step = 1; step < maxNewTokens && !eosReached; step++) {
    const feeds = { input_ids: makeTensor(ort, 'int64', new BigInt64Array([BigInt(nextToken)]), [1, 1]) };
    for (const [key, value] of Object.entries(pastKv)) feeds[key] = value;
    const stepOut = await stepSession.run(feeds);
    const stepKeys = Object.keys(stepOut);
    const stepLogitsKey = stepKeys.find((k) => k.includes('logits')) ?? stepKeys[0];
    const logits = tensorDataAsFloat32(stepOut[stepLogitsKey]);
    processor.process(logits, [...promptIds, ...tokens], promptIds.length);
    nextToken = argmax(logits);
    tokens.push(nextToken);
    eosReached = nextToken === eosId;
    for (const key of stepKeys) {
      if (key.startsWith('present')) pastKv[key.replace(/^present\./, 'past_key_values.')] = stepOut[key];
    }
  }

  releaseSession(stepSession);

  const decodedText = variantState.tokenizer.decode(tokens, { skipSpecialTokens: true }).trim();
  let alignment = null;
  if (enableAlign && fs.existsSync(path.join(variantState.variantDir, GRAPH_FILES.decoder_align)) && tokens.length > 0) {
    const alignSession = await createSession(ort, variantState.variantDir, variantState.manifestRaw, 'decoder_align', variantState.runtimeBackend);
    const allTokens = [...promptIds, ...tokens];
    const alignOut = await alignSession.run({
      input_ids: makeTensor(ort, 'int64', new BigInt64Array(allTokens.map(BigInt)), [1, allTokens.length]),
      encoder_hidden_states: enc,
    });
    const alignTensor = alignOut[Object.keys(alignOut)[0]];
    const alignData = tensorDataAsFloat32(alignTensor);
    const [batch, textSteps, frames] = alignTensor.dims;
    let rowSumMin = Infinity;
    let rowSumMax = -Infinity;
    let rowSumTotal = 0;
    let nonNegative = true;
    for (let t = 0; t < textSteps; t++) {
      let sum = 0;
      for (let f = 0; f < frames; f++) {
        const value = alignData[t * frames + f] ?? 0;
        if (value < 0) nonNegative = false;
        sum += value;
      }
      rowSumMin = Math.min(rowSumMin, sum);
      rowSumMax = Math.max(rowSumMax, sum);
      rowSumTotal += sum;
    }
    const dtw = processSplitGraphAlignment({
      alignmentData: alignData,
      totalTokens: allTokens.length,
      promptLen: promptIds.length,
      textTokenCount: tokens.length,
      frameCount: frames,
      timePrecisionSeconds: 0.02,
    });
    let monotonic = true;
    for (let i = 1; i < dtw.length; i++) {
      if (dtw[i] < dtw[i - 1]) monotonic = false;
    }
    alignment = {
      shape: [batch, textSteps, frames],
      row_sum_min: Number(rowSumMin.toFixed(4)),
      row_sum_mean: Number((rowSumTotal / textSteps).toFixed(4)),
      row_sum_max: Number(rowSumMax.toFixed(4)),
      non_negative: nonNegative,
      dtw_monotonic: monotonic,
      dtw_count: dtw.length,
    };
    releaseSession(alignSession);
  }

  return {
    filename: fixture.filename,
    language,
    original_sample_rate: originalSampleRate,
    prompt_ids: promptIds,
    generation_controls: controls,
    runtime_backend: variantState.runtimeBackend,
    encoder_shape: encoderDims,
    tokens,
    token_count: tokens.length,
    decoded_text: decodedText,
    eos_reached: eosReached,
    alignment,
    duration_sec: Number(((performance.now() - t0) / 1000).toFixed(3)),
  };
}

function tokenMatch(a, b) {
  const n = Math.max(a.length, b.length);
  if (n === 0) return { exact: true, ratio: 1, matches: 0, compared: 0 };
  let matches = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] === b[i]) matches++;
  return { exact: a.length === b.length && matches === a.length, ratio: matches / n, matches, compared: n };
}

function compareResults(allResults) {
  const baseline = allResults.find((r) => r.variant === 'fp32');
  if (!baseline) throw new Error('fp32 baseline is required');
  const byFixture = new Map(baseline.fixtures.map((f) => [f.filename, f]));
  for (const variantResult of allResults) {
    for (const fixture of variantResult.fixtures) {
      const base = byFixture.get(fixture.filename);
      if (!base) continue;
      fixture.comparison_to_fp32 = tokenMatch(base.tokens, fixture.tokens);
      fixture.prompt_matches_fp32 = JSON.stringify(base.prompt_ids) === JSON.stringify(fixture.prompt_ids);
      fixture.controls_match_fp32 = JSON.stringify(base.generation_controls) === JSON.stringify(fixture.generation_controls);
      fixture.text_matches_fp32 = base.decoded_text === fixture.decoded_text;
    }
  }
}

function assertValidationPasses(allResults) {
  const failures = [];
  for (const result of allResults) {
    for (const fixture of result.fixtures) {
      if (!fixture.prompt_matches_fp32) failures.push(`${result.variant}/${fixture.filename}: prompt mismatch vs fp32`);
      if (!fixture.controls_match_fp32) failures.push(`${result.variant}/${fixture.filename}: generation controls mismatch vs fp32`);
      if (fixture.comparison_to_fp32 && !fixture.comparison_to_fp32.exact && !fixture.text_matches_fp32) failures.push(`${result.variant}/${fixture.filename}: token and text mismatch vs fp32 (${fixture.comparison_to_fp32.matches}/${fixture.comparison_to_fp32.compared})`);
      if (fixture.alignment) {
        if (!fixture.alignment.non_negative) failures.push(`${result.variant}/${fixture.filename}: alignment has negative values`);
        if (!fixture.alignment.dtw_monotonic) failures.push(`${result.variant}/${fixture.filename}: DTW timestamps not monotonic`);
        if (fixture.alignment.row_sum_min < 0.99 || fixture.alignment.row_sum_max > 1.01) failures.push(`${result.variant}/${fixture.filename}: alignment row sums outside tolerance`);
      }
    }
  }
  if (failures.length > 0) throw new Error(`Validation failed:\n${failures.join('\n')}`);
}

function mdEscape(value) {
  return String(value).replace(/\|/g, '\\|').replace(/\n/g, ' ');
}

function generateReport({ modelDir, fixtures, results, maxNewTokens, align }) {
  const lines = [];
  lines.push('# Whisper Large v3 Turbo — Node/WASM Splitgraph Validation Report');
  lines.push('');
  lines.push(`**Generated**: ${new Date().toISOString()}`);
  lines.push(`**Artifacts**: ${modelDir}`);
  lines.push('**Backend**: Node CLI (`fp32`/`fp16` use onnxruntime-node CPU because these large variants exceed WASM memory on this host; `q8` uses onnxruntime-web WASM CPU)');
  lines.push(`**max_new_tokens**: ${maxNewTokens}`);
  lines.push(`**Alignment validation**: ${align ? 'enabled' : 'disabled'}`);
  lines.push('');
  lines.push('## Scope');
  lines.push('');
  lines.push('- Validates existing fp32/fp16/q8 splitgraph variants locally in Node CLI.');
  lines.push('- Uses fp32 as the Node CPU baseline; fp16 is also checked on Node CPU, and q8 is validated with the WASM CPU execution provider. No WebGPU/browser automation is included.');
  lines.push('- Uses language suffixes from fixture filenames: `.tr.*` → Turkish, `.en.*` → English.');
  lines.push('- Decoding path is greedy `temperature=0`; beam search is not implemented here.');
  lines.push('');
  lines.push('## Fixtures');
  lines.push('');
  lines.push('| Fixture | Language | Reference |');
  lines.push('|---------|----------|-----------|');
  for (const fixture of fixtures) lines.push(`| ${mdEscape(fixture.filename)} | ${fixture.language} | ${fixture.referenceText ? 'yes' : 'no'} |`);
  lines.push('');
  lines.push('## Generation Controls');
  lines.push('');
  lines.push('| Fixture | Variant | Language | Task | no_timestamps | max_new_tokens | suppress_tokens | begin_suppress_tokens | Decoding |');
  lines.push('|---------|---------|----------|------|---------------|----------------|-----------------|-----------------------|----------|');
  for (const result of results) {
    for (const f of result.fixtures) {
      const c = f.generation_controls;
      lines.push(`| ${mdEscape(f.filename)} | ${result.variant} | ${c.language} | ${c.task} | ${c.no_timestamps} | ${c.max_new_tokens} | ${c.suppress_tokens.length} tokens | [${c.begin_suppress_tokens.join(', ')}] | ${c.decoding}, temp=${c.temperature} |`);
    }
  }
  lines.push('');
  lines.push('## Prompt Consistency');
  lines.push('');
  lines.push('| Fixture | Prompt language | fp32 prompt IDs | fp16 match | q8 match |');
  lines.push('|---------|-----------------|-----------------|------------|----------|');
  const resultMap = new Map(results.map((r) => [r.variant, r]));
  for (const fixture of fixtures) {
    const fp32 = resultMap.get('fp32')?.fixtures.find((f) => f.filename === fixture.filename);
    const fp16 = resultMap.get('fp16')?.fixtures.find((f) => f.filename === fixture.filename);
    const q8 = resultMap.get('q8')?.fixtures.find((f) => f.filename === fixture.filename);
    lines.push(`| ${mdEscape(fixture.filename)} | ${fixture.language} | [${fp32?.prompt_ids.join(', ') ?? ''}] | ${fp16?.prompt_matches_fp32 ? 'yes' : 'NO'} | ${q8?.prompt_matches_fp32 ? 'yes' : 'NO'} |`);
  }
  lines.push('');
  lines.push('## Token/Text Comparison vs fp32');
  lines.push('');
  lines.push('| Fixture | Variant | Tokens | EOS | Token match vs fp32 | Text match | Decoded text | Time |');
  lines.push('|---------|---------|--------|-----|---------------------|------------|--------------|------|');
  for (const result of results) {
    for (const f of result.fixtures) {
      const cmp = f.comparison_to_fp32;
      const match = cmp ? `${cmp.exact ? 'exact' : 'DIFF'} (${cmp.matches}/${cmp.compared}, ${(cmp.ratio * 100).toFixed(1)}%)` : 'baseline';
      lines.push(`| ${mdEscape(f.filename)} | ${result.variant} | ${f.token_count} | ${f.eos_reached} | ${match} | ${f.text_matches_fp32 ?? 'baseline'} | ${mdEscape(f.decoded_text.slice(0, 100))} | ${f.duration_sec}s |`);
    }
  }
  lines.push('');
  lines.push('## Alignment/DTW Validation');
  lines.push('');
  lines.push('| Fixture | Variant | Shape | Row sums min/mean/max | Non-negative | Monotonic DTW |');
  lines.push('|---------|---------|-------|-----------------------|--------------|---------------|');
  for (const result of results) {
    for (const f of result.fixtures) {
      const a = f.alignment;
      lines.push(`| ${mdEscape(f.filename)} | ${result.variant} | ${a ? `[${a.shape.join(', ')}]` : 'n/a'} | ${a ? `${a.row_sum_min}/${a.row_sum_mean}/${a.row_sum_max}` : 'n/a'} | ${a?.non_negative ?? 'n/a'} | ${a?.dtw_monotonic ?? 'n/a'} |`);
    }
  }
  lines.push('');
  lines.push('## Status Summary');
  lines.push('');
  lines.push('| Variant | Node CLI | Runtime backend | Prompt parity | Token parity vs fp32 | Alignment sanity | Status |');
  lines.push('|---------|----------|-----------------|---------------|----------------------|------------------|--------|');
  for (const result of results) {
    const promptOk = result.fixtures.every((f) => f.prompt_matches_fp32);
    const tokenOk = result.fixtures.every((f) => !f.comparison_to_fp32 || f.comparison_to_fp32.exact);
    const alignOk = result.fixtures.every((f) => !f.alignment || (f.alignment.non_negative && f.alignment.dtw_monotonic && f.alignment.row_sum_min >= 0.99 && f.alignment.row_sum_max <= 1.01));
    const runtimeBackend = result.fixtures[0]?.runtime_backend ?? 'unknown';
    lines.push(`| ${result.variant} | pass | ${runtimeBackend} | ${promptOk ? 'pass' : 'fail'} | ${result.variant === 'fp32' ? 'baseline' : tokenOk ? 'pass' : 'fail'} | ${alignOk ? 'pass' : 'fail'} | ${promptOk && tokenOk && alignOk ? 'pass' : 'fail'} |`);
  }
  lines.push('');
  lines.push('## Deferred / Manual');
  lines.push('');
  lines.push('Current Node CLI validation is intentionally strict: prompt/generation-control parity passes, but any token/text/EOS divergence is reported before WebGPU is attempted.');
  lines.push('');
  lines.push('fp16 parity requires converting float16 logits/alignment tensors back to float32 before logit processors and argmax; raw uint16 half bits are not comparable logits.');
  lines.push('');
  lines.push('q8 uses ONNX Runtime Web WASM CPU. Extended greedy decoding can diverge from fp32 because the decoder is quantized; those token/EOS differences remain visible in the comparison table instead of being hidden.');
  lines.push('');
  lines.push('WebGPU smoke is intentionally not automated here. After Node/WASM validation passes, WebGPU should be tested manually in the browser/app.');
  lines.push('');
  lines.push('Beam search for the 4-graph splitgraph runtime is not implemented in this validation pass; keep it as the next decoding task after greedy parity is stable.');
  lines.push('');
  lines.push('Mixed dtype, q4/q4f16, exporter changes, browser automation, and published HF artifact changes are out of scope for this report.');
  lines.push('');
  return `${lines.join('\n')}\n`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const modelDir = path.resolve(args.modelDir);
  const fixturesDir = path.resolve(args.fixtures);
  const report = path.resolve(args.report);
  const fixtures = discoverFixtures(fixturesDir);
  if (fixtures.length === 0) throw new Error(`No fixtures found in ${fixturesDir}`);
  console.log(`Fixtures: ${fixtures.map((f) => `${f.filename}:${f.language}`).join(', ')}`);
  const ort = await initWhisperOrt('wasm');
  const results = [];
  for (const variant of args.variants) {
    console.log(`\n== ${variant} ==`);
    const state = await loadVariant(ort, modelDir, variant);
    const variantResult = { variant, load_sec: Number((state.loadMs / 1000).toFixed(3)), fixtures: [] };
    for (const fixture of fixtures) {
      console.log(`  ${fixture.filename} (${fixture.language})`);
      variantResult.fixtures.push(await runFixture(state, fixture, args.maxNewTokens, args.align));
    }
    results.push(variantResult);
  }
  compareResults(results);
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(report, generateReport({ modelDir, fixtures, results, maxNewTokens: args.maxNewTokens, align: args.align }));
  const jsonPath = report.replace(/\.md$/, '.json');
  fs.writeFileSync(jsonPath, JSON.stringify({ modelDir, fixtures, results, maxNewTokens: args.maxNewTokens, align: args.align }, null, 2));
  if (args.strict) assertValidationPasses(results);
  console.log(`\nReport: ${report}`);
  console.log(`JSON:   ${jsonPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
