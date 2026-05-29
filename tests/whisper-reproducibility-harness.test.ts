import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { resolveWhisperArtifacts, initWhisperOrt, createWhisperOrtSession } from '../src/models/whisper-seq2seq/ort.js';
import { parseWhisperManifest } from '../src/models/whisper-seq2seq/manifest.js';
import { WhisperTokenizer } from '../src/models/whisper-seq2seq/tokenizer.js';
import { WhisperMelProcessor } from '../src/audio/whisper-mel.js';
import { WhisperTimestampLogitProcessor } from '../src/models/whisper-seq2seq/processors.js';
import { processSplitGraphAlignment } from '../src/models/whisper-seq2seq/executor.js';
import type { OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';
import type { WhisperExecutionBackend } from '../src/models/whisper-seq2seq/types.js';

function argmax(arr: Float32Array): number {
  let maxIdx = 0, maxVal = arr[0] ?? -Infinity;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? -Infinity) > maxVal) { maxVal = arr[i]!; maxIdx = i; }
  }
  return maxIdx;
}

/**
 * Normalize token sequences for EOS comparison.
 *
 * PyTorch generate() appends EOS to the output (token 50257 at the end).
 * Our TypeScript loop stops ON EOS without appending it.
 * Both are equivalent — treat [tokens..., EOS] and [tokens...] as matching
 * if both stopped because EOS was predicted.
 */
function normalizeEos(tokens: number[], eosTokenId: number, eosWasStopReason: boolean): number[] {
  if (eosWasStopReason && tokens[tokens.length - 1] !== eosTokenId) {
    // TS stopped without appending EOS — append it for comparison
    return [...tokens, eosTokenId];
  }
  return tokens;
}

function compareTokens(
  tsTokens: number[],
  refTokens: number[],
  eosTokenId: number,
  tsEosStop: boolean,
  refHasEos: boolean,
): { matches: number; minLen: number; matchPct: number; mismatches: string[] } {
  const tsNorm = normalizeEos(tsTokens, eosTokenId, tsEosStop);
  const refNorm = refHasEos ? refTokens : normalizeEos(refTokens, eosTokenId, true);

  const minLen = Math.min(tsNorm.length, refNorm.length);
  let matches = 0;
  const mismatches: string[] = [];
  for (let i = 0; i < minLen; i++) {
    if (tsNorm[i] === refNorm[i]) {
      matches++;
    } else {
      mismatches.push(`  [${i}] TS=${tsNorm[i]} vs REF=${refNorm[i]}`);
    }
  }

  if (tsNorm.length !== refNorm.length) {
    mismatches.push(`  Length: TS=${tsNorm.length}, REF=${refNorm.length}`);
  }

  return {
    matches,
    minLen,
    matchPct: minLen > 0 ? (100 * matches / minLen) : 0,
    mismatches,
  };
}

interface ReferenceJson {
  audio: { path: string; sample_rate: number; duration_seconds: number; num_samples: number };
  model: { id: string; export_dir: string; format: string; d_model: number; decoder_layers: number; decoder_attention_heads: number };
  prompt_ids: number[];
  pytorch: {
    no_timestamps: { tokens: number[]; text: string };
    with_timestamps: { tokens: number[]; text: string; timestamp_tokens: number[] };
  };
  onnx_python: {
    tokens: number[];
    generated: number[];
    alignment: { shape: number[]; text_shape: number[]; row_sum_min: number; row_sum_max: number; row_sum_mean: number } | null;
  };
  mel_features_path?: string;
}

describe('Whisper splitgraph reproducibility harness (vs HF Transformers)', () => {
  it('feature-input mode: 100% token match using Python mel features', async () => {
    const refPath = process.env.WHISPER_REFERENCE_JSON;
    if (!refPath) { console.warn('Skipping: set WHISPER_REFERENCE_JSON'); return; }
    if (!fs.existsSync(refPath)) { console.warn(`Skipping: ${refPath} not found`); return; }

    const ref: ReferenceJson = JSON.parse(fs.readFileSync(refPath, 'utf-8'));
    const melPath = ref.mel_features_path;
    if (!melPath || !fs.existsSync(melPath)) {
      console.warn('Skipping feature-input mode: no mel_features_path in reference (use --export-mel)');
      return;
    }

    const { ort, encSess, initSess, stepSess, alignSess, tokenizer, manifestRaw, manifest }
      = await setupSessions(ref);

    const { dModel, decoderLayers } = manifest.modelConfig;
    const numMelBins = (manifestRaw.num_mel_bins as number) ?? 80;
    const maxSrcPos = (manifestRaw.max_source_positions as number) ?? 3000;
    const vocabSize = (manifestRaw.vocab_size as number) ?? 51865;

    // ── Load Python mel features directly (bypass TS mel processor) ──
    const melNpy = fs.readFileSync(melPath);
    // Parse NPY header: magic (6 bytes) + version (2 bytes) + header_len (2 bytes LE) + header dict
    const headerLen = new DataView(melNpy.buffer, melNpy.byteOffset + 8, 2).getUint16(0, true);
    const headerStr = new TextDecoder().decode(melNpy.subarray(10, 10 + headerLen));
    // Extract shape from header like: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 80, 3000)}
    const shapeMatch = headerStr.match(/shape':\s*\(([^)]+)\)/);
    if (!shapeMatch) throw new Error('Failed to parse NPY shape header');
    const dataOffset = 10 + headerLen;
    const melData = new Float32Array(melNpy.buffer, melNpy.byteOffset + dataOffset);

    console.log(`Loaded Python mel features: ${melData.length} floats, shape ${shapeMatch[1]}`);

    const { allTokens, generated, encOut } = await runDecodeLoop({
      ort, encSess, initSess, stepSess, tokenizer,
      melData, numMelBins, maxSrcPos, dModel, vocabSize,
      promptIds: ref.prompt_ids, decoderLayers,
      suppressTokens: getSuppressTokens(manifestRaw),
      beginSuppressTokens: getBeginSuppressTokens(manifestRaw),
    });

    // ═══════════════════════════════════════════════════════
    // FEATURE-INPUT MODE: require 100% token match
    // ═══════════════════════════════════════════════════════
    const refTokens = ref.onnx_python.tokens;
    const decoded = tokenizer.decode(allTokens, { skipSpecialTokens: true });
    const refDecoded = tokenizer.decode(refTokens, { skipSpecialTokens: true });
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const tsEosStop = generated[generated.length - 1] === eosId
      || allTokens[allTokens.length - 1] === eosId;
    const refHasEos = refTokens[refTokens.length - 1] === eosId;

    const cmp = compareTokens(allTokens, refTokens, eosId, tsEosStop, refHasEos);

    console.log('\n=== Feature-Input Reproducibility (Python mel) ===');
    console.log(`TypeScript tokens (${allTokens.length}): ${allTokens.slice(0, 20)}${allTokens.length > 20 ? '...' : ''}`);
    console.log(`Ref tokens       (${refTokens.length}): ${refTokens.slice(0, 20)}${refTokens.length > 20 ? '...' : ''}`);
    console.log(`Token match: ${cmp.matches}/${cmp.minLen} (${cmp.matchPct.toFixed(1)}%)`);
    console.log(`TS text:  "${decoded.substring(0, 100)}"`);
    console.log(`Ref text: "${refDecoded.substring(0, 100)}"`);

    if (cmp.matchPct < 100) {
      console.log('Mismatches (first 10):');
      for (const m of cmp.mismatches.slice(0, 10)) console.log(m);
    }

    // With identical mel features and identical ONNX graphs, tokens MUST match 100%
    expect(cmp.matchPct).toBe(100);

    // Text should match after normalization
    const tsText = decoded.trim().toLowerCase().replace(/\s+/g, ' ');
    const pyText = ref.pytorch.no_timestamps.text.trim().toLowerCase().replace(/\s+/g, ' ');
    expect(tsText).toBe(pyText);

    // Alignment validation
    await validateAlignment(ort, alignSess, allTokens, encOut, generated, ref, maxSrcPos);

    console.log('\nFeature-input reproducibility PASSED (100% token match)');
  }, 180000);

  it('wav-input mode: >=80% token match using TS mel frontend', async () => {
    const refPath = process.env.WHISPER_REFERENCE_JSON;
    if (!refPath) { console.warn('Skipping: set WHISPER_REFERENCE_JSON'); return; }
    if (!fs.existsSync(refPath)) { console.warn(`Skipping: ${refPath} not found`); return; }

    const ref: ReferenceJson = JSON.parse(fs.readFileSync(refPath, 'utf-8'));
    const audioPath = ref.audio.path;
    if (!fs.existsSync(audioPath)) { console.warn(`Skipping: audio not found at ${audioPath}`); return; }

    const { ort, encSess, initSess, stepSess, alignSess, tokenizer, manifestRaw, manifest }
      = await setupSessions(ref);

    const { dModel, decoderLayers } = manifest.modelConfig;
    const numMelBins = (manifestRaw.num_mel_bins as number) ?? 80;
    const maxSrcPos = (manifestRaw.max_source_positions as number) ?? 3000;
    const vocabSize = (manifestRaw.vocab_size as number) ?? 51865;

    // ── Load audio and run TS mel processor ──
    const audioBuf = fs.readFileSync(audioPath);
    const audioSamples = new Float32Array(
      new Uint8Array(audioBuf.buffer, audioBuf.byteOffset + 44, (audioBuf.length - 44)).buffer,
    );

    const melProc = new WhisperMelProcessor({ nMels: numMelBins });
    const melResult = melProc.process(audioSamples);
    const paddedMel = WhisperMelProcessor.padToFrames(melResult, maxSrcPos);

    const { allTokens, generated, encOut } = await runDecodeLoop({
      ort, encSess, initSess, stepSess, tokenizer,
      melData: paddedMel, numMelBins, maxSrcPos, dModel, vocabSize,
      promptIds: ref.prompt_ids, decoderLayers,
      suppressTokens: getSuppressTokens(manifestRaw),
      beginSuppressTokens: getBeginSuppressTokens(manifestRaw),
    });

    // ═══════════════════════════════════════════════════════
    // WAV-INPUT MODE: allow >=80% (mel frontend may differ)
    // ═══════════════════════════════════════════════════════
    const refTokens = ref.onnx_python.tokens;
    const decoded = tokenizer.decode(allTokens, { skipSpecialTokens: true });
    const refDecoded = tokenizer.decode(refTokens, { skipSpecialTokens: true });
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const tsEosStop = generated[generated.length - 1] === eosId
      || allTokens[allTokens.length - 1] === eosId;
    const refHasEos = refTokens[refTokens.length - 1] === eosId;

    const cmp = compareTokens(allTokens, refTokens, eosId, tsEosStop, refHasEos);

    console.log('\n=== WAV-Input Reproducibility (TS mel) ===');
    console.log(`TypeScript tokens (${allTokens.length}): ${allTokens.slice(0, 20)}${allTokens.length > 20 ? '...' : ''}`);
    console.log(`Ref tokens       (${refTokens.length}): ${refTokens.slice(0, 20)}${refTokens.length > 20 ? '...' : ''}`);
    console.log(`Token match: ${cmp.matches}/${cmp.minLen} (${cmp.matchPct.toFixed(1)}%)`);
    console.log(`TS text:  "${decoded.substring(0, 100)}"`);
    console.log(`Ref text: "${refDecoded.substring(0, 100)}"`);

    if (cmp.matchPct < 100) {
      console.log('Mismatches (first 5):');
      for (const m of cmp.mismatches.slice(0, 5)) console.log(m);
    }

    // With TS mel frontend, allow >=80% due to possible frontend differences
    expect(cmp.matchPct).toBeGreaterThanOrEqual(80);

    // No timestamp tokens in no_timestamps mode
    const tsBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const hasTimestamps = generated.some((t) => t >= tsBegin);
    expect(hasTimestamps).toBe(false);

    // Alignment validation
    await validateAlignment(ort, alignSess, allTokens, encOut, generated, ref, maxSrcPos);

    console.log('\nWAV-input reproducibility PASSED');
  }, 180000);
});

// ── Helpers ────────────────────────────────────────────────────

function getSuppressTokens(manifestRaw: Record<string, unknown>): number[] {
  const st = manifestRaw.special_tokens as Record<string, unknown> | undefined;
  return Array.isArray(st?.suppress_tokens) ? st.suppress_tokens as number[] : [];
}

function getBeginSuppressTokens(manifestRaw: Record<string, unknown>): number[] {
  const st = manifestRaw.special_tokens as Record<string, unknown> | undefined;
  return Array.isArray(st?.begin_suppress_tokens) ? st.begin_suppress_tokens as number[] : [];
}

async function setupSessions(ref: ReferenceJson) {
  const modelDir = ref.model.export_dir;
  const f = (...parts: string[]) => path.join(modelDir, ...parts);
  const required = ['encoder_model.onnx', 'decoder_init.onnx', 'decoder_step.onnx', 'tokenizer.json', 'manifest.json'];
  if (!required.every((n) => fs.existsSync(f(n)))) throw new Error(`Missing model files in ${modelDir}`);

  const manifestRaw = JSON.parse(fs.readFileSync(f('manifest.json'), 'utf-8')) as Record<string, unknown>;
  const manifest = parseWhisperManifest(manifestRaw);

  expect(manifest.modelConfig.dModel).toBe(ref.model.d_model);
  expect(manifest.modelConfig.decoderLayers).toBe(ref.model.decoder_layers);
  expect(manifest.modelConfig.decoderAttentionHeads).toBe(ref.model.decoder_attention_heads);

  const source = {
    kind: 'splitgraph' as const,
    artifacts: {
      encoderUrl: `file://${f('encoder_model.onnx')}`,
      decoderInitUrl: `file://${f('decoder_init.onnx')}`,
      decoderStepUrl: `file://${f('decoder_step.onnx')}`,
      decoderAlignUrl: fs.existsSync(f('decoder_align.onnx')) ? `file://${f('decoder_align.onnx')}` : undefined,
      tokenizerUrl: `file://${f('tokenizer.json')}`,
      manifestUrl: `file://${f('manifest.json')}`,
    },
  };
  const resolved = resolveWhisperArtifacts(source, 'wasm');
  const ort = await initWhisperOrt('wasm');
  const be: WhisperExecutionBackend = 'wasm';
  const encSess = await createWhisperOrtSession(ort, resolved.artifacts.encoderUrl, { backendId: be });
  const initSess = await createWhisperOrtSession(ort, resolved.decoderInitUrl!, { backendId: be });
  const stepSess = await createWhisperOrtSession(ort, resolved.decoderStepUrl!, { backendId: be });
  const hasAlign = fs.existsSync(f('decoder_align.onnx'));
  const alignSess = hasAlign ? await createWhisperOrtSession(ort, resolved.decoderAlignUrl!, { backendId: be }) : null;
  const tokenizer = await WhisperTokenizer.fromUrl(resolved.artifacts.tokenizerUrl);

  return { ort, encSess, initSess, stepSess, alignSess, tokenizer, manifestRaw, manifest };
}

async function runDecodeLoop(params: {
  ort: Awaited<ReturnType<typeof initWhisperOrt>>;
  encSess: Awaited<ReturnType<typeof createWhisperOrtSession>>;
  initSess: Awaited<ReturnType<typeof createWhisperOrtSession>>;
  stepSess: Awaited<ReturnType<typeof createWhisperOrtSession>>;
  tokenizer: WhisperTokenizer;
  melData: Float32Array;
  numMelBins: number;
  maxSrcPos: number;
  dModel: number;
  vocabSize: number;
  promptIds: number[];
  decoderLayers: number;
  suppressTokens: number[];
  beginSuppressTokens: number[];
}) {
  const { ort, encSess, initSess, stepSess, tokenizer, melData, numMelBins, maxSrcPos, dModel, vocabSize, promptIds, decoderLayers, suppressTokens, beginSuppressTokens } = params;

  const melTensor = new ort.Tensor('float32', melData, [1, numMelBins, maxSrcPos]);
  const encOut = (await encSess.run({ input_features: melTensor }))['last_hidden_state'] as OrtTensorLike<Float32Array>;
  expect(encOut.dims).toEqual([1, maxSrcPos, dModel]);

  const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
  const noTsId = tokenizer.getTokenId('<|notimestamps|>') ?? 50363;
  const tsBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;

  const processor = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId, noTimestampsTokenId: noTsId, timestampBegin: tsBegin,
    suppressTokens, beginSuppressTokens,
  });

  // Init
  const promptArr = new BigInt64Array(promptIds.map((id) => BigInt(id)));
  const initOut = await initSess.run({
    input_ids: new ort.Tensor('int64', promptArr, [1, promptIds.length]),
    encoder_hidden_states: encOut,
  });
  const logitsKey = Object.keys(initOut).find((k) => k.includes('logits'))!;
  const initLogits = (initOut[logitsKey] as OrtTensorLike<Float32Array>).data;
  const tsVocabSize = (initOut[logitsKey] as OrtTensorLike<Float32Array>).dims[2] ?? 51865;
  expect(tsVocabSize).toBe(vocabSize);

  const pastKv: Record<string, OrtTensorLike<Float32Array>> = {};
  for (const [k, v] of Object.entries(initOut)) {
    if (k.startsWith('present')) pastKv[k.replace('present.', 'past_key_values.')] = v as OrtTensorLike<Float32Array>;
  }
  expect(Object.keys(pastKv).length).toBe(4 * decoderLayers);

  // First token
  const lastOffset = initLogits.length - tsVocabSize;
  const firstLogits = initLogits.subarray(lastOffset);
  processor.process(firstLogits, promptIds, promptIds.length);
  let nextToken = argmax(firstLogits);
  const generated: number[] = [nextToken];

  // Step loop
  for (let s = 1; s < 128; s++) {
    const stepIn = new BigInt64Array([BigInt(nextToken)]);
    const stepFeeds: Record<string, unknown> = { input_ids: new ort.Tensor('int64', stepIn, [1, 1]) };
    for (const [k, v] of Object.entries(pastKv)) stepFeeds[k] = v;
    const stepOut = await stepSess.run(stepFeeds);
    const sLogitsKey = Object.keys(stepOut).find((k) => k.includes('logits'))!;
    const sLogits = (stepOut[sLogitsKey] as OrtTensorLike<Float32Array>).data;
    processor.process(sLogits, [...promptIds, ...generated], promptIds.length);
    nextToken = argmax(sLogits);
    if (nextToken === eosId) break;
    generated.push(nextToken);
    for (const [k, v] of Object.entries(stepOut)) {
      if (k.startsWith('present')) pastKv[k.replace('present.', 'past_key_values.')] = v as OrtTensorLike<Float32Array>;
    }
  }

  return { allTokens: [...promptIds, ...generated], generated, encOut };
}

async function validateAlignment(
  ort: Awaited<ReturnType<typeof initWhisperOrt>>,
  alignSess: Awaited<ReturnType<typeof createWhisperOrtSession>> | null,
  allTokens: number[], encOut: OrtTensorLike<Float32Array>,
  generated: number[], ref: ReferenceJson, maxSrcPos: number,
) {
  if (!alignSess || generated.length === 0) return;

  const alignArr = new BigInt64Array(allTokens.map((id) => BigInt(id)));
  const alignOut = await alignSess.run({
    input_ids: new ort.Tensor('int64', alignArr, [1, allTokens.length]),
    encoder_hidden_states: encOut,
  });
  const alignKey = Object.keys(alignOut)[0]!;
  const alignment = alignOut[alignKey] as OrtTensorLike<Float32Array>;
  const [aB, aT, aS] = alignment.dims;

  expect(aB).toBe(1);
  expect(aS).toBe(maxSrcPos);

  const data = alignment.data;
  for (let ti = 0; ti < aT; ti++) {
    let sum = 0;
    for (let si = 0; si < aS; si++) sum += data[ti * aS + si] ?? 0;
    expect(sum).toBeGreaterThan(0.99);
    expect(sum).toBeLessThan(1.01);
  }

  const dtwTs = processSplitGraphAlignment({
    alignmentData: data, totalTokens: allTokens.length,
    promptLen: ref.prompt_ids.length, textTokenCount: generated.length,
    frameCount: aS, timePrecisionSeconds: 0.02,
  });
  expect(dtwTs).toHaveLength(generated.length + 1);
  for (let i = 1; i < dtwTs.length; i++) {
    expect(dtwTs[i]!).toBeGreaterThanOrEqual(dtwTs[i - 1]!);
  }
}
