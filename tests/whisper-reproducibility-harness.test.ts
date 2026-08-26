import { afterAll, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import type * as OrtNode from 'onnxruntime-node';
import { parseWhisperManifest } from '../src/models/whisper-seq2seq/manifest.js';
import { parseWhisperGenerationConfig } from '../src/models/whisper-seq2seq/generation-config.js';
import { WhisperTokenizer } from '../src/models/whisper-seq2seq/tokenizer.js';
import { WhisperMelProcessor } from '../src/audio/whisper-mel.js';
import { WhisperTimestampLogitProcessor } from '../src/models/whisper-seq2seq/processors.js';
import { processSplitGraphAlignment } from '../src/models/whisper-seq2seq/executor.js';
import { decodeAudioSourceToMonoPcm } from '../src/runtime/media.js';

function argmax(arr: Float32Array): number {
  let maxIdx = 0, maxVal = arr[0] ?? -Infinity;
  for (let i = 1; i < arr.length; i++) {
    if ((arr[i] ?? -Infinity) > maxVal) { maxVal = arr[i]!; maxIdx = i; }
  }
  return maxIdx;
}

/**
 * onnxruntime-node resolves native binaries at import time, so this harness
 * loads it lazily. The artifact-gated tests below only reach it after the
 * WHISPER_REFERENCE_JSON gate passes, which keeps plain `npm test` runnable
 * in environments installed with ONNXRUNTIME_NODE_INSTALL=skip.
 */
let ort: typeof OrtNode;
async function ensureOrt(): Promise<typeof OrtNode> {
  ort ??= await import('onnxruntime-node');
  return ort;
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
  refEosStop: boolean,
): { matches: number; minLen: number; matchPct: number; mismatches: string[] } {
  const tsNorm = normalizeEos(tsTokens, eosTokenId, tsEosStop);
  const refNorm = normalizeEos(refTokens, eosTokenId, refEosStop);

  const minLen = Math.min(tsNorm.length, refNorm.length);
  const maxLen = Math.max(tsNorm.length, refNorm.length);
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
    matchPct: maxLen > 0 ? (100 * matches / maxLen) : 0,
    mismatches,
  };
}

interface ReferenceJson {
  audio: { path: string; sample_rate: number; duration_seconds: number; num_samples: number };
  model: {
    id: string;
    export_dir: string;
    encoder_dir?: string;
    decoder_dir?: string;
    format: string;
    d_model: number;
    decoder_layers: number;
    decoder_attention_heads: number;
  };
  prompt_ids: number[];
  pytorch: {
    no_timestamps: { tokens: number[]; text: string };
    with_timestamps: { tokens: number[]; text: string; timestamp_tokens: number[] };
  };
  onnx_python?: {
    tokens: number[];
    generated: number[];
    eos_stopped?: boolean;
    alignment: { shape: number[]; text_shape: number[]; row_sum_min: number; row_sum_max: number; row_sum_mean: number } | null;
  };
  decode?: { max_new_tokens?: number; language?: string; task?: string; no_timestamps?: boolean };
  mel_features_path?: string;
}

interface WhisperGraphDimensions {
  readonly encoderInputType: 'float16' | 'float32';
  readonly inputFeatureFrames: number;
  readonly encoderSequenceLength: number;
  readonly numMelBins: number;
  readonly dModel: number;
  readonly vocabSize: number;
}

type ReproSessions = Awaited<ReturnType<typeof createSessions>>;
let sharedSessions: Promise<ReproSessions> | undefined;

afterAll(async () => {
  if (!sharedSessions) return;
  const { encSess, initSess, stepSess, alignSess } = await sharedSessions;
  await Promise.all([
    encSess.release(),
    initSess.release(),
    stepSess.release(),
    alignSess?.release(),
  ]);
});

describe('Whisper splitgraph reproducibility harness (vs HF Transformers)', () => {
  it('feature-input: 100% token match using Python/HF mel features (no TS frontend tolerance)', async () => {
    const refPath = process.env.WHISPER_REFERENCE_JSON;
    if (!refPath) { console.warn('Skipping: set WHISPER_REFERENCE_JSON'); return; }
    if (!fs.existsSync(refPath)) { console.warn(`Skipping: ${refPath} not found`); return; }

    const ref: ReferenceJson = JSON.parse(fs.readFileSync(refPath, 'utf-8'));
    const melPath = process.env.WHISPER_REFERENCE_MEL ?? ref.mel_features_path;
    if (!melPath || !fs.existsSync(melPath)) {
      console.warn('Skipping feature-input mode: no mel_features_path in reference (use --export-mel)');
      return;
    }

    const { encSess, initSess, stepSess, alignSess, tokenizer, manifest, generationConfig, dimensions }
      = await setupSessions(ref);

    const { decoderLayers } = manifest.modelConfig;
    const { encoderInputType, inputFeatureFrames, encoderSequenceLength, numMelBins, dModel, vocabSize } = dimensions;

    // ── Load Python mel features directly (bypass TS mel processor) ──
    const melNpy = fs.readFileSync(melPath);
    // Parse NPY header: magic (6 bytes) + version (2 bytes) + header_len (2 bytes LE) + header dict
    const headerLen = new DataView(melNpy.buffer, melNpy.byteOffset + 8, 2).getUint16(0, true);
    const headerStr = new TextDecoder().decode(melNpy.subarray(10, 10 + headerLen));
    // Extract shape from header like: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 80, 3000)}
    const shapeMatch = headerStr.match(/shape':\s*\(([^)]+)\)/);
    if (!shapeMatch) throw new Error('Failed to parse NPY shape header');
    const npyShape = shapeMatch[1]!.split(',').map((value) => Number.parseInt(value.trim(), 10)).filter(Number.isFinite);
    expect(npyShape).toEqual([1, numMelBins, inputFeatureFrames]);
    const dataOffset = 10 + headerLen;
    const melData = new Float32Array(
      melNpy.buffer,
      melNpy.byteOffset + dataOffset,
      numMelBins * inputFeatureFrames,
    );
    expect(melData.length).toBe(numMelBins * inputFeatureFrames);

    console.log(`Loaded Python mel features: ${melData.length} floats, shape ${shapeMatch[1]}`);

    const { allTokens, generated, encOut, eosStopped } = await runDecodeLoop({
      encSess, initSess, stepSess, tokenizer,
      melData, encoderInputType, numMelBins, inputFeatureFrames, encoderSequenceLength, dModel, vocabSize,
      promptIds: ref.prompt_ids, decoderLayers,
      suppressTokens: [...(generationConfig.suppressTokens ?? [])],
      beginSuppressTokens: [...(generationConfig.beginSuppressTokens ?? [])],
      maxNewTokens: ref.decode?.max_new_tokens ?? 128,
    });

    // ═══════════════════════════════════════════════════════
    // FEATURE-INPUT MODE: require 100% token match
    // ═══════════════════════════════════════════════════════
    const referenceDecode = getReferenceDecode(ref, tokenizer);
    const refTokens = referenceDecode.tokens;
    const decoded = tokenizer.decode(allTokens, { skipSpecialTokens: true });
    const refDecoded = tokenizer.decode(refTokens, { skipSpecialTokens: true });
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const refEosStop = referenceDecode.eosStopped;

    const cmp = compareTokens(allTokens, refTokens, eosId, eosStopped, refEosStop);

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
    await validateAlignment(alignSess, allTokens, encOut, generated, ref, encoderSequenceLength);

    console.log('\nFeature-input reproducibility PASSED (100% token match)');
  }, 180000);

  it('wav-input: >=80% token match using TS mel frontend (tolerance for HF frontend differences, will tighten)', async () => {
    const refPath = process.env.WHISPER_REFERENCE_JSON;
    if (!refPath) { console.warn('Skipping: set WHISPER_REFERENCE_JSON'); return; }
    if (!fs.existsSync(refPath)) { console.warn(`Skipping: ${refPath} not found`); return; }

    const ref: ReferenceJson = JSON.parse(fs.readFileSync(refPath, 'utf-8'));
    const audioPath = process.env.WHISPER_REFERENCE_AUDIO ?? ref.audio.path;
    if (!fs.existsSync(audioPath)) { console.warn(`Skipping: audio not found at ${audioPath}`); return; }

    const { encSess, initSess, stepSess, alignSess, tokenizer, manifest, generationConfig, dimensions }
      = await setupSessions(ref);

    const { decoderLayers } = manifest.modelConfig;
    const { encoderInputType, inputFeatureFrames, encoderSequenceLength, numMelBins, dModel, vocabSize } = dimensions;

    // ── Load audio and run TS mel processor ──
    const audioBuf = fs.readFileSync(audioPath);
    const audioBytes = Uint8Array.from(audioBuf);
    const decodedAudio = await decodeAudioSourceToMonoPcm(new Blob([audioBytes]), {
      strategy: 'native-rate',
      targetSampleRate: 16000,
    });
    const audioSamples = decodedAudio.pcm;

    const melProc = new WhisperMelProcessor({ nMels: numMelBins });
    const melResult = melProc.process(audioSamples);
    const paddedMel = WhisperMelProcessor.padToFrames(melResult, inputFeatureFrames);

    const { allTokens, generated, encOut, eosStopped } = await runDecodeLoop({
      encSess, initSess, stepSess, tokenizer,
      melData: paddedMel, encoderInputType, numMelBins, inputFeatureFrames, encoderSequenceLength, dModel, vocabSize,
      promptIds: ref.prompt_ids, decoderLayers,
      suppressTokens: [...(generationConfig.suppressTokens ?? [])],
      beginSuppressTokens: [...(generationConfig.beginSuppressTokens ?? [])],
      maxNewTokens: ref.decode?.max_new_tokens ?? 128,
    });

    // ═══════════════════════════════════════════════════════
    // WAV-INPUT MODE: allow >=80% (mel frontend may differ)
    // ═══════════════════════════════════════════════════════
    const referenceDecode = getReferenceDecode(ref, tokenizer);
    const refTokens = referenceDecode.tokens;
    const decoded = tokenizer.decode(allTokens, { skipSpecialTokens: true });
    const refDecoded = tokenizer.decode(refTokens, { skipSpecialTokens: true });
    const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

    const refEosStop = referenceDecode.eosStopped;

    const cmp = compareTokens(allTokens, refTokens, eosId, eosStopped, refEosStop);

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

    // WAV-INPUT TOLERANCE: >=80% because TS WhisperMelProcessor may differ from
    // PyTorch's WhisperFeatureExtractor. This threshold exists ONLY for the
    // wav-input path. Feature-input (Python mel) path requires 100%.
    expect(cmp.matchPct).toBeGreaterThanOrEqual(80);

    // No timestamp tokens in no_timestamps mode
    const tsBegin = tokenizer.getTokenId('<|0.00|>') ?? 50364;
    const hasTimestamps = generated.some((t) => t >= tsBegin);
    expect(hasTimestamps).toBe(false);

    // Alignment validation
    await validateAlignment(alignSess, allTokens, encOut, generated, ref, encoderSequenceLength);

    console.log('\nWAV-input reproducibility PASSED');
  }, 180000);
});

// ── Helpers ────────────────────────────────────────────────────

function getReferenceDecode(
  ref: ReferenceJson,
  tokenizer: WhisperTokenizer,
): { readonly tokens: number[]; readonly eosStopped: boolean } {
  const eosTokenId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
  if (ref.onnx_python) {
    return {
      tokens: ref.onnx_python.tokens,
      eosStopped: ref.onnx_python.eos_stopped
        ?? ref.onnx_python.tokens[ref.onnx_python.tokens.length - 1] === eosTokenId,
    };
  }
  const tokens = ref.pytorch.no_timestamps.tokens;
  return { tokens, eosStopped: tokens[tokens.length - 1] === eosTokenId };
}

async function setupSessions(ref: ReferenceJson) {
  sharedSessions ??= createSessions(ref);
  return sharedSessions;
}

async function createSessions(ref: ReferenceJson) {
  const modelDirOverride = process.env.WHISPER_REFERENCE_MODEL_DIR;
  const modelDir = modelDirOverride ?? ref.model.export_dir;
  const encoderDir = process.env.WHISPER_REFERENCE_ENCODER_DIR
    ?? modelDirOverride
    ?? ref.model.encoder_dir
    ?? modelDir;
  const decoderDir = process.env.WHISPER_REFERENCE_DECODER_DIR
    ?? modelDirOverride
    ?? ref.model.decoder_dir
    ?? modelDir;
  const encoderFile = (...parts: string[]) => path.join(encoderDir, ...parts);
  const decoderFile = (...parts: string[]) => path.join(decoderDir, ...parts);
  if (!fs.existsSync(encoderFile('encoder_model.onnx'))) {
    throw new Error(`Missing encoder_model.onnx in ${encoderDir}`);
  }
  const requiredDecoderFiles = ['decoder_init.onnx', 'decoder_step.onnx', 'tokenizer.json', 'manifest.json'];
  if (!requiredDecoderFiles.every((name) => fs.existsSync(decoderFile(name)))) {
    throw new Error(`Missing decoder model files in ${decoderDir}`);
  }

  const manifestRaw = JSON.parse(fs.readFileSync(decoderFile('manifest.json'), 'utf-8')) as Record<string, unknown>;
  const manifest = parseWhisperManifest(manifestRaw);
  const specialTokens = (manifestRaw.special_tokens ?? {}) as Record<string, unknown>;
  const generationConfigPath = decoderFile('generation_config.json');
  const generationConfigRaw = fs.existsSync(generationConfigPath)
    ? JSON.parse(fs.readFileSync(generationConfigPath, 'utf-8')) as Record<string, unknown>
    : {};
  const generationConfig = parseWhisperGenerationConfig({
    alignment_heads: manifestRaw.alignment_heads,
    no_timestamps_token_id: specialTokens.no_timestamps_token_id,
    suppress_tokens: specialTokens.suppress_tokens,
    begin_suppress_tokens: specialTokens.begin_suppress_tokens,
    max_length: manifestRaw.max_target_positions,
    ...generationConfigRaw,
  });

  expect(manifest.modelConfig.dModel).toBe(ref.model.d_model);
  expect(manifest.modelConfig.decoderLayers).toBe(ref.model.decoder_layers);
  expect(manifest.modelConfig.decoderAttentionHeads).toBe(ref.model.decoder_attention_heads);

  await ensureOrt();
  const sessionOpts = { graphOptimizationLevel: 'all' as const, executionMode: 'parallel' as const };
  const encSess = await ort.InferenceSession.create(encoderFile('encoder_model.onnx'), sessionOpts);
  const initSess = await ort.InferenceSession.create(decoderFile('decoder_init.onnx'), sessionOpts);
  const stepSess = await ort.InferenceSession.create(decoderFile('decoder_step.onnx'), sessionOpts);
  const hasAlign = fs.existsSync(decoderFile('decoder_align.onnx'));
  const alignSess = hasAlign ? await ort.InferenceSession.create(decoderFile('decoder_align.onnx'), sessionOpts) : null;
  const tokenizer = await WhisperTokenizer.fromUrl(`file://${decoderFile('tokenizer.json')}`);
  const dimensions = readWhisperGraphDimensions(encSess, initSess);

  expect(dimensions.dModel).toBe(manifest.modelConfig.dModel);
  expect(dimensions.numMelBins).toBe(manifest.modelConfig.numMelBins);

  return { encSess, initSess, stepSess, alignSess, tokenizer, manifest, generationConfig, dimensions };
}

function readWhisperGraphDimensions(
  encSess: OrtNode.InferenceSession,
  initSess: OrtNode.InferenceSession,
): WhisperGraphDimensions {
  const encoderInput = encSess.inputMetadata.find((entry) => entry.name === 'input_features');
  const encoderOutput = encSess.outputMetadata.find((entry) => entry.name.startsWith('last_hidden_state'));
  const logitsOutput = initSess.outputMetadata.find((entry) => entry.name.includes('logits'));
  if (!encoderInput || !encoderOutput || !logitsOutput) {
    throw new Error('Whisper graph metadata is missing input_features, last_hidden_state, or logits.');
  }
  if (encoderInput.type !== 'float16' && encoderInput.type !== 'float32') {
    throw new Error(`Unsupported Whisper encoder input type: ${encoderInput.type}.`);
  }

  return {
    encoderInputType: encoderInput.type,
    numMelBins: requireStaticDimension(encoderInput.shape, 1, 'encoder input mel bins'),
    inputFeatureFrames: requireStaticDimension(encoderInput.shape, 2, 'encoder input frames'),
    encoderSequenceLength: requireStaticDimension(encoderOutput.shape, 1, 'encoder output positions'),
    dModel: requireStaticDimension(encoderOutput.shape, 2, 'encoder output width'),
    vocabSize: requireStaticDimension(logitsOutput.shape, 2, 'decoder vocabulary'),
  };
}

function requireStaticDimension(
  shape: readonly (number | string)[],
  index: number,
  label: string,
): number {
  const value = shape[index];
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new Error(`Expected a static positive ${label} dimension, received ${String(value)}.`);
  }
  return value;
}

async function runDecodeLoop(params: {
  encSess: OrtNode.InferenceSession;
  initSess: OrtNode.InferenceSession;
  stepSess: OrtNode.InferenceSession;
  tokenizer: WhisperTokenizer;
  melData: Float32Array;
  encoderInputType: 'float16' | 'float32';
  numMelBins: number;
  inputFeatureFrames: number;
  encoderSequenceLength: number;
  dModel: number;
  vocabSize: number;
  promptIds: number[];
  decoderLayers: number;
  suppressTokens: number[];
  beginSuppressTokens: number[];
  maxNewTokens: number;
}) {
  const {
    encSess, initSess, stepSess, tokenizer, melData, encoderInputType, numMelBins,
    inputFeatureFrames, encoderSequenceLength, dModel, vocabSize,
    promptIds, decoderLayers, suppressTokens, beginSuppressTokens, maxNewTokens,
  } = params;

  const melTensor = encoderInputType === 'float16'
    ? new ort.Tensor('float16', float32ToFloat16Bits(melData), [1, numMelBins, inputFeatureFrames])
    : new ort.Tensor('float32', melData, [1, numMelBins, inputFeatureFrames]);
  const encoderOutputs = await encSess.run({ input_features: melTensor });
  const encoderOutputName = Object.keys(encoderOutputs).find((name) => name.startsWith('last_hidden_state'))
    ?? Object.keys(encoderOutputs)[0]!;
  const encOut = encoderOutputs[encoderOutputName] as OrtNode.Tensor;
  expect(encOut.dims).toEqual([1, encoderSequenceLength, dModel]);

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
  const initLogits = (initOut[logitsKey] as OrtNode.Tensor).data as Float32Array;
  const tsVocabSize = (initOut[logitsKey] as OrtNode.Tensor).dims[2] ?? 51865;
  expect(tsVocabSize).toBe(vocabSize);

  const pastKv: Record<string, OrtNode.Tensor> = {};
  for (const [k, v] of Object.entries(initOut)) {
    if (k.startsWith('present')) pastKv[k.replace('present.', 'past_key_values.')] = v as OrtNode.Tensor;
  }
  expect(Object.keys(pastKv).length).toBe(4 * decoderLayers);

  // First token
  const lastOffset = initLogits.length - tsVocabSize;
  const firstLogits = initLogits.subarray(lastOffset);
  processor.process(firstLogits, promptIds, promptIds.length);
  let nextToken = argmax(firstLogits);
  let eosStopped = nextToken === eosId;
  const generated: number[] = eosStopped ? [] : [nextToken];

  // Step loop
  while (!eosStopped && generated.length < maxNewTokens) {
    const stepIn = new BigInt64Array([BigInt(nextToken)]);
    const stepFeeds: Record<string, unknown> = { input_ids: new ort.Tensor('int64', stepIn, [1, 1]) };
    for (const [k, v] of Object.entries(pastKv)) stepFeeds[k] = v;
    const stepOut = await stepSess.run(stepFeeds);
    const sLogitsKey = Object.keys(stepOut).find((k) => k.includes('logits'))!;
    const sLogits = (stepOut[sLogitsKey] as OrtNode.Tensor).data as Float32Array;
    processor.process(sLogits, [...promptIds, ...generated], promptIds.length);
    nextToken = argmax(sLogits);
    if (nextToken === eosId) {
      eosStopped = true;
      break;
    }
    generated.push(nextToken);
    for (const [k, v] of Object.entries(stepOut)) {
      if (k.startsWith('present')) pastKv[k.replace('present.', 'past_key_values.')] = v as OrtNode.Tensor;
    }
  }

  return { allTokens: [...promptIds, ...generated], generated, encOut, eosStopped };
}

function float32ToFloat16Bits(values: Float32Array): Uint16Array {
  const output = new Uint16Array(values.length);
  const floatView = new Float32Array(1);
  const uintView = new Uint32Array(floatView.buffer);

  for (let i = 0; i < values.length; i++) {
    floatView[0] = values[i]!;
    const bits = uintView[0]!;
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    let halfExponent = exponent - 127 + 15;
    let mantissa = bits & 0x7fffff;

    if (exponent === 0xff) {
      output[i] = sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
      continue;
    }
    if (halfExponent >= 0x1f) {
      output[i] = sign | 0x7c00;
      continue;
    }
    if (halfExponent <= 0) {
      if (halfExponent < -10) {
        output[i] = sign;
        continue;
      }
      mantissa |= 0x800000;
      const shift = 14 - halfExponent;
      let halfMantissa = mantissa >>> shift;
      const remainder = mantissa & ((1 << shift) - 1);
      const halfway = 1 << (shift - 1);
      if (remainder > halfway || (remainder === halfway && (halfMantissa & 1) !== 0)) {
        halfMantissa += 1;
      }
      output[i] = sign | halfMantissa;
      continue;
    }

    let halfMantissa = mantissa >>> 13;
    const remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
      halfMantissa += 1;
      if (halfMantissa === 0x400) {
        halfMantissa = 0;
        halfExponent += 1;
      }
    }
    output[i] = halfExponent >= 0x1f
      ? sign | 0x7c00
      : sign | (halfExponent << 10) | halfMantissa;
  }

  return output;
}

describe('Whisper reference mel dtype conversion', () => {
  it('converts float32 values to IEEE float16 bits with round-to-nearest-even', () => {
    const values = new Float32Array([
      -2,
      -1,
      -0,
      0,
      1,
      2,
      2 ** -24,
      2 ** -25,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      Number.NaN,
    ]);

    expect(Array.from(float32ToFloat16Bits(values))).toEqual([
      0xc000,
      0xbc00,
      0x8000,
      0x0000,
      0x3c00,
      0x4000,
      0x0001,
      0x0000,
      0x7c00,
      0xfc00,
      0x7e00,
    ]);
  });
});

async function validateAlignment(
  alignSess: OrtNode.InferenceSession | null,
  allTokens: number[], encOut: OrtNode.Tensor,
  generated: number[], ref: ReferenceJson, maxSrcPos: number,
) {
  if (!alignSess || generated.length === 0) return;

  const alignArr = new BigInt64Array(allTokens.map((id) => BigInt(id)));
  const alignOut = await alignSess.run({
    input_ids: new ort.Tensor('int64', alignArr, [1, allTokens.length]),
    encoder_hidden_states: encOut,
  });
  const alignKey = Object.keys(alignOut)[0]!;
  const alignment = alignOut[alignKey] as OrtNode.Tensor;

  // dims may be a number[] or readonly number[]
  const dims = alignment.dims;
  const [aB, aT, aS] = [dims[0] ?? 1, dims[1] ?? 0, dims[2] ?? maxSrcPos];

  expect(aB).toBe(1);
  expect(aS).toBe(maxSrcPos);

  const data = alignment.data as Float32Array;
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
