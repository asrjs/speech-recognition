#!/usr/bin/env node
/**
 * Step 3-5: Full decode + token-by-token verification — fp16io vs fp32.
 *
 * Runs the complete decode pipeline with both fp32 and fp16io models,
 * compares token sequences and transcripts.
 *
 * Usage:
 *   node tests/smoke/verify-step3-5-decode.mjs
 *
 * Env vars:
 *   MODEL_FP32    — fp32 model dir
 *   MODEL_FP16IO  — fp16io model dir
 *   AUDIO_PATH    — test audio
 */
import path from 'node:path';
import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { WhisperTokenizer, fetchText } from '../../dist/models/whisper-seq2seq/index.js';
import { WhisperMelProcessor } from '../../dist/audio/whisper-mel.js';
import { splitGraphDecodeLoop } from '../../dist/models/whisper-seq2seq/executor.js';
import { WhisperTimestampLogitProcessor } from '../../dist/models/whisper-seq2seq/processors.js';
import { parseWhisperGenerationConfig, parseWhisperModelConfig } from '../../dist/models/whisper-seq2seq/generation-config.js';

const MODEL_FP32 = process.env.MODEL_FP32 || '/mnt/n/github/asrjs/webgpu-agent-test/models/fp32';
const MODEL_FP16IO = process.env.MODEL_FP16IO || '/mnt/n/github/asrjs/webgpu-agent-test/models/fp16_iofp32';
const AUDIO_PATH = process.env.AUDIO_PATH || 'tests/fixtures/jfk2.en.wav';

function loadWavMono(wavPath) {
  const b = fs.readFileSync(wavPath);
  const channels = b.readUInt16LE(22);
  const frameCount = Math.floor((b.length - 44) / (2 * channels));
  const pcm = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) sum += b.readInt16LE(44 + (i * channels + ch) * 2) / 32768;
    pcm[i] = sum / channels;
  }
  return pcm;
}

async function runDecode(modelDir, label) {
  const configRaw = JSON.parse(await fetchText(path.join(modelDir, 'config.json')));
  const modelConfig = parseWhisperModelConfig(configRaw);
  const genConfig = parseWhisperGenerationConfig(JSON.parse(await fetchText(path.join(modelDir, 'generation_config.json'))));
  const tokenizer = await WhisperTokenizer.fromUrl(path.join(modelDir, 'tokenizer.json'));
  const melBins = modelConfig.numMelBins ?? 128;

  const pcm = loadWavMono(AUDIO_PATH);
  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const padded = WhisperMelProcessor.padToFrames(melProc.process(pcm), 3000);

  // Load sessions
  const enc = await ort.InferenceSession.create(path.join(modelDir, 'encoder_model.onnx'));
  const decInit = await ort.InferenceSession.create(path.join(modelDir, 'decoder_init.onnx'));
  const decStep = await ort.InferenceSession.create(path.join(modelDir, 'decoder_step.onnx'));

  // Encode
  const featTensor = new ort.Tensor('float32', padded, [1, melBins, 3000]);
  const encOut = await enc.run({ input_features: featTensor });
  const encHs = encOut[Object.keys(encOut)[0]];

  // Prompt (SOT, lang, task, notimestamps)
  const promptTokens = [
    tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
    tokenizer.getTokenId('<|en|>') ?? 50268,
    tokenizer.getTokenId('<|transcribe|>') ?? 50360,
    tokenizer.getTokenId('<|notimestamps|>') ?? 50364,
  ];
  const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

  // Logit processor
  const tsProc = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId,
    noTimestampsTokenId: genConfig.noTimestampsTokenId ?? 50363,
    timestampBegin: tokenizer.getTokenId('<|0.00|>') ?? 50364,
    suppressTokens: genConfig.suppressTokens ?? [],
    beginSuppressTokens: genConfig.beginSuppressTokens ?? [],
  });

  // Decode with runInit/runStep callbacks
  let kvDims = {};
  const t0 = performance.now();
  const result = await splitGraphDecodeLoop({
    promptTokens,
    encoderHiddenStates: encHs.data,
    eosTokenId: eosId,
    maxNewTokens: 200,
    modelConfig,
    processLogits: (l, t, b) => tsProc.process(l, t, b),
    runInit: async (prompt) => {
      const ids = new BigInt64Array(prompt.map(id => BigInt(id)));
      const out = await decInit.run({
        input_ids: new ort.Tensor('int64', ids, [1, prompt.length]),
        encoder_hidden_states: encHs,
      });
      const lk = Object.keys(out).find(k => k.includes('logits')) || Object.keys(out)[0];
      const lt = out[lk];
      const vs = lt.dims[lt.dims.length - 1] || 0;
      const pkv = {};
      kvDims = {};
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith('present')) {
          pkv[k] = v.data;
          kvDims[k] = v.dims;
          kvDims[k.replace(/^present\./, 'past_key_values.')] = v.dims;
        }
      }
      return { logits: lt.data, vocabSize: vs, presentKv: pkv };
    },
    runStep: async (tokenId, pastKv) => {
      const feeds = { input_ids: new ort.Tensor('int64', new BigInt64Array([BigInt(tokenId)]), [1, 1]) };
      for (const [name, data] of Object.entries(pastKv)) {
        const sn = name.replace(/^present\./, 'past_key_values.');
        const dims = kvDims[name] || kvDims[sn] || kvDims[name.replace(/^past_key_values\./, 'present.')];
        if (dims) feeds[sn] = new ort.Tensor('float32', new Float32Array(data), dims);
      }
      const out = await decStep.run(feeds);
      const lk = Object.keys(out).find(k => k.includes('logits')) || Object.keys(out)[0];
      const lt = out[lk];
      const vs = lt.dims[lt.dims.length - 1] || 0;
      const pkv = {};
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith('present')) {
          const pn = k.replace(/^present/, 'past_key_values');
          pkv[pn] = v.data;
          kvDims[pn] = v.dims;
        }
      }
      // Preserve encoder KV from previous step
      for (const [k, v] of Object.entries(pastKv)) {
        if (k.includes('encoder') && !pkv[k]) pkv[k] = v;
      }
      return { logits: lt.data, vocabSize: vs, presentKv: pkv };
    },
  });
  const decodeTime = performance.now() - t0;

  // Decode tokens to text
  const transcript = tokenizer.decode([...promptTokens.slice(3), ...result.tokens], { skipSpecialTokens: true });
  const tokenIds = [...result.tokens];

  await enc.release();
  await decInit.release();
  await decStep.release();

  return { label, tokenIds, transcript: transcript.trim(), decodeTime, promptTokens };
}

async function main() {
  console.log('═══════════════════════════════════════════');
  console.log('Step 3-5: Full Decode + Token-by-Token');
  console.log('═══════════════════════════════════════════');
  console.log(`  fp32:   ${MODEL_FP32}`);
  console.log(`  fp16io: ${MODEL_FP16IO}`);
  console.log(`  Audio:  ${AUDIO_PATH}\n`);

  // Run both
  console.log('Running fp32 decode...');
  const fp32 = await runDecode(MODEL_FP32, 'fp32');
  console.log(`  ${fp32.decodeTime.toFixed(0)}ms, ${fp32.tokenIds.length} tokens`);
  console.log(`  Transcript: "${fp32.transcript}"\n`);

  console.log('Running fp16io decode...');
  const fp16io = await runDecode(MODEL_FP16IO, 'fp16io');
  console.log(`  ${fp16io.decodeTime.toFixed(0)}ms, ${fp16io.tokenIds.length} tokens`);
  console.log(`  Transcript: "${fp16io.transcript}"\n`);

  // ── Step 4: Transcript comparison ──
  console.log('═══════════════════════════════════════════');
  console.log('Step 4: Transcript Comparison');
  console.log('═══════════════════════════════════════════');
  const transcriptMatch = fp32.transcript === fp16io.transcript;
  console.log(`  Match: ${transcriptMatch ? '✅ IDENTICAL' : '❌ DIFFERENT'}`);
  if (!transcriptMatch) {
    console.log(`  fp32:   "${fp32.transcript}"`);
    console.log(`  fp16io: "${fp16io.transcript}"`);
  }

  // ── Step 5: Token-by-token comparison ──
  console.log('\n═══════════════════════════════════════════');
  console.log('Step 5: Token-by-Token Comparison');
  console.log('═══════════════════════════════════════════');

  const maxLen = Math.max(fp32.tokenIds.length, fp16io.tokenIds.length);
  let firstDiff = -1;
  let matchCount = 0;

  for (let i = 0; i < maxLen; i++) {
    const t32 = fp32.tokenIds[i];
    const t16 = fp16io.tokenIds[i];
    const match = t32 === t16;
    if (match) matchCount++;
    if (!match && firstDiff === -1) firstDiff = i;
  }

  console.log(`  Total tokens: fp32=${fp32.tokenIds.length} fp16io=${fp16io.tokenIds.length}`);
  console.log(`  Matching: ${matchCount}/${maxLen}`);
  console.log(`  First difference: ${firstDiff === -1 ? 'None' : `position ${firstDiff}`}`);

  // Show first 10 tokens
  console.log('\n  First 10 tokens:');
  for (let i = 0; i < Math.min(10, maxLen); i++) {
    const t32 = fp32.tokenIds[i];
    const t16 = fp16io.tokenIds[i];
    const match = t32 === t16;
    console.log(`    [${i}] fp32=${t32} fp16io=${t16} ${match ? '✅' : '❌'}`);
  }

  // Show all tokens
  console.log('\n  Full token sequences:');
  console.log(`    fp32:   [${fp32.tokenIds.join(', ')}]`);
  console.log(`    fp16io: [${fp16io.tokenIds.join(', ')}]`);

  // ── Timing ──
  console.log('\n═══════════════════════════════════════════');
  console.log('Timing');
  console.log('═══════════════════════════════════════════');
  console.log(`  fp32 decode:   ${fp32.decodeTime.toFixed(0)}ms`);
  console.log(`  fp16io decode: ${fp16io.decodeTime.toFixed(0)}ms`);
  console.log(`  Ratio:         ${(fp16io.decodeTime / fp32.decodeTime).toFixed(2)}x`);

  // ── Verdict ──
  const allTokensMatch = firstDiff === -1;
  const pass = transcriptMatch && allTokensMatch;

  console.log('\n═══════════════════════════════════════════');
  console.log(`Status: ${pass ? '✅ ALL PASS' : '❌ FAIL'}`);
  console.log(`  Transcript match: ${transcriptMatch ? '✅' : '❌'}`);
  console.log(`  Token-by-token:   ${allTokensMatch ? '✅' : '❌'}`);
  console.log('═══════════════════════════════════════════');

  process.exit(pass ? 0 : 1);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
