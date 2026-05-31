#!/usr/bin/env node
/**
 * Step 2: Encoder output verification — fp16io vs fp32.
 *
 * Compare encoder hidden states between fp32 baseline and fp16io variant.
 * fp16io = fp16 internal weights + fp32 I/O (keep_io_types=True).
 *
 * Metrics:
 *   - Cosine similarity per sample (should be > 0.999)
 *   - MSE of hidden states
 *   - Max absolute difference
 *   - Per-frame cosine similarity (to detect drift in specific frames)
 *
 * Usage:
 *   node tests/smoke/verify-step2-encoder.mjs
 *
 * Env vars:
 *   MODEL_FP32    — path to fp32 model dir (default: /mnt/n/.../models/fp32)
 *   MODEL_FP16IO  — path to fp16io model dir (default: /mnt/n/.../models/fp16_iofp32)
 *   AUDIO_PATH    — test audio (default: jfk2.en.wav from webgpu-agent-test)
 */
import path from 'node:path';
import fs from 'node:fs';
import * as ort from 'onnxruntime-node';
import { WhisperMelProcessor } from '../../dist/audio/whisper-mel.js';
import { parseWhisperModelConfig } from '../../dist/models/whisper-seq2seq/generation-config.js';
import { fetchText } from '../../dist/models/whisper-seq2seq/index.js';

// ── Config ──
const MODEL_FP32 = process.env.MODEL_FP32 || '/mnt/n/github/asrjs/webgpu-agent-test/models/fp32';
const MODEL_FP16IO = process.env.MODEL_FP16IO || '/mnt/n/github/asrjs/webgpu-agent-test/models/fp16_iofp32';
const AUDIO_PATH = process.env.AUDIO_PATH || '/mnt/n/github/asrjs/webgpu-agent-test/jfk2.en.wav';
const COSINE_THRESHOLD = 0.999;
const MSE_THRESHOLD = 0.01;

// ── Audio loading (same as verify-step1) ──
function loadWavMono(path) {
  const b = fs.readFileSync(path);
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4), sz = b.readUInt32LE(off + 4), st = off + 8;
    if (id === 'fmt ') fmt = { channels: b.readUInt16LE(st + 2), sampleRate: b.readUInt32LE(st + 4), bitsPerSample: b.readUInt16LE(st + 14) };
    else if (id === 'data') data = b.subarray(st, st + sz);
    off = st + sz + (sz % 2);
  }
  if (!fmt || !data) throw new Error('bad wav');
  const bytes = fmt.bitsPerSample / 8, frames = Math.floor(data.length / bytes / fmt.channels);
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let s = 0;
    for (let ch = 0; ch < fmt.channels; ch++) {
      const p = (i * fmt.channels + ch) * bytes;
      s += data.readInt16LE(p) / 32768;
    }
    out[i] = s / fmt.channels;
  }
  if (fmt.sampleRate === 16000) return out;
  const r = 16000 / fmt.sampleRate, o = new Float32Array(Math.max(1, Math.floor(out.length * r)));
  for (let i = 0; i < o.length; i++) {
    const x = i / r, x0 = Math.floor(x), x1 = Math.min(out.length - 1, x0 + 1), t = x - x0;
    o[i] = (out[x0] ?? 0) * (1 - t) + (out[x1] ?? 0) * t;
  }
  return o;
}

// ── Metrics ──
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function mse(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum / a.length;
}

function maxAbsDiff(a, b) {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > max) max = d;
  }
  return max;
}

function tensorStats(data, label) {
  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < min) min = data[i];
    if (data[i] > max) max = data[i];
    sum += data[i];
  }
  const mean = sum / data.length;
  let varSum = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / data.length);
  return { label, min, max, mean, std, count: data.length };
}

// ── Main ──
async function main() {
  console.log('═══════════════════════════════════════════');
  console.log('Step 2: Encoder Output Verification');
  console.log('═══════════════════════════════════════════');
  console.log(`  fp32 model:   ${MODEL_FP32}`);
  console.log(`  fp16io model: ${MODEL_FP16IO}`);
  console.log(`  Audio:        ${AUDIO_PATH}`);
  console.log('');

  // Load config to get num_mel_bins
  const configRaw = JSON.parse(await fetchText(path.join(MODEL_FP32, 'config.json')));
  const modelConfig = parseWhisperModelConfig(configRaw);
  const melBins = modelConfig.numMelBins ?? 128;
  console.log(`  num_mel_bins: ${melBins}`);

  // Load audio & compute mel
  const pcm = loadWavMono(AUDIO_PATH);
  console.log(`  Audio: ${pcm.length} samples @ ${(pcm.length / 16000).toFixed(1)}s`);

  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const padded = WhisperMelProcessor.padToFrames(melProc.process(pcm), 3000);
  console.log(`  Mel: ${padded.length} values (${melBins} × 3000)`);

  // Create input tensor (float32 for both — fp16io has fp32 I/O)
  const inputTensor = new ort.Tensor('float32', padded, [1, melBins, 3000]);

  // Load & run fp32 encoder
  console.log('\nLoading fp32 encoder...');
  const t0 = performance.now();
  const encFp32 = await ort.InferenceSession.create(path.join(MODEL_FP32, 'encoder_model.onnx'));
  console.log(`  Loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  console.log('Running fp32 encoder...');
  const t1 = performance.now();
  const outFp32 = await encFp32.run({ input_features: inputTensor });
  const fp32Time = performance.now() - t1;
  const fp32Keys = Object.keys(outFp32);
  const fp32Data = outFp32[fp32Keys[0]].data;
  console.log(`  Output: ${fp32Keys[0]} — dims: [${outFp32[fp32Keys[0]].dims}] — ${fp32Time.toFixed(0)}ms`);

  // Load & run fp16io encoder
  console.log('\nLoading fp16io encoder...');
  const t2 = performance.now();
  const encFp16io = await ort.InferenceSession.create(path.join(MODEL_FP16IO, 'encoder_model.onnx'));
  console.log(`  Loaded in ${((performance.now() - t2) / 1000).toFixed(1)}s`);

  console.log('Running fp16io encoder...');
  const t3 = performance.now();
  const outFp16io = await encFp16io.run({ input_features: inputTensor });
  const fp16ioTime = performance.now() - t3;
  const fp16ioKeys = Object.keys(outFp16io);
  const fp16ioData = outFp16io[fp16ioKeys[0]].data;
  console.log(`  Output: ${fp16ioKeys[0]} — dims: [${outFp16io[fp16ioKeys[0]].dims}] — ${fp16ioTime.toFixed(0)}ms`);

  // Validate shapes match
  const fp32Dims = outFp32[fp32Keys[0]].dims;
  const fp16ioDims = outFp16io[fp16ioKeys[0]].dims;
  const shapeMatch = fp32Dims.length === fp16ioDims.length &&
    fp32Dims.every((d, i) => d === fp16ioDims[i]);

  if (!shapeMatch) {
    console.log(`\n❌ FAIL: Output shape mismatch!`);
    console.log(`  fp32:   [${fp32Dims}]`);
    console.log(`  fp16io: [${fp16ioDims}]`);
    process.exit(1);
  }

  const totalElements = fp32Data.length;
  console.log(`\nOutput shape: [${fp32Dims}] — ${totalElements} elements`);

  // ── Global metrics ──
  const globalCosine = cosineSimilarity(fp32Data, fp16ioData);
  const globalMse = mse(fp32Data, fp16ioData);
  const globalMaxDiff = maxAbsDiff(fp32Data, fp16ioData);

  console.log('\n── Global Metrics ──');
  console.log(`  Cosine similarity: ${globalCosine.toFixed(6)}`);
  console.log(`  MSE:               ${globalMse.toExponential(4)}`);
  console.log(`  Max |diff|:        ${globalMaxDiff.toExponential(4)}`);

  // ── Stats ──
  const fp32Stats = tensorStats(fp32Data, 'fp32');
  const fp16ioStats = tensorStats(fp16ioData, 'fp16io');

  console.log('\n── Distribution Stats ──');
  for (const s of [fp32Stats, fp16ioStats]) {
    console.log(`  ${s.label}: min=${s.min.toFixed(4)} max=${s.max.toFixed(4)} mean=${s.mean.toFixed(4)} std=${s.std.toFixed(4)}`);
  }

  // ── Per-frame cosine similarity ──
  // Output shape: [1, frames, hidden_dim] or [1, hidden_dim, frames]
  // Whisper encoder output is typically [batch, hidden_dim, frames]
  const batchDim = fp32Dims[0];
  const dim1 = fp32Dims[1];
  const dim2 = fp32Dims[2];

  // Determine layout: [batch, hidden, frames] or [batch, frames, hidden]
  // Whisper encoder output is [batch, hidden_dim, frames] = [1, 1280, 1500]
  // hidden_dim is typically 1280 for large-v3-turbo, frames is variable
  let hiddenDim, nFrames;
  if (dim1 > dim2) {
    // [batch, hidden, frames]
    hiddenDim = dim1;
    nFrames = dim2;
  } else {
    // [batch, frames, hidden]
    hiddenDim = dim2;
    nFrames = dim1;
  }

  console.log(`\n── Per-Frame Cosine Similarity (${nFrames} frames, hidden=${hiddenDim}) ──`);

  let minFrameCos = 1, maxFrameCos = 1, sumCos = 0;
  const frameCoss = [];
  const isHiddenFirst = dim1 > dim2;

  for (let f = 0; f < nFrames; f++) {
    const frameFp32 = new Float32Array(hiddenDim);
    const frameFp16io = new Float32Array(hiddenDim);

    for (let h = 0; h < hiddenDim; h++) {
      const idx = isHiddenFirst
        ? (h * nFrames + f)   // [hidden, frames] layout
        : (f * hiddenDim + h); // [frames, hidden] layout
      frameFp32[h] = fp32Data[idx];
      frameFp16io[h] = fp16ioData[idx];
    }

    const cos = cosineSimilarity(frameFp32, frameFp16io);
    frameCoss.push(cos);
    sumCos += cos;
    if (cos < minFrameCos) minFrameCos = cos;
    if (cos > maxFrameCos) maxFrameCos = cos;
  }

  const meanFrameCos = sumCos / nFrames;

  // Show worst frames
  const sortedIdx = frameCoss.map((c, i) => ({ cos: c, idx: i })).sort((a, b) => a.cos - b.cos);
  console.log(`  Mean:   ${meanFrameCos.toFixed(6)}`);
  console.log(`  Min:    ${minFrameCos.toFixed(6)} (frame ${sortedIdx[0].idx})`);
  console.log(`  Max:    ${maxFrameCos.toFixed(6)} (frame ${sortedIdx[sortedIdx.length - 1].idx})`);

  console.log(`\n  Worst 5 frames:`);
  for (let i = 0; i < Math.min(5, sortedIdx.length); i++) {
    const { cos, idx } = sortedIdx[i];
    console.log(`    Frame ${idx}: cosine=${cos.toFixed(6)}`);
  }

  console.log(`\n  Best 5 frames:`);
  for (let i = 0; i < Math.min(5, sortedIdx.length); i++) {
    const { cos, idx } = sortedIdx[sortedIdx.length - 1 - i];
    console.log(`    Frame ${idx}: cosine=${cos.toFixed(6)}`);
  }

  // ── Per-layer drift analysis ──
  // Group hidden dims into layers (Whisper large-v3-turbo has 32 layers × 40 dims = 1280)
  // Actually, let's just check per-dim statistics
  console.log(`\n── Per-Dimension Drift (first 20 of ${hiddenDim}) ──`);
  const dimDrifts = [];
  for (let d = 0; d < hiddenDim; d++) {
    let dimMse = 0;
    for (let f = 0; f < nFrames; f++) {
      const idx = isHiddenFirst ? (d * nFrames + f) : (f * hiddenDim + d);
      const diff = fp32Data[idx] - fp16ioData[idx];
      dimMse += diff * diff;
    }
    dimDrifts.push({ dim: d, mse: dimMse / nFrames });
  }

  dimDrifts.sort((a, b) => b.mse - a.mse);

  for (let i = 0; i < Math.min(20, dimDrifts.length); i++) {
    const { dim, mse: dMse } = dimDrifts[i];
    console.log(`    dim[${dim}]: MSE=${dMse.toExponential(4)}`);
  }

  // ── Timing comparison ──
  console.log(`\n── Timing ──`);
  console.log(`  fp32 encoder:   ${fp32Time.toFixed(0)}ms`);
  console.log(`  fp16io encoder: ${fp16ioTime.toFixed(0)}ms`);
  console.log(`  Speedup:        ${(fp32Time / fp16ioTime).toFixed(2)}x`);

  // ── NaN/Inf check ──
  let fp32NaN = 0, fp16ioNaN = 0, fp32Inf = 0, fp16ioInf = 0;
  for (let i = 0; i < totalElements; i++) {
    if (Number.isNaN(fp32Data[i])) fp32NaN++;
    if (Number.isNaN(fp16ioData[i])) fp16ioNaN++;
    if (!Number.isFinite(fp32Data[i]) && !Number.isNaN(fp32Data[i])) fp32Inf++;
    if (!Number.isFinite(fp16ioData[i]) && !Number.isNaN(fp16ioData[i])) fp16ioInf++;
  }
  console.log(`\n── NaN/Inf ──`);
  console.log(`  fp32:   NaN=${fp32NaN} Inf=${fp32Inf}`);
  console.log(`  fp16io: NaN=${fp16ioNaN} Inf=${fp16ioInf}`);

  // ── Verdict ──
  const cosinePass = globalCosine >= COSINE_THRESHOLD;
  const msePass = globalMse < MSE_THRESHOLD;
  const noNan = fp16ioNaN === 0 && fp16ioInf === 0;
  const pass = cosinePass && msePass && noNan;

  console.log(`\n═══════════════════════════════════════════`);
  console.log(`Results:`);
  console.log(`  Cosine similarity: ${globalCosine.toFixed(6)} ${cosinePass ? '✅' : '❌'} (threshold: ≥ ${COSINE_THRESHOLD})`);
  console.log(`  MSE:               ${globalMse.toExponential(4)} ${msePass ? '✅' : '❌'} (threshold: < ${MSE_THRESHOLD})`);
  console.log(`  NaN/Inf:           ${noNan ? '✅ None' : '❌ Present'}`);
  console.log(`\nStatus: ${pass ? '✅ PASS' : '❌ FAIL'}`);

  if (!pass) {
    if (!cosinePass) {
      console.log(`\nCosine similarity ${globalCosine.toFixed(6)} < ${COSINE_THRESHOLD}`);
      console.log('This indicates significant distribution drift between fp32 and fp16io encoder outputs.');
      console.log('The fp16 internal computation has drifted the hidden states enough to affect decoder behavior.');
    }
    if (!msePass) {
      console.log(`\nMSE ${globalMse.toExponential(4)} exceeds threshold ${MSE_THRESHOLD}`);
    }
  }

  // Cleanup
  await encFp32.release();
  await encFp16io.release();

  process.exit(pass ? 0 : 1);
}

main().catch(e => {
  console.error('FATAL:', e);
  process.exit(1);
});
