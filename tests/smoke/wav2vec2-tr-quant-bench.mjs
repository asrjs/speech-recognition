#!/usr/bin/env node
/**
 * Wav2Vec2 Turkish quantization benchmark.
 * Compares fp32, fp16, q8, and ORT-optimized fp32 variants.
 *
 * Metrics:
 *   - Model size (MB)
 *   - External data (yes/no)
 *   - Session creation time (ms)
 *   - Inference time (ms) — 3 runs, take median
 *   - WER/CER against reference text
 *
 * Usage: node tests/smoke/wav2vec2-tr-quant-bench.mjs
 */
import { readFileSync, statSync } from 'node:fs';
import { basename } from 'node:path';
import { pathToFileURL } from 'node:url';

// ── Models ──
const VARIANTS = [
  {
    name: 'fp32',
    model: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.onnx',
    data: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.onnx.data',
  },
  {
    name: 'fp16',
    model: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.fp16.onnx',
    data: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.fp16.onnx.data',
  },
  {
    name: 'q8',
    model: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.q8.onnx',
    data: null, // inline
  },
  {
    name: 'opt-fp32',
    model: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.opt.onnx',
    data: null, // inline after optimizer
  },
];

// ── Turkish reference text (from sidecar JSON) ──
const REFERENCE =
  'bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar o kadar korkunç ve tahrip ediciydi ki bu salgınlar neticesinde cemiyet fonksiyonları altüst olmakta ülkelerin sosyal ve ekonomik gelişmeleri yıllarca duraklamakta idi';

function tokenize(text) {
  return text.toLowerCase().replace(/[.,!?;:]/g, '').split(/\s+/).filter(Boolean);
}

function wer(reference, hypothesis) {
  const ref = tokenize(reference);
  const hyp = tokenize(hypothesis);
  const m = ref.length;
  const n = hyp.length;
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n] / m;
}

function cer(reference, hypothesis) {
  const ref = reference.toLowerCase().replace(/\s+/g, '').replace(/[.,!?;:]/g, '');
  const hyp = hypothesis.toLowerCase().replace(/\s+/g, '').replace(/[.,!?;:]/g, '');
  const m = ref.length;
  const n = hyp.length;
  const dp = Array.from({ length: m + 1 }, () => new Int32Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = ref[i - 1] === hyp[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n] / m;
}

function decodeWav(buf) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  if (view.getUint32(0) !== 0x52494646) throw new Error('Not RIFF');
  let off = 12, fmt = null, data = null;
  while (off + 8 <= buf.byteLength) {
    const ckId = String.fromCharCode(...new Uint8Array(buf.buffer, buf.byteOffset + off, 4));
    const ckSize = view.getUint32(off + 4, true);
    if (ckId === 'fmt ') {
      fmt = { channels: view.getUint16(off + 10, true), sampleRate: view.getUint32(off + 12, true), bps: view.getUint16(off + 22, true) };
    } else if (ckId === 'data') {
      data = { off: off + 8, size: ckSize };
    }
    off += 8 + ckSize + (ckSize % 2);
  }
  if (!fmt || !data) throw new Error('Bad WAV');
  const bps8 = fmt.bps / 8;
  const fc = Math.floor(data.size / (bps8 * fmt.channels));
  const pcm = new Float32Array(fc);
  for (let f = 0; f < fc; f++) {
    let s = 0;
    for (let c = 0; c < fmt.channels; c++) s += view.getInt16(data.off + (f * fmt.channels + c) * bps8, true) / 32768;
    pcm[f] = s / fmt.channels;
  }
  return { pcm, sampleRate: fmt.sampleRate, durSec: fc / fmt.sampleRate };
}

async function main() {
  const wavBuf = readFileSync('tests/fixtures/019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav');
  const audio = decodeWav(wavBuf);

  console.log(`\n=== Wav2Vec2 Turkish Quantization Benchmark ===`);
  console.log(`Audio: 18.6s, 16kHz, mono`);
  console.log(`Reference: "${REFERENCE.slice(0, 80)}..."\n`);

  const results = [];

  for (const variant of VARIANTS) {
    console.log(`--- ${variant.name} ---`);

    const modelPath = variant.model;
    const modelStat = statSync(modelPath);
    let dataSize = 0;
    if (variant.data) {
      try { dataSize = statSync(variant.data).size; } catch {}
    }
    const totalMb = (modelStat.size + dataSize) / (1024 * 1024);

    try {
      // Create ORT session directly (bypass executor for dtype control)
      const { default: ort } = await import('onnxruntime-web');

      // Configure WASM paths
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;

      const sessionOpts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      if (!variant.data) {
        // inline model — external data not needed
      } else {
        // need to resolve paths for Node.js
        sessionOpts.externalData = [{
          data: variant.data,
          path: basename(variant.data),
        }];
      }

      // Load session
      const loadStart = performance.now();
      const session = await ort.InferenceSession.create(modelPath, sessionOpts);
      const loadMs = performance.now() - loadStart;

      // Load tokenizer
      const { Wav2Vec2CharTokenizer } = await import('../../dist/models/wav2vec2/tokenizer.js');
      const tokenizer = await Wav2Vec2CharTokenizer.fromUrl(
        'https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-turkish/resolve/main/vocab.json'
      );

      // Determine input dtype
      const inputDtype = variant.name === 'fp16' ? 'float16' : 'float32';
      const isFloat16 = inputDtype === 'float16';

      // Run inference 3 times
      const runMsList = [];
      let transcript = '';
      for (let run = 0; run < 3; run++) {
        let inputTensor;
        if (isFloat16) {
          // Manual float32 → float16 conversion
          const samples = audio.pcm.length;
          const f16 = new Uint16Array(samples);
          const f32 = new Float32Array(1);
          const u16 = new Uint16Array(f32.buffer);
          for (let i = 0; i < samples; i++) {
            f32[0] = audio.pcm[i];
            // Extract float16 from float32 bits
            const f32bits = u16[1] << 16 | u16[0];
            const sign = (f32bits >>> 31) & 1;
            let exp = (f32bits >>> 23) & 0xff;
            let mant = f32bits & 0x7fffff;
            if (exp === 0) {
              f16[i] = sign << 15; // zero/subnormal → zero
            } else if (exp === 0xff) {
              f16[i] = (sign << 15) | 0x7c00 | ((mant >> 13) & 0x3ff); // NaN/Inf
            } else {
              exp = exp - 127 + 15;
              if (exp >= 31) {
                f16[i] = (sign << 15) | 0x7c00; // overflow → Inf
              } else if (exp <= 0) {
                f16[i] = sign << 15; // underflow → zero
              } else {
                f16[i] = (sign << 15) | (exp << 10) | ((mant >> 13) & 0x3ff);
              }
            }
          }
          inputTensor = new ort.Tensor('float16', f16, [1, samples]);
        } else {
          inputTensor = new ort.Tensor('float32', audio.pcm, [1, audio.pcm.length]);
        }

        const t0 = performance.now();
        const outputs = await session.run({ input_values: inputTensor });
        const logitsTensor = outputs.logits;
        const [_, frames, vocabSize] = logitsTensor.dims;

        // fp16 logits need float16→float32 conversion
        let logits;
        if (isFloat16) {
          const f16data = new Uint16Array(logitsTensor.data.buffer, logitsTensor.data.byteOffset, frames * vocabSize);
          logits = new Float32Array(frames * vocabSize);
          const tmp32 = new Float32Array(1);
          const tmp16 = new Uint16Array(tmp32.buffer);
          for (let i = 0; i < f16data.length; i++) {
            // float16 → float32
            const half = f16data[i];
            const sign = (half >>> 15) & 1;
            const exp = (half >>> 10) & 0x1f;
            const mant = half & 0x3ff;
            if (exp === 0) {
              tmp16[0] = 0; tmp16[1] = sign << 15;
            } else if (exp === 0x1f) {
              tmp16[0] = 0; tmp16[1] = (sign << 15) | 0x7f80 | mant;
            } else {
              tmp16[0] = 0; tmp16[1] = (sign << 15) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            logits[i] = tmp32[0];
          }
        } else {
          logits = new Float32Array(logitsTensor.data.buffer, logitsTensor.data.byteOffset, frames * vocabSize);
        }

        // CTC decode
        const tokens = [];
        for (let f = 0; f < frames; f++) {
          let bestId = 0, bestVal = -Infinity;
          for (let v = 0; v < vocabSize; v++) {
            if (logits[f * vocabSize + v] > bestVal) { bestVal = logits[f * vocabSize + v]; bestId = v; }
          }
          tokens.push(bestId);
        }
        const collapsed = [];
        for (const t of tokens) {
          if (t === 0) continue; // blank
          if (collapsed.length > 0 && collapsed[collapsed.length - 1] === t) continue;
          collapsed.push(t);
        }
        const chars = collapsed.map(id => tokenizer.decodeTokenPiece(id) ?? '').join('');
        transcript = chars.replace(/\|/g, ' ').trim();

        runMsList.push(performance.now() - t0);
      }

      runMsList.sort((a, b) => a - b);
      const inferenceMs = runMsList[1];

      const w = wer(REFERENCE, transcript);
      const c = cer(REFERENCE, transcript);

      results.push({
        name: variant.name,
        sizeMb: totalMb.toFixed(0),
        externalData: !!variant.data,
        loadMs: loadMs.toFixed(0),
        inferenceMs: inferenceMs.toFixed(0),
        wer: (w * 100).toFixed(1),
        cer: (c * 100).toFixed(1),
        transcript: transcript.slice(0, 90) + '...',
      });

      console.log(`  Size: ${totalMb.toFixed(0)} MB, Load: ${loadMs.toFixed(0)}ms, Infer(median): ${inferenceMs.toFixed(0)}ms`);
      console.log(`  WER: ${(w * 100).toFixed(1)}%, CER: ${(c * 100).toFixed(1)}%`);
      console.log(`  Text: ${transcript.slice(0, 90)}...`);
    } finally {
      // session auto-disposed
    }
    console.log();
  }

  // Print table
  console.log('=== RESULTS TABLE ===');
  console.log('');
  console.log('| Variant | Size (MB) | Ext Data | Load (ms) | Infer (ms) | WER (%) | CER (%) |');
  console.log('|---------|-----------|----------|-----------|------------|---------|---------|');
  for (const r of results) {
    console.log(`| ${r.name.padEnd(8)}| ${r.sizeMb.padStart(6)} | ${r.externalData ? 'yes' : 'no '}     | ${r.loadMs.padStart(6)} | ${r.inferenceMs.padStart(8)} | ${r.wer.padStart(6)} | ${r.cer.padStart(6)} |`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
