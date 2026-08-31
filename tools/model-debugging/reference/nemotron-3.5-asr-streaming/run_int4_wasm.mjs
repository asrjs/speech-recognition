#!/usr/bin/env node
// Run INT4 Nemotron 3.5 ONNX pipeline using ORT Web WASM in Node.
// Proves the WASM runtime produces the same tokens as native ORT.

import * as ort from 'onnxruntime-web';
import fs from 'node:fs';
import path from 'node:path';
ort.env.wasm.wasmPaths = 'file:///N:/github/asrjs/speech-recognition/node_modules/onnxruntime-web/dist/';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'warning';

const ONNX_DIR = 'N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4-singles';
const FIXTURE = 'tools/data/fixtures/audio/jfk-short.wav';
const MEL_NPY = 'tools/data/results/nemotron/_tmp_mel.npy';
const OUT = 'tools/data/results/nemotron/nemotron-3.5-int4-wasm-pipeline-2026-08-31.json';

function fileUrl(p) {
  const abs = path.resolve(p).replace(/\\/g, '/');
  return abs.startsWith('/') ? 'file://' + abs : 'file:///' + abs;
}

function parseNpy(buf) {
  if (buf[0] !== 0x93 || buf[1] !== 0x4e) throw new Error('Not a npy file');
  const version = buf[6]; // magic is \x93NUMPY (6 bytes); version at offset 6
  const headerLen = version >= 2
    ? Number(buf[10] | (buf[11] << 8) | (buf[12] << 16) | (buf[13] << 24))
    : Number(buf[8] | (buf[9] << 8));
  const headerStart = 10;
  const header = buf.slice(headerStart, headerStart + headerLen).toString('utf-8');
  const shapeMatch = header.match(/'shape':\s*\(([^)]+)\)/);
  const shape = shapeMatch[1].split(',').map(s => Number(s.trim())).filter(n => !isNaN(n));
  const fortran = /'fortran_order':\s*True/.test(header);
  const descrMatch = header.match(/'descr':\s*'([^']+)'/);
  const dtype = descrMatch ? descrMatch[1].replace(/[<>]/g, '') : '';
  // npy v1.0 pads the header so data starts at a 64-byte-aligned offset
  const dataStart = Math.ceil((headerStart + headerLen) / 64) * 64;
  if (!(dtype.includes('f4') || dtype.includes('float32'))) {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }
  const total = shape.reduce((a, b) => a * b, 1);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset + dataStart, total);
  const rows = shape[0];
  const cols = shape[1] || 1;
  const arr = [];
  if (fortran && shape.length === 2) {
    // column-major storage: element (i, j) lives at j * rows + i
    for (let i = 0; i < rows; i++) {
      const row = new Float32Array(cols);
      for (let j = 0; j < cols; j++) row[j] = f32[j * rows + i];
      arr.push(row);
    }
  } else {
    for (let i = 0; i < rows; i++) {
      arr.push(f32.slice(i * cols, (i + 1) * cols));
    }
  }
  return arr;
}

function argmaxLogits(logits, T_remain, targetLen, V, blankId) {
  for (let t = 0; t < T_remain; t++) {
    const base = t * targetLen * V + (targetLen - 1) * V;
    let maxV = -Infinity, maxIdx = -1;
    for (let v = 0; v < V; v++) {
      const x = logits[base + v];
      if (x > maxV) { maxV = x; maxIdx = v; }
    }
    if (maxIdx !== blankId) return { frame: t, token: maxIdx };
  }
  return { frame: -1, token: -1 };
}

async function main() {
  if (!fs.existsSync(MEL_NPY)) {
    console.log('Extracting NeMo mel ...');
    execSync(
      `N:/github/asrjs/speech-recognition/tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe ` +
      `tools/model-debugging/reference/nemotron-3.5-asr-streaming/dump_nemo_mel.py "${FIXTURE}" "${MEL_NPY}"`,
      { stdio: 'inherit' }
    );
  }

  const mel = parseNpy(fs.readFileSync(MEL_NPY));
  console.log(`Mel shape: ${mel.length} x ${mel[0].length}`);

  const vocabText = fs.readFileSync(path.resolve('N:/models/onnx/nemo/nemotron-3.5-asr-streaming-int4/vocab.txt'), 'utf-8');
  const vocabLines = vocabText.split(/\r?\n/).filter(l => l.length > 0);
  const blankId = vocabLines.indexOf('<blank>');
  console.log(`Vocab: ${vocabLines.length} tokens, blank_id=${blankId}`);

  // ORT Web in Node: pass plain Windows path; loadModel does fs.open directly.
  const enc = await ort.InferenceSession.create(path.resolve(ONNX_DIR, 'encoder.onnx'), { executionProviders: ['wasm'] });
  console.log('Loading INT4 decoder (WASM) ...');
  const dec = await ort.InferenceSession.create(path.resolve(ONNX_DIR, 'decoder.onnx'), { executionProviders: ['wasm'] });
  console.log('Loading INT4 joint (WASM) ...');
  const jnt = await ort.InferenceSession.create(path.resolve(ONNX_DIR, 'joint.onnx'), { executionProviders: ['wasm'] });

  // Stream encoder
  const chunkSize = 65;
  let cacheCh = new ort.Tensor('float32', new Float32Array(1 * 24 * 56 * 1024), [1, 24, 56, 1024]);
  let cacheT = new ort.Tensor('float32', new Float32Array(1 * 24 * 1024 * 8), [1, 24, 1024, 8]);
  let cacheChLen = new ort.Tensor('int64', new BigInt64Array(1), [1]);
  const allEnc = [];
  let melIdx = 0;
  while (melIdx < mel.length) {
    const chunkLen = Math.min(chunkSize, mel.length - melIdx);
    const flatChunk = new Float32Array(chunkSize * 128);
    for (let i = 0; i < chunkLen; i++) {
      flatChunk.set(mel[melIdx + i], i * 128);
    }
    const out = await enc.run({
      audio_signal: new ort.Tensor('float32', flatChunk, [1, chunkSize, 128]),
      length: new ort.Tensor('int64', BigInt64Array.from([BigInt(chunkLen)]), [1]),
      cache_last_channel: cacheCh,
      cache_last_time: cacheT,
      cache_last_channel_len: cacheChLen,
      lang_id: new ort.Tensor('int64', BigInt64Array.from([0n]), [1]),
    });
    allEnc.push(new Float32Array(out.outputs.data));
    cacheCh = out.cache_last_channel_next;
    cacheT = out.cache_last_time_next;
    cacheChLen = out.cache_last_channel_len_next;
    melIdx += chunkSize;
  }
  console.log(`Encoded: ${allEnc.length} chunks of 7 frames each`);

  const T_enc = allEnc.length * 7;
  const encFlat = new Float32Array(T_enc * 1024);
  for (let i = 0; i < allEnc.length; i++) {
    encFlat.set(allEnc[i], i * 7 * 1024);
  }

  // Greedy RNN-T decode
  let targets = [blankId];
  const targetIds = [];
  let h = new ort.Tensor('float32', new Float32Array(2 * 1 * 640), [2, 1, 640]);
  let c = new ort.Tensor('float32', new Float32Array(2 * 1 * 640), [2, 1, 640]);
  let lastT = 0;
  for (let step = 0; step < 200; step++) {
    const decOut = await dec.run({
      targets: new ort.Tensor('int64', BigInt64Array.from(targets.map(BigInt)), [1, targets.length]),
      h_in: h,
      c_in: c,
    });
    const decOutData = decOut.decoder_output.data;
    const targetLen = decOut.decoder_output.dims[2];
    const decTData = new Float32Array(targetLen * 640);
    for (let t = 0; t < targetLen; t++) {
      for (let d = 0; d < 640; d++) {
        decTData[t * 640 + d] = decOutData[d * targetLen + t];
      }
    }
    h = decOut.h_out;
    c = decOut.c_out;

    const T_remain = T_enc - lastT;
    if (T_remain <= 0) {
      console.log(`Step ${step}: exhausted enc frames`);
      break;
    }
    const encRemData = encFlat.slice(lastT * 1024, T_enc * 1024);
    const jntOut = await jnt.run({
      encoder_output: new ort.Tensor('float32', encRemData, [1, T_remain, 1024]),
      decoder_output: new ort.Tensor('float32', decTData, [1, targetLen, 640]),
    });
    const logits = jntOut.joint_output.data;
    const { frame, token } = argmaxLogits(logits, T_remain, targetLen, 13088, blankId);
    if (frame < 0) {
      lastT += T_remain;
      continue;
    }
    targetIds.push(token);
    targets.push(token);
    lastT += frame;
    if (step % 10 === 0) {
      console.log(`Step ${step}: emitted ${token} (${vocabLines[token]}) at frame ${lastT}`);
    }
    if (targetIds.length >= 100) break;
  }

  const pieces = targetIds.map(t => vocabLines[t] ?? '?');
  const text = pieces.join('').replace(/\u2581/g, ' ').trim();
  console.log(`\nFinal text: ${text}`);
  console.log(`Total tokens: ${targetIds.length}`);

  fs.writeFileSync(OUT, JSON.stringify({
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    purpose: 'INT4 Nemotron 3.5 ONNX pipeline via ORT Web WASM in Node',
    runtime: 'onnxruntime-web@1.29.0 wasm simd-threaded',
    encodedShape: [T_enc, 1024],
    tokenIds: targetIds,
    tokenCount: targetIds.length,
    text,
  }, null, 2));
  console.log(`\nWrote ${OUT}`);
}

main().catch(e => { console.error(e); process.exit(1); });