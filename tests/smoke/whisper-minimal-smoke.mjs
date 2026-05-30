#!/usr/bin/env node
// Minimal whisper transcription — bypass executor, direct session bridge.
// Tests: can we get a correct transcription for JFK short audio?
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const MODEL = '/tmp/whisper-base-4graph/fp32';
const AUDIO = path.resolve(process.cwd(), 'tests/fixtures/JFK_Short.en.wav');
const EXPECT = 'country';

async function main() {
  console.log('=== Minimal Whisper Smoke ===\n');

  const { initWhisperOrt, createWhisperOrtSession } = await import('../../dist/models/whisper-seq2seq/ort.js');
  const { WhisperTokenizer } = await import('../../dist/models/whisper-seq2seq/tokenizer.js');
  const { WhisperMelProcessor } = await import('../../dist/audio/whisper-mel.js');
  const { default: ort } = await import('onnxruntime-web');

  await initWhisperOrt({ backend: 'wasm', cpuThreads: 2 });

  // Load model
  console.log('1. Loading sessions...');
  const enc = await createWhisperOrtSession(ort, path.join(MODEL, 'encoder_model.onnx'), { backendId: 'wasm' });
  const init = await createWhisperOrtSession(ort, path.join(MODEL, 'decoder_init.onnx'), { backendId: 'wasm' });
  const step = await createWhisperOrtSession(ort, path.join(MODEL, 'decoder_step.onnx'), { backendId: 'wasm' });
  const tok = await WhisperTokenizer.fromUrl(path.join(MODEL, 'tokenizer.json'));
  console.log('   done');

  // Load audio
  console.log('2. Loading audio...');
  const wav = readFileSync(AUDIO);
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  let off = 12, dataOff = 0, dataLen = 0, sr = 16000, nch = 1, bps = 16, fmtFound = false;
  while (off < view.byteLength - 8) {
    const id = String.fromCharCode(view.getUint8(off), view.getUint8(off+1), view.getUint8(off+2), view.getUint8(off+3));
    const sz = view.getUint32(off+4, true);
    if (id === 'fmt ') { nch = view.getUint16(off+10, true); sr = view.getUint32(off+12, true); bps = view.getUint16(off+22, true); fmtFound = true; }
    else if (id === 'data') { dataOff = off + 8; dataLen = sz; if (fmtFound) break; }
    off += 8 + sz;
  }
  const ns = dataLen / (bps/8);
  const pcm = new Float32Array(ns);
  const dv = new DataView(wav.buffer, wav.byteOffset + dataOff, dataLen);
  for (let i = 0; i < ns; i++) pcm[i] = dv.getInt16(i * 2, true) / 32768;
  const mono = nch === 1 ? pcm : pcm.filter((_, i) => i % nch === 0);
  console.log(`   ${path.basename(AUDIO)}: ${(mono.length/sr).toFixed(1)}s`);

  // Mel
  console.log('3. Mel...');
  const melProc = new WhisperMelProcessor({ nMels: 80 });
  const mel = WhisperMelProcessor.padToFrames(melProc.process(mono), 3000);

  // Encoder
  console.log('4. Encoder...');
  const encOut = await enc.run({ input_features: new ort.Tensor('float32', mel, [1, 80, 3000]) });
  const encKeys = Object.keys(encOut);
  const encData = new Float32Array(encOut[encKeys[0]].data.buffer, encOut[encKeys[0]].data.byteOffset, encOut[encKeys[0]].data.length);
  const encDims = encOut[encKeys[0]].dims;

  // Decode
  console.log('5. Decoding...');
  const prompt = [tok.getTokenId('<|startoftranscript|>') ?? 50258, tok.getTokenId('<|en|>') ?? 50268, tok.getTokenId('<|transcribe|>') ?? 50359];
  const eos = tok.getTokenId('<|endoftext|>') ?? 50257;

  // Init (splitgraph: only input_ids + encoder_hidden_states — NO past_key_values)
  const initOut = await init.run({
    input_ids: new ort.Tensor('int64', BigInt64Array.from(prompt.map(BigInt)), [1, prompt.length]),
    encoder_hidden_states: new ort.Tensor('float32', encData, encDims),
  });
  const logitsT = initOut.logits || initOut[Object.keys(initOut)[0]];
  // Splitgraph init: logits shape [1, promptLen, vocab]. Take last position.
  const vocab = logitsT.dims[2] ?? 51866;
  const promptLen = (logitsT.dims[1] ?? prompt.length);
  const lastPosOffset = (promptLen - 1) * vocab;

  const firstLogits = new Float32Array(logitsT.data.buffer, logitsT.data.byteOffset + lastPosOffset * 4, vocab);
  let maxVal = -Infinity, maxIdx = 0;
  for (let i = 0; i < vocab; i++) { if (firstLogits[i] > maxVal) { maxVal = firstLogits[i]; maxIdx = i; } }
  const tokens = [maxIdx];
  console.log(`   first token: ${maxIdx} ("${tok.decode([maxIdx])}")`);

  // Extract present KV from init output
  let pastKv = {};
  for (const [k, v] of Object.entries(initOut)) {
    if (k.startsWith('present.')) pastKv[k] = v;
  }

  // Autoregressive loop
  for (let s = 1; s < 200; s++) {
    const stepFeeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(tokens[tokens.length-1])]), [1, 1]),
    };
    for (const [k, v] of Object.entries(pastKv)) {
      const pkName = k.replace(/^present\./, 'past_key_values.');
      const data = new Float32Array(v.data.buffer, v.data.byteOffset, v.data.length);
      stepFeeds[pkName] = new ort.Tensor('float32', data, v.dims);
    }
    const stepOut = await step.run(stepFeeds);
    const sLogitsT = stepOut.logits || stepOut[Object.keys(stepOut)[0]];
    const sLogits = new Float32Array(sLogitsT.data.buffer, sLogitsT.data.byteOffset, vocab);
    maxVal = -Infinity; maxIdx = 0;
    for (let i = 0; i < vocab; i++) { if (sLogits[i] > maxVal) { maxVal = sLogits[i]; maxIdx = i; } }
    tokens.push(maxIdx);
    // Update pastKv: step output has decoder KV only, preserve encoder KV from input
    const newPastKv = {};
    for (const [k, v] of Object.entries(stepOut)) {
      if (k.startsWith('present.')) newPastKv[k] = v;
    }
    // Preserve encoder KV from previous iteration (step model doesn't output encoder KV)
    for (const [k, v] of Object.entries(pastKv)) {
      if (k.includes('.encoder.') && !newPastKv[k]) newPastKv[k] = v;
    }
    pastKv = newPastKv;
    if (maxIdx === eos) break;
  }

  const text = tok.decode(tokens.filter(t => t < 50364 || t > 50614));
  console.log(`\n   text: "${text}"`);
  console.log(`   tokens: ${tokens.length}`);

  const ok = text.toLowerCase().includes(EXPECT);
  console.log(ok ? `\n=== PASS (found "${EXPECT}") ===` : `\n=== FAIL (missing "${EXPECT}") ===`);
  process.exit(ok ? 0 : 1);
}
main().catch(e => { console.error(e); process.exit(1); });
