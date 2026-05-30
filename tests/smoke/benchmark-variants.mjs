#!/usr/bin/env node
// Benchmark all large-v3-turbo variants
import path from "node:path";
import fs from "node:fs";
import * as ort from "onnxruntime-node";

const { WhisperMelProcessor } = await import("../../dist/audio/whisper-mel.js");

const buffer = fs.readFileSync("tests/fixtures/jfk2.en.wav");
const channels = buffer.readUInt16LE(22);
const frameCount = Math.floor((buffer.length - 44) / (2 * channels));
const pcm = new Float32Array(frameCount);
for (let i = 0; i < frameCount; i++) {
  let sum = 0;
  for (let ch = 0; ch < channels; ch++) sum += buffer.readInt16LE(44 + (i * channels + ch) * 2) / 32768;
  pcm[i] = sum / channels;
}
const melProc = new WhisperMelProcessor({ nMels: 128 });
const padded = WhisperMelProcessor.padToFrames(melProc.process(pcm), 3000);

async function benchmark(label, base) {
  const enc = await ort.InferenceSession.create(base + "/encoder_model.onnx");
  const decInit = await ort.InferenceSession.create(base + "/decoder_init.onnx");
  const decStep = await ort.InferenceSession.create(base + "/decoder_step.onnx");

  let decodeTimes = [];

  for (let iter = 0; iter < 3; iter++) {
    let t0 = performance.now();

    const encOut = await enc.run({ input_features: new ort.Tensor("float32", padded, [1, 128, 3000]) });
    const encHs = encOut[Object.keys(encOut)[0]];

    const prompt = [50258, 50268, 50359];
    const initOut = await decInit.run({
      input_ids: new ort.Tensor("int64", new BigInt64Array(prompt.map(BigInt)), [1, 3]),
      encoder_hidden_states: encHs,
    });

    const lk = Object.keys(initOut).find(k => k.includes("logits"));
    const vs = initOut[lk].dims[initOut[lk].dims.length - 1];
    const lastRow = initOut[lk].data.slice(initOut[lk].data.length - vs);
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < lastRow.length; i++) { if (lastRow[i] > maxVal) { maxVal = lastRow[i]; maxIdx = i; } }
    let token = maxIdx;
    let pastKv = {}, kvDims = {};
    for (const [k, v] of Object.entries(initOut)) {
      if (k.startsWith("present")) {
        const pn = k.replace(/^present\./, "past_key_values.");
        pastKv[pn] = new Float32Array(v.data);
        kvDims[pn] = v.dims;
      }
    }

    let steps = 0;
    while (token !== 50257 && steps < 50) {
      const feeds = { input_ids: new ort.Tensor("int64", new BigInt64Array([BigInt(token)]), [1, 1]) };
      for (const [name, data] of Object.entries(pastKv)) {
        feeds[name] = new ort.Tensor("float32", data, kvDims[name]);
      }
      const out = await decStep.run(feeds);
      const lk2 = Object.keys(out).find(k => k.includes("logits"));
      const vs2 = out[lk2].dims[out[lk2].dims.length - 1];
      const lr2 = out[lk2].data.slice(out[lk2].data.length - vs2);
      maxIdx = 0; maxVal = -Infinity;
      for (let i = 0; i < lr2.length; i++) { if (lr2[i] > maxVal) { maxVal = lr2[i]; maxIdx = i; } }
      token = maxIdx;
      steps++;
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith("present")) {
          const pn = k.replace(/^present/, "past_key_values");
          pastKv[pn] = new Float32Array(v.data);
          kvDims[pn] = v.dims;
        }
      }
    }
    decodeTimes.push(performance.now() - t0);
  }

  const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
  console.log(`  ${avg(decodeTimes).toFixed(0)}ms total (3-run avg, ${label})`);
  await Promise.all([enc.release(), decInit.release(), decStep.release()]);
}

console.log("=== large-v3-turbo variant benchmark ===");
console.log("Audio: JFK (11s) on onnxruntime-node CPU\n");

console.log("1. FP32 (baseline):");
await benchmark("encoder fp32 + decoder fp32", "/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/fp32");

console.log("2. Q8 (full quantized):");
await benchmark("encoder q8 + decoder q8", "/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8");

console.log("3. MIXED (recommended):");
await benchmark("encoder q8 + decoder fp32", "/tmp/whisper-mixed-q8-enc-fp32-dec");
