#!/usr/bin/env node
/**
 * Whisper large-v3-turbo fp32 persistent smoke — onnxruntime-node (native).
 *
 * PRIMARY DEVELOPMENT TARGET for large Whisper models.
 * Native ORT has no WASM heap limit — all sessions coexist.
 * Streaming-ready: encoder + decoder_init + decoder_step all loaded.
 *
 * Usage:
 *   node tests/smoke/whisper-large-v3-turbo-native.mjs [--fp16] [--mixed]
 *
 *   --mixed    Use q8 encoder + fp16 decoder (mixed precision, best perf)
 *   --fp16     Use fp16 variant (requires fp16 input conversion)
 *
 * Env vars:
 *   WHISPER_LARGE_DIR — model dir (default: .../fp32)
 */
import path from "node:path";
import fs from "node:fs";
import * as ort from "onnxruntime-node";

// Float32 → Float16 bit-preserving conversion for ORT
function float32ToFloat16Bits(f32) {
  const len = f32.length;
  const u16 = new Uint16Array(len);
  const view = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < len; i++) {
    view.setFloat32(0, f32[i]);
    const bits = view.getUint32(0);
    const sign = (bits >>> 16) & 0x8000;
    const exp = ((bits >>> 23) & 0xff) - 127 + 15;
    const mantissa = (bits >>> 13) & 0x3ff;
    if (exp <= 0) { u16[i] = sign; continue; }
    if (exp > 30) { u16[i] = sign | 0x7c00; continue; }
    u16[i] = sign | (exp << 10) | mantissa;
  }
  return u16;
}

async function main() {
  const useFp16 = process.argv.includes("--fp16");
  const useMixed = process.argv.includes("--mixed");
  const variant = useMixed ? "mixed" : (useFp16 ? "fp16" : "fp32");
  const base = useMixed
    ? (process.env.WHISPER_LARGE_DIR ?? "/tmp/whisper-mixed-q8-enc-f16-dec")
    : (process.env.WHISPER_LARGE_DIR ?? `/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/${variant}`);
  const wavPath = process.argv.find(a => a.endsWith(".wav")) ?? "tests/fixtures/jfk2.en.wav";

  const { WhisperTokenizer, fetchText } = await import("../../dist/models/whisper-seq2seq/index.js");
  const { WhisperMelProcessor } = await import("../../dist/audio/whisper-mel.js");
  const { splitGraphDecodeLoop } = await import("../../dist/models/whisper-seq2seq/executor.js");
  const { WhisperTimestampLogitProcessor } = await import("../../dist/models/whisper-seq2seq/processors.js");
  const { parseWhisperGenerationConfig, parseWhisperModelConfig } = await import("../../dist/models/whisper-seq2seq/generation-config.js");

  console.log(`Model: ${base} (onnxruntime-node, persistent, ${variant})`);
  console.log(`Audio: ${wavPath}`);

  const tokenizer = await WhisperTokenizer.fromUrl(path.join(base, "tokenizer.json"));
  const genConfig = parseWhisperGenerationConfig(JSON.parse(await fetchText(path.join(base, "generation_config.json"))));
  const configRaw = JSON.parse(await fetchText(path.join(base, "config.json")));
  const modelConfig = parseWhisperModelConfig(configRaw);
  const melBins = modelConfig.numMelBins ?? (useFp16 ? 128 : 128);

  // Decode WAV
  const buffer = fs.readFileSync(wavPath);
  const channels = buffer.readUInt16LE(22);
  const frameCount = Math.floor((buffer.length - 44) / (2 * channels));
  const pcm = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) sum += buffer.readInt16LE(44 + (i * channels + ch) * 2) / 32768;
    pcm[i] = sum / channels;
  }

  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const padded = WhisperMelProcessor.padToFrames(melProc.process(pcm), 3000);

  // PERSISTENT: all sessions loaded together (streaming-ready)
  console.log("Loading sessions...");
  const t0 = performance.now();
  const enc = await ort.InferenceSession.create(path.join(base, "encoder_model.onnx"));
  console.log(`  encoder (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
  const t1 = performance.now();
  const decInit = await ort.InferenceSession.create(path.join(base, "decoder_init.onnx"));
  console.log(`  decoder_init (${((performance.now() - t1) / 1000).toFixed(1)}s)`);
  const t2 = performance.now();
  const decStep = await ort.InferenceSession.create(path.join(base, "decoder_step.onnx"));
  console.log(`  decoder_step (${((performance.now() - t2) / 1000).toFixed(1)}s)`);
  console.log("All persistent — streaming-ready.");

  // Encoder
  const featTensor = new ort.Tensor("float32", padded, [1, melBins, 3000]);
  const encOut = await enc.run({ input_features: featTensor });
  const encHs = encOut[Object.keys(encOut)[0]];

  // Prompt tokens
  const promptTokens = [
    tokenizer.getTokenId("<|startoftranscript|>") ?? 50258,
    tokenizer.getTokenId("<|en|>") ?? 50268,
    tokenizer.getTokenId("<|transcribe|>") ?? 50359,
  ];
  const eosId = tokenizer.getTokenId("<|endoftext|>") ?? 50257;

  const tsProc = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId,
    noTimestampsTokenId: genConfig.noTimestampsTokenId ?? 50363,
    timestampBegin: tokenizer.getTokenId("<|0.00|>") ?? 50364,
    suppressTokens: genConfig.suppressTokens ?? [],
    beginSuppressTokens: genConfig.beginSuppressTokens ?? [],
  });

  // Decode
  let kvDims = {};
  const decodeStart = performance.now();
  const result = await splitGraphDecodeLoop({
    promptTokens, encoderHiddenStates: encHs.data, eosTokenId: eosId,
    maxNewTokens: 200, modelConfig,
    processLogits: (l, t, b) => tsProc.process(l, t, b),
    runInit: async (prompt) => {
      const ids = new BigInt64Array(prompt.map(id => BigInt(id)));
      const out = await decInit.run({
        input_ids: new ort.Tensor("int64", ids, [1, prompt.length]),
        encoder_hidden_states: encHs,
      });
      const lk = Object.keys(out).find(k => k.includes("logits")) || Object.keys(out)[0];
      const lt = out[lk];
      const vs = lt.dims[lt.dims.length - 1] || 0;
      const pkv = {};
      kvDims = {};
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith("present")) {
          pkv[k] = v.data;
          kvDims[k] = v.dims;
          kvDims[k.replace(/^present\./, "past_key_values.")] = v.dims;
        }
      }
      return { logits: lt.data, vocabSize: vs, presentKv: pkv };
    },
    runStep: async (tokenId, pastKv) => {
      const feeds = { input_ids: new ort.Tensor("int64", new BigInt64Array([BigInt(tokenId)]), [1, 1]) };
      for (const [name, data] of Object.entries(pastKv)) {
        const sn = name.replace(/^present\./, "past_key_values.");
        const dims = kvDims[name] || kvDims[sn] || kvDims[name.replace(/^past_key_values\./, "present.")];
        if (dims) feeds[sn] = new ort.Tensor("float32", new Float32Array(data), dims);
      }
      const out = await decStep.run(feeds);
      const lk = Object.keys(out).find(k => k.includes("logits")) || Object.keys(out)[0];
      const lt = out[lk];
      const vs = lt.dims[lt.dims.length - 1] || 0;
      const pkv = {};
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith("present")) {
          const pn = k.replace(/^present/, "past_key_values");
          pkv[pn] = v.data;
          kvDims[pn] = v.dims;
        }
      }
      for (const [k, v] of Object.entries(pastKv)) {
        if (k.includes("encoder") && !pkv[k]) pkv[k] = v;
      }
      return { logits: lt.data, vocabSize: vs, presentKv: pkv };
    },
  });

  const decodeSec = ((performance.now() - decodeStart) / 1000).toFixed(1);
  const text = tokenizer.decode([...promptTokens.slice(3), ...result.tokens], { skipSpecialTokens: true });
  console.log(`\nDecode: ${decodeSec}s, ${result.tokens.length} tokens`);
  console.log(text.trim());

  await Promise.all([enc.release(), decInit.release(), decStep.release()]);

  // Verify transcription
  const expected = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim();
  const match = norm(text).includes(norm(expected).substring(0, 40));
  if (!match) {
    console.error("\nTRANSCRIPTION MISMATCH — expected JFK text");
    process.exit(1);
  }

  const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`\nLARGE-V3-TURBO ${variant.toUpperCase()} NATIVE SMOKE PASSED (${totalSec}s total)`);
}

main().catch(e => { console.error(e.stack); process.exit(1); });
