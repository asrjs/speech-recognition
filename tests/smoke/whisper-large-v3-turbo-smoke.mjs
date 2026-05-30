#!/usr/bin/env node
/**
 * Whisper large-v3-turbo fp32 smoke test (8GB-friendly sequential lifecycle).
 *
 * Loads encoder → runs → disposes, then loads decoders → decodes → disposes.
 * Peak memory: max(encoder, decoder_init + decoder_step), not sum of all.
 *
 * Requires: /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/fp32/
 *
 * Usage: node tests/smoke/whisper-large-v3-turbo-smoke.mjs [--audio <path>]
 */
import path from "node:path";
import fs from "node:fs";

async function main() {
  const { WhisperTokenizer, fetchText } = await import("../../dist/models/whisper-seq2seq/index.js");
  const { initWhisperOrt, createWhisperOrtSession } = await import("../../dist/models/whisper-seq2seq/ort.js");
  const { WhisperMelProcessor } = await import("../../dist/audio/whisper-mel.js");
  const { splitGraphDecodeLoop } = await import("../../dist/models/whisper-seq2seq/executor.js");
  const { WhisperTimestampLogitProcessor } = await import("../../dist/models/whisper-seq2seq/processors.js");
  const { parseWhisperGenerationConfig, parseWhisperModelConfig } = await import("../../dist/models/whisper-seq2seq/generation-config.js");

  const base = process.env.WHISPER_LARGE_V3_TURBO_DIR ?? "/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/fp32";
  const wavPath = process.argv.find(a => a === "--audio") ? process.argv[process.argv.indexOf("--audio") + 1] : "tests/fixtures/jfk2.en.wav";
  const sessOpts = (dataFile) => ({ backendId: "wasm", externalDataUrl: path.join(base, dataFile), externalDataPath: dataFile });

  console.log(`Model: ${base}`);
  console.log(`Audio: ${wavPath}`);

  const ort = await initWhisperOrt({ cpuThreads: 1 });
  const tokenizer = await WhisperTokenizer.fromUrl(path.join(base, "tokenizer.json"));
  const genConfig = parseWhisperGenerationConfig(JSON.parse(await fetchText(path.join(base, "generation_config.json"))));
  const configRaw = JSON.parse(await fetchText(path.join(base, "config.json")));
  const modelConfig = parseWhisperModelConfig(configRaw);

  // Decode WAV
  const buffer = fs.readFileSync(wavPath);
  const channels = buffer.readUInt16LE(22);
  const dataOffset = 44;
  const frameCount = Math.floor((buffer.length - dataOffset) / (2 * channels));
  const pcm = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) sum += buffer.readInt16LE(dataOffset + (i * channels + ch) * 2) / 32768;
    pcm[i] = sum / channels;
  }

  // Mel (large-v3-turbo uses 128 mel bins)
  const melBins = 128;
  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const melResult = melProc.process(pcm);
  const padded = WhisperMelProcessor.padToFrames(melResult, 3000);

  // === Phase 1: Encoder only ===
  console.log("Loading fp32 encoder (2.5GB external data)...");
  const enc = await createWhisperOrtSession(ort, path.join(base, "encoder_model.onnx"), sessOpts("encoder_model.onnx.data"));
  const featTensor = new ort.Tensor("float32", padded, [1, melBins, 3000]);
  const encOut = await enc.run({ input_features: featTensor });
  const encHs = encOut[Object.keys(encOut)[0]];
  console.log(`Encoder done — hidden states shape: [${encHs.dims.join(", ")}]`);
  await enc.release();
  console.log("Encoder disposed.");

  // === Phase 2: Decoders ===
  console.log("Loading fp32 decoders...");
  const decInit = await createWhisperOrtSession(ort, path.join(base, "decoder_init.onnx"), sessOpts("decoder_init.onnx.data"));
  const decStep = await createWhisperOrtSession(ort, path.join(base, "decoder_step.onnx"), sessOpts("decoder_step.onnx.data"));
  console.log("Decoders loaded.");

  const promptTokens = [
    tokenizer.getTokenId("<|startoftranscript|>") ?? 50258,
    tokenizer.getTokenId("<|en|>") ?? 50268,
    tokenizer.getTokenId("<|transcribe|>") ?? 50359,
  ];
  const eosId = tokenizer.getTokenId("<|endoftext|>") ?? 50257;
  const timestampBegin = tokenizer.getTokenId("<|0.00|>") ?? 50364;

  const tsProc = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId, noTimestampsTokenId: genConfig.noTimestampsTokenId ?? 50363,
    timestampBegin, suppressTokens: genConfig.suppressTokens ?? [],
    beginSuppressTokens: genConfig.beginSuppressTokens ?? [],
  });

  let kvDims = {};
  const started = performance.now();
  const result = await splitGraphDecodeLoop({
    promptTokens, encoderHiddenStates: encHs.data,
    eosTokenId: eosId, maxNewTokens: 200, modelConfig,
    processLogits: (logits, tokens, begin) => tsProc.process(logits, tokens, begin),
    runInit: async (prompt) => {
      const ids = new BigInt64Array(prompt.map(id => BigInt(id)));
      const out = await decInit.run({ input_ids: new ort.Tensor("int64", ids, [1, prompt.length]), encoder_hidden_states: encHs });
      const lk = Object.keys(out).find(k => k.includes("logits")) || Object.keys(out)[0];
      const lt = out[lk]; const vs = lt.dims[lt.dims.length - 1] || 0;
      const pkv = {}; kvDims = {};
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith("present")) { pkv[k] = v.data; kvDims[k] = v.dims; kvDims[k.replace(/^present\./, "past_key_values.")] = v.dims; }
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
      const lt = out[lk]; const vs = lt.dims[lt.dims.length - 1] || 0;
      const pkv = {};
      for (const [k, v] of Object.entries(out)) {
        if (k.startsWith("present")) { const pn = k.replace(/^present/, "past_key_values"); pkv[pn] = v.data; kvDims[pn] = v.dims; }
      }
      for (const [k, v] of Object.entries(pastKv)) { if (k.includes("encoder") && !pkv[k]) pkv[k] = v; }
      return { logits: lt.data, vocabSize: vs, presentKv: pkv };
    },
  });
  const elapsed = ((performance.now() - started) / 1000).toFixed(1);

  const text = tokenizer.decode([...promptTokens.slice(3), ...result.tokens], { skipSpecialTokens: true });
  console.log(`\nOutput (${elapsed}s, ${result.tokens.length} tokens):`);
  console.log(text.trim());

  await decInit.release();
  await decStep.release();
  console.log("\nLARGE-V3-TURBO FP32 SMOKE PASSED");
}

main().catch(e => { console.error(e.stack); process.exit(1); });
