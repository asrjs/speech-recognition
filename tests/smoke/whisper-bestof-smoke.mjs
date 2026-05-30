#!/usr/bin/env node
import path from "node:path";
import fs from "node:fs";

async function main() {
  const { WhisperTokenizer, fetchText } = await import("../../dist/models/whisper-seq2seq/index.js");
  const { createWhisperOrtSession, initWhisperOrt } = await import("../../dist/models/whisper-seq2seq/ort.js");
  const { WhisperMelProcessor } = await import("../../dist/audio/whisper-mel.js");
  const { splitGraphDecodeLoop } = await import("../../dist/models/whisper-seq2seq/executor.js");
  const { WhisperTimestampLogitProcessor } = await import("../../dist/models/whisper-seq2seq/processors.js");
  const { parseWhisperGenerationConfig, parseWhisperModelConfig } = await import("../../dist/models/whisper-seq2seq/generation-config.js");

  const baseDir = "/tmp/whisper-base-4graph/fp32";
  const wavPath = "tests/fixtures/jfk2.en.wav";

  const ort = await initWhisperOrt({ wasmPaths: undefined, cpuThreads: 1 });

  const sessionOpts = { backendId: 'wasm' };
  const encoderSession = await createWhisperOrtSession(ort, path.join(baseDir, "encoder_model.onnx"), sessionOpts);
  const decInitSession = await createWhisperOrtSession(ort, path.join(baseDir, "decoder_init.onnx"), sessionOpts);
  const decStepSession = await createWhisperOrtSession(ort, path.join(baseDir, "decoder_step.onnx"), sessionOpts);

  const tokenizer = await WhisperTokenizer.fromUrl(path.join(baseDir, "tokenizer.json"));
  const genConfigRaw = JSON.parse(await fetchText(path.join(baseDir, "generation_config.json")));
  const configRaw = JSON.parse(await fetchText(path.join(baseDir, "config.json")));

  const genConfig = parseWhisperGenerationConfig(genConfigRaw);
  const modelConfig = parseWhisperModelConfig(configRaw);
  if (!modelConfig.numMelBins && typeof genConfigRaw.num_mel_bins === "number") {
    modelConfig.numMelBins = genConfigRaw.num_mel_bins;
  }

  // Decode WAV
  const buffer = fs.readFileSync(wavPath);
  const channels = buffer.readUInt16LE(22);
  const bitsPerSample = buffer.readUInt16LE(34);
  const dataOffset = 44;
  const dataLen = buffer.length - dataOffset;
  const bytesPerSample = bitsPerSample / 8;
  const frameCount = Math.floor(dataLen / (bytesPerSample * channels));
  const pcm = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) {
      sum += buffer.readInt16LE(dataOffset + (i * channels + ch) * 2) / 32768;
    }
    pcm[i] = sum / channels;
  }

  // Mel
  const melBins = modelConfig.numMelBins ?? 80;
  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const melResult = melProc.process(pcm);
  const melInputFrames = 3000;
  const padded = WhisperMelProcessor.padToFrames(melResult, melInputFrames);

  const featTensor = new ort.Tensor("float32", padded, [1, melBins, melInputFrames]);
  const encOut = await encoderSession.run({ input_features: featTensor });
  const encHs = encOut[Object.keys(encOut)[0]];

  // Prompt
  const promptTokens = [
    tokenizer.getTokenId("<|startoftranscript|>") ?? 50258,
    tokenizer.getTokenId("<|en|>") ?? 50268,
    tokenizer.getTokenId("<|transcribe|>") ?? 50359,
  ];

  const eosId = tokenizer.getTokenId("<|endoftext|>") ?? 50257;
  const timestampBegin = tokenizer.getTokenId("<|0.00|>") ?? 50364;

  const tsProc = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId,
    noTimestampsTokenId: genConfig.noTimestampsTokenId ?? 50363,
    timestampBegin,
    suppressTokens: genConfig.suppressTokens ?? [],
    beginSuppressTokens: genConfig.beginSuppressTokens ?? [],
  });

  let kvDims = {};

  // Helper to create standard runInit/runStep callbacks
  function makeCallbacks() {
    const state = { kvDims: {} };
    return {
      get kvDims() { return state.kvDims; },
      set kvDims(v) { state.kvDims = v; },
      runInit: async (prompt, _encHs, _dims) => {
        const inputIds = new BigInt64Array(prompt.map((id) => BigInt(id)));
        const inputIdsTensor = new ort.Tensor("int64", inputIds, [1, prompt.length]);
        const out = await decInitSession.run({ input_ids: inputIdsTensor, encoder_hidden_states: encHs });
        const logitsKey = Object.keys(out).find((k) => k.includes("logits")) || Object.keys(out)[0];
        const logitsTensor = out[logitsKey];
        const vocabSize = logitsTensor.dims[logitsTensor.dims.length - 1] || 0;
        const presentKv = {};
        state.kvDims = {};
        for (const [k, v] of Object.entries(out)) {
          if (k.startsWith("present")) {
            presentKv[k] = v.data;
            state.kvDims[k] = v.dims;
            state.kvDims[k.replace(/^present\./, "past_key_values.")] = v.dims;
          }
        }
        return { logits: logitsTensor.data, vocabSize, presentKv };
      },
      runStep: async (tokenId, pastKv) => {
        const inputIdsTensor = new ort.Tensor("int64", new BigInt64Array([BigInt(tokenId)]), [1, 1]);
        const feeds = { input_ids: inputIdsTensor };
        for (const [name, data] of Object.entries(pastKv)) {
          const stepName = name.replace(/^present\./, "past_key_values.");
          const dims = state.kvDims[name] || state.kvDims[stepName] || state.kvDims[name.replace(/^past_key_values\./, "present.")];
          if (dims) feeds[stepName] = new ort.Tensor("float32", new Float32Array(data), dims);
        }
        const out = await decStepSession.run(feeds);
        const logitsKey = Object.keys(out).find((k) => k.includes("logits")) || Object.keys(out)[0];
        const logitsTensor = out[logitsKey];
        const vocabSize = logitsTensor.dims[logitsTensor.dims.length - 1] || 0;
        const presentKv = {};
        for (const [k, v] of Object.entries(out)) {
          if (k.startsWith("present")) {
            const pastName = k.replace(/^present/, "past_key_values");
            presentKv[pastName] = v.data;
            state.kvDims[pastName] = v.dims;
          }
        }
        for (const [k, v] of Object.entries(pastKv)) {
          if (k.includes("encoder") && !presentKv[k]) presentKv[k] = v;
        }
        return { logits: logitsTensor.data, vocabSize, presentKv };
      },
    };
  }

  // === bestOf=1 ===
  const cb1 = makeCallbacks();
  const r1 = await splitGraphDecodeLoop({
    promptTokens,
    encoderHiddenStates: encHs.data,
    eosTokenId: eosId,
    maxNewTokens: 100,
    modelConfig,
    bestOf: 1,
    processLogits: (logits, genTokens, beginIdx) => tsProc.process(logits, genTokens, beginIdx),
    runInit: cb1.runInit,
    runStep: cb1.runStep,
  });
  const text1 = tokenizer.decode([...promptTokens.slice(3), ...r1.tokens], { skipSpecialTokens: true });
  console.log("bestOf=1:", text1.trim());

  // === bestOf=3 ===
  const cb2 = makeCallbacks();
  const r2 = await splitGraphDecodeLoop({
    promptTokens,
    encoderHiddenStates: encHs.data,
    eosTokenId: eosId,
    maxNewTokens: 100,
    modelConfig,
    bestOf: 3,
    processLogits: (logits, genTokens, beginIdx) => tsProc.process(logits, genTokens, beginIdx),
    runInit: cb2.runInit,
    runStep: cb2.runStep,
  });
  const text2 = tokenizer.decode([...promptTokens.slice(3), ...r2.tokens], { skipSpecialTokens: true });
  console.log("bestOf=3:", text2.trim());

  console.log("\nPASS: bestOf smoke test");
  await encoderSession.release();
  await decInitSession.release();
  await decStepSession.release();
}

main().catch((err) => { console.error(err.stack); process.exit(1); });
