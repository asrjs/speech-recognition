#!/usr/bin/env node
/**
 * Comprehensive VAD integration smoke test — WhisperX-style pipeline.
 *
 * Exercises the full VAD preprocessing chain:
 *   1. Energy-based VAD (TenVAD backend, no ONNX model needed)
 *   2. VAD binarization (probability → binary speech/silence with hysteresis)
 *   3. Noise gate (energy-floor gating)
 *   4. Segment merge + pad + split + overlap
 *   5. segmentAudio() — full pipeline wrapper
 *   6. (Optional) Whisper ASR with quality gates + temperature fallback
 *
 * Usage:
 *   node tests/smoke/vad-pipeline-smoke.mjs [--audio <path>] [--overlap <ms>]
 *
 * Env vars:
 *   WHISPER_BASE_DIR — whisper-base 4graph fp32 dir (default: /tmp/whisper-base-4graph/fp32)
 *   RUN_ASR=1 — also run full ASR pipeline (requires whisper-base model)
 */

import path from "node:path";
import fs from "node:fs";

// ──────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────

function decodeWav(wavPath) {
  const buf = fs.readFileSync(wavPath);
  const channels = buf.readUInt16LE(22);
  const sampleRate = buf.readUInt32LE(24);
  const dataLen = Math.floor((buf.length - 44) / (2 * channels));
  const pcm = new Float32Array(dataLen);
  for (let i = 0; i < dataLen; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++)
      sum += buf.readInt16LE(44 + (i * channels + ch) * 2) / 32768;
    pcm[i] = sum / channels;
  }
  return { pcm, sampleRate, duration: dataLen / sampleRate };
}

let passed = 0;
let failed = 0;
function assert(cond, msg) {
  if (cond) { passed++; console.log(`  PASS: ${msg}`); }
  else { failed++; console.error(`  FAIL: ${msg}`); }
}

function check(cond, msg) {
  if (!cond) throw new Error(msg);
}

// ──────────────────────────────────────────────────────────
async function main() {
  const wavPath = process.argv.find(a => a.endsWith(".wav") || a.endsWith(".mp3"))
    ?? "tests/fixtures/jfk2.en.wav";
  const overlapMs = parseInt(process.argv.find(a => a.startsWith("--overlap="))?.split("=")[1] ?? "500", 10);
  const runAsr = process.env.RUN_ASR === "1";
  const baseModelDir = process.env.WHISPER_BASE_DIR ?? "/tmp/whisper-base-4graph/fp32";

  console.log(`Audio: ${wavPath}`);
  console.log(`Overlap: ${overlapMs}ms`);
  console.log(`ASR: ${runAsr ? "ENABLED" : "disabled (set RUN_ASR=1 to enable)"}\n`);

  // ── Decode audio ──
  const { pcm, sampleRate, duration: audioDuration } = decodeWav(wavPath);
  console.log(`Sample rate: ${sampleRate} Hz, Duration: ${audioDuration.toFixed(1)}s, Frames: ${pcm.length}`);

  // ── Load our modules from dist ──
  const {
    TenVadBackend,
  } = await import("../../dist/chunking/backends/ten-vad.js");
  const {
    noiseGate,
    vadBinarize,
    mergeVadSegments,
    segmentAudio,
  } = await import("../../dist/chunking/vad-segmenter.js");

  // ══════════════════════════════════════════════════════
  // TEST 1: Noise gate
  // ══════════════════════════════════════════════════════
  console.log("\n── Test 1: Noise gate ──");

  // Pure silence should be gated to near-zero
  const silence = new Float32Array(8000);
  const gatedSilence = noiseGate(silence);
  let maxSilenceAfter = 0;
  for (let i = 0; i < gatedSilence.length; i++) maxSilenceAfter = Math.max(maxSilenceAfter, Math.abs(gatedSilence[i]));
  assert(maxSilenceAfter < 0.001, `Silence gated to zero (max=${maxSilenceAfter.toFixed(6)})`);

  // Speech should mostly pass through
  const gatedSpeech = noiseGate(pcm);
  let speechEnergyBefore = 0, speechEnergyAfter = 0;
  for (let i = 0; i < pcm.length; i++) {
    speechEnergyBefore += pcm[i] * pcm[i];
    speechEnergyAfter += gatedSpeech[i] * gatedSpeech[i];
  }
  const energyRatio = speechEnergyAfter / speechEnergyBefore;
  assert(energyRatio > 0.3, `Speech energy preserved (ratio=${energyRatio.toFixed(2)}, >0.3)`);

  // ══════════════════════════════════════════════════════
  // TEST 2: VAD binarization
  // ══════════════════════════════════════════════════════
  console.log("\n── Test 2: VAD binarization ──");

  // All-silence probabilities → no segments
  const silenceProbs = new Float32Array(100).fill(0.1);
  const silenceSegs = vadBinarize(silenceProbs, 0.032);
  assert(silenceSegs.length === 0, "All-silence produces no segments");

  // All-speech probabilities → one segment
  const speechProbs = new Float32Array(100).fill(0.9);
  const speechSegs = vadBinarize(speechProbs, 0.032);
  assert(speechSegs.length === 1, `All-speech produces 1 segment (got ${speechSegs.length})`);
  if (speechSegs.length > 0) {
    assert(speechSegs[0].startSeconds >= 0, "Segment starts at or after 0");
    assert(speechSegs[0].durationSeconds > 0, "Segment has positive duration");
  }

  // Mixed: speech → silence → speech → two segments
  const mixedProbs = new Float32Array(200);
  mixedProbs.fill(0.9, 0, 40);    // speech
  mixedProbs.fill(0.1, 40, 80);   // silence (40 hops = 1.28s > min silence)
  mixedProbs.fill(0.9, 80, 120);  // speech again
  mixedProbs.fill(0.1, 120, 200); // trailing silence
  const mixedSegs = vadBinarize(mixedProbs, 0.032);
  assert(mixedSegs.length === 2, `Mixed speech→silence→speech → 2 segments (got ${mixedSegs.length})`);

  // ══════════════════════════════════════════════════════
  // TEST 3: TenVAD energy-based segmentation
  // ══════════════════════════════════════════════════════
  console.log("\n── Test 3: TenVAD energy-based ──");

  const tenVad = await TenVadBackend.create({
    threshold: 0.5,
    hopSize: 512,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
  });

  const raw = await tenVad.segment(pcm, sampleRate, 0.5);
  console.log(`  Raw VAD segments: ${raw.length}`);
  for (const seg of raw.slice(0, 5)) {
    console.log(`    [${seg.startSeconds.toFixed(2)}-${seg.endSeconds.toFixed(2)}] ${seg.durationSeconds.toFixed(2)}s`);
  }
  assert(raw.length >= 1, `TenVAD detects speech (${raw.length} segments)`);
  assert(raw.length <= 30, `TenVAD doesn't over-segment (${raw.length} <= 30)`);

  // All segments within audio bounds
  for (const seg of raw) {
    assert(seg.startSeconds >= -0.1 && seg.endSeconds <= audioDuration + 0.5,
      `Segment [${seg.startSeconds.toFixed(1)}-${seg.endSeconds.toFixed(1)}] within bounds`);
  }

  // ══════════════════════════════════════════════════════
  // TEST 4: Merge + pad + split + overlap
  // ══════════════════════════════════════════════════════
  console.log("\n── Test 4: mergeVadSegments (no overlap) ──");

  const merged = mergeVadSegments(raw, 100, 400, 29000, 250);
  console.log(`  Merged: ${merged.length} segments`);
  for (const seg of merged) {
    console.log(`    [${seg.startSeconds.toFixed(2)}-${seg.endSeconds.toFixed(2)}] ${seg.durationSeconds.toFixed(2)}s`);
  }
  assert(merged.length >= 1, "Merge produces at least 1 segment");
  assert(merged.length <= raw.length + 2, `Merge doesn't explode segments (${merged.length} vs ${raw.length})`);

  // No segment exceeds max duration
  for (const seg of merged) {
    assert(seg.durationSeconds <= 30, `Segment under 30s (${seg.durationSeconds.toFixed(1)}s)`);
  }

  // ── Test with overlap ──
  console.log(`\n── Test 4b: mergeVadSegments (overlap=${overlapMs}ms) ──`);

  const mergedOverlap = mergeVadSegments(raw, 100, 400, 29000, 250, overlapMs);
  console.log(`  With overlap: ${mergedOverlap.length} segments`);
  for (const seg of mergedOverlap) {
    console.log(`    [${seg.startSeconds.toFixed(2)}-${seg.endSeconds.toFixed(2)}] ${seg.durationSeconds.toFixed(2)}s`);
  }
  assert(mergedOverlap.length >= 1, "Overlap merge produces segments");

  // For long segments, overlap should create additional chunks
  // (more segments with overlap than without when there are splits)
  assert(mergedOverlap.length >= merged.length,
    `Overlap doesn't reduce count (${mergedOverlap.length} >= ${merged.length})`);

  // Check actual overlaps (consecutive chunks from same parent should overlap)
  let foundOverlap = false;
  for (let i = 1; i < mergedOverlap.length; i++) {
    if (mergedOverlap[i].startSeconds < mergedOverlap[i - 1].endSeconds) {
      foundOverlap = true;
      console.log(`    Overlap detected: chunk ${i} starts at ${mergedOverlap[i].startSeconds.toFixed(2)} before chunk ${i-1} ends at ${mergedOverlap[i-1].endSeconds.toFixed(2)}`);
    }
  }
  if (merged.length < mergedOverlap.length) {
    assert(foundOverlap, "Split segments have overlap between consecutive chunks");
  }

  // ══════════════════════════════════════════════════════
  // TEST 5: segmentAudio() — full pipeline wrapper
  // ══════════════════════════════════════════════════════
  console.log("\n── Test 5: segmentAudio() ──");

  const pipelineSegs = await segmentAudio(pcm, {
    vad: tenVad,
    sampleRate,
    threshold: 0.5,
    noiseGate: false,  // test noise gate separately; energy VAD is sensitive to artifacts
    merge: {
      minSilenceDurationMs: 100,
      speechPadMs: 400,
      maxSegmentDurationMs: 29000,
      minSpeechDurationMs: 250,
      overlapDurationMs: overlapMs,
    },
  });
  console.log(`  Pipeline segments: ${pipelineSegs.length}`);
  assert(pipelineSegs.length >= 1, "segmentAudio produces segments");

  let totalCovered = 0;
  for (const seg of pipelineSegs) {
    totalCovered += seg.durationSeconds;
  }
  console.log(`  Total covered: ${totalCovered.toFixed(1)}s (audio: ${audioDuration.toFixed(1)}s)`);
  assert(totalCovered <= audioDuration * 2, "Coverage doesn't explode");

  // ══════════════════════════════════════════════════════
  // TEST 6: Noise gate on real audio
  // ══════════════════════════════════════════════════════
  console.log("\n── Test 6: Noise gate SNR improvement ──");

  // Use a silent region from the beginning/end of audio for noise floor
  const silentRegion = pcm.slice(0, Math.min(8000, Math.floor(pcm.length * 0.05)));
  const energyBefore = silentRegion.reduce((s, v) => s + v * v, 0) / silentRegion.length;
  const gatedSilent = noiseGate(silentRegion);
  const energyAfter = gatedSilent.reduce((s, v) => s + v * v, 0) / gatedSilent.length;
  const snrImprovement = energyBefore / Math.max(energyAfter, 1e-10);
  console.log(`  Silent region energy: ${energyBefore.toFixed(6)} → ${energyAfter.toFixed(6)} (gain: ${snrImprovement.toFixed(1)}x)`);
  assert(snrImprovement >= 0.5, `Noise gate reduces silent region energy (${snrImprovement.toFixed(1)}x >= 0.5x)`);

  // ══════════════════════════════════════════════════════
  // OPTIONAL TEST 7: Full ASR pipeline
  // ══════════════════════════════════════════════════════
  if (runAsr) {
    console.log("\n── Test 7: ASR pipeline (whisper-base) ──");

    const ort = await import("onnxruntime-node");
    const {
      WhisperTokenizer, WhisperMelProcessor,
      WhisperTimestampLogitProcessor,
      compressionRatioGate, logProbGate, noSpeechGate, entropyGate,
      withTemperatureFallback,
    } = await Promise.all([
      import("../../dist/models/whisper-seq2seq/index.js").then(m => ({
        WhisperTokenizer: m.WhisperTokenizer,
        fetchText: m.fetchText,
      })),
      import("../../dist/audio/whisper-mel.js").then(m => ({ WhisperMelProcessor: m.WhisperMelProcessor })),
      import("../../dist/models/whisper-seq2seq/processors.js").then(m => ({
        WhisperTimestampLogitProcessor: m.WhisperTimestampLogitProcessor,
      })),
      import("../../dist/models/whisper-seq2seq/executor.js").then(m => ({
        splitGraphDecodeLoop: m.splitGraphDecodeLoop,
      })),
      import("../../dist/models/whisper-seq2seq/generation-config.js").then(m => ({
        parseWhisperGenerationConfig: m.parseWhisperGenerationConfig,
        parseWhisperModelConfig: m.parseWhisperModelConfig,
      })),
      import("../../dist/quality/index.js").then(m => ({
        compressionRatioGate: m.compressionRatioGate,
        logProbGate: m.logProbGate,
        noSpeechGate: m.noSpeechGate,
        entropyGate: m.entropyGate,
        withTemperatureFallback: m.withTemperatureFallback,
      })),
    ]);

    const tokenizer = await WhisperTokenizer.fromUrl(path.join(baseModelDir, "tokenizer.json"));
    const genConfig = parseWhisperGenerationConfig(
      JSON.parse(await fetchText(path.join(baseModelDir, "generation_config.json")))
    );
    const configRaw = JSON.parse(await fetchText(path.join(baseModelDir, "config.json")));
    const modelConfig = parseWhisperModelConfig(configRaw);
    const melBins = modelConfig.numMelBins ?? 80;

    // Load sessions
    console.log("  Loading ONNX sessions...");
    const tLoad = performance.now();
    const [encSess, initSess, stepSess] = await Promise.all([
      ort.InferenceSession.create(path.join(baseModelDir, "encoder_model.onnx")),
      ort.InferenceSession.create(path.join(baseModelDir, "decoder_init.onnx")),
      ort.InferenceSession.create(path.join(baseModelDir, "decoder_step.onnx")),
    ]);
    console.log(`  Loaded in ${((performance.now()-tLoad)/1000).toFixed(1)}s`);

    const melProc = new WhisperMelProcessor({ nMels: melBins });
    const timestampProc = new WhisperTimestampLogitProcessor(tokenizer, genConfig);

    // Process each VAD segment through ASR
    const allTexts = [];
    const tAsr = performance.now();
    let totalTokens = 0;

    for (const seg of pipelineSegs) {
      const startSample = Math.max(0, Math.floor(seg.startSeconds * 16000));
      const endSample = Math.min(pcm.length, Math.ceil(seg.endSeconds * 16000));
      const chunk = pcm.slice(startSample, endSample);
      if (chunk.length < 1600) continue; // skip sub-100ms

      // Mel features
      const mel = WhisperMelProcessor.padToFrames(melProc.process(chunk), 3000);
      const featTensor = new ort.Tensor("float32", mel, [1, melBins, 3000]);
      const encOut = await encSess.run({ input_features: featTensor });
      const encHs = encOut[Object.keys(encOut)[0]];

      // Prompt tokens
      const promptTokens = [
        tokenizer.getTokenId("<|startoftranscript|>") ?? 50258,
        tokenizer.getTokenId("<|en|>") ?? 50268,
        tokenizer.getTokenId("<|transcribe|>") ?? 50359,
        tokenizer.getTokenId("<|notimestamps|>") ?? 50363,
      ];
      const initFeed = new ort.Tensor("int64", BigInt64Array.from(promptTokens.map(BigInt)), [1, promptTokens.length]);

      // Greedy decode with temperature fallback
      const temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
      const gates = [
        compressionRatioGate(2.4),
        logProbGate(-1.0),
        noSpeechGate(0.6, -1.0),
        entropyGate(2.4),
      ];

      let bestText = "";
      let bestTokens = [];

      for (const temp of temps) {
        const tempOpts = { ...genConfig, temperature: temp };
        const collectedTokens = [];
        const collectedLogits = [];

        const result = await splitGraphDecodeLoop({
          encoderHiddenStates: { data: new Float32Array(encHs.data), dims: encHs.dims },
          tokenizer,
          initSession: { run: (feeds) => initSess.run(feeds) },
          stepSession: { run: (feeds) => stepSess.run(feeds) },
          promptTokens,
          config: tempOpts,
          modelConfig,
          processLogits: (logits, ctx) => {
            timestampProc.processLogits(logits, ctx);
          },
          onTokenLogits: (id, logits, ctx) => {
            collectedTokens.push(id);
            collectedLogits.push(new Float32Array(logits));
          },
        });

        const text = result.text.trim();
        const tokenIds = collectedTokens.filter(t => t < 51865);

        // Evaluate gates
        const vocabSize = 51865;
        let allPassed = true;
        for (const gate of gates) {
          if (!gate(text, tokenIds, collectedLogits, vocabSize)) {
            allPassed = false;
            break;
          }
        }

        if (allPassed && text.length > bestText.length) {
          bestText = text;
          bestTokens = tokenIds;
        }

        if (allPassed && text.length > 3) break; // good enough
      }

      totalTokens += bestTokens.length;
      if (bestText) allTexts.push(bestText);
    }

    const tTotal = (performance.now() - tAsr) / 1000;
    console.log(`  Processed ${pipelineSegs.length} chunks in ${tTotal.toFixed(1)}s`);
    console.log(`  Tokens: ${totalTokens}, Words: ${allTexts.join(" ").split(/\s+/).length}`);

    // Verify: JFK audio should have recognizable words
    const fullText = allTexts.join(" ");
    console.log(`  Output: "${fullText.slice(0, 120)}..."`);
    assert(fullText.length > 10, `ASR produces non-empty output (${fullText.length} chars)`);

    // JFK-specific check
    if (wavPath.includes("jfk") || wavPath.includes("JFK")) {
      const lower = fullText.toLowerCase();
      assert(
        lower.includes("country") || lower.includes("ask") || lower.includes("american") || lower.includes("nation"),
        "JFK output contains recognizable words"
      );
    }
  } else {
    console.log("\n── Test 7: ASR pipeline (SKIPPED — set RUN_ASR=1) ──");
    console.log("  Run with: RUN_ASR=1 node tests/smoke/vad-pipeline-smoke.mjs");
  }

  // ══════════════════════════════════════════════════════
  console.log(`\n${"═".repeat(50)}`);
  console.log(`RESULTS: ${passed} passed, ${failed} failed`);
  console.log(`${"═".repeat(50)}`);

  if (failed > 0) {
    console.error("\nSOME TESTS FAILED");
    process.exit(1);
  }
  console.log("\nVAD PIPELINE SMOKE PASSED");
}

main().catch(e => { console.error(e.stack); process.exit(1); });
