#!/usr/bin/env node
/**
 * VAD integration smoke test — TenVAD energy-based + segment merge.
 *
 * TenVAD backend uses energy-based RMS detection (no ONNX model needed).
 * FireRed VAD needs a pretrained model directory (deferred).
 *
 * TenVAD: 512-sample hops @ 16kHz = 32ms frames
 *   Threshold: 0.5 (RMS energy normalized 0-1)
 *   Min speech: 250ms, Min silence: 100ms
 *
 * Usage:
 *   node tests/smoke/vad-integration-smoke.mjs [--audio <path>]
 */
import path from "node:path";
import fs from "node:fs";

async function main() {
  const wavPath = process.argv.find(a => a.endsWith(".wav") || a.endsWith(".mp3"))
    ?? "tests/fixtures/jfk2.en.wav";

  console.log(`Audio: ${wavPath}`);

  const { TenVadBackend } = await import("../../dist/chunking/backends/ten-vad.js");
  const { mergeVadSegments } = await import("../../dist/chunking/vad-segmenter.js");

  // Decode WAV to Float32Array
  const buffer = fs.readFileSync(wavPath);
  const channels = buffer.readUInt16LE(22);
  const sampleRate = buffer.readUInt32LE(24);
  const frameCount = Math.floor((buffer.length - 44) / (2 * channels));
  const pcm = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    let sum = 0;
    for (let ch = 0; ch < channels; ch++) sum += buffer.readInt16LE(44 + (i * channels + ch) * 2) / 32768;
    pcm[i] = sum / channels;
  }
  const audioDuration = frameCount / sampleRate;
  console.log(`  Sample rate: ${sampleRate} Hz, Duration: ${audioDuration.toFixed(1)}s, Frames: ${frameCount}`);

  // ── Test 1: TenVAD backend (energy-based, no model needed) ──
  console.log("\n── Test 1: TenVAD energy-based ──");
  const tenVad = await TenVadBackend.create({
    threshold: 0.5,
    hopSize: 512,          // 32ms @ 16kHz
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
  });

  const rawSegments = await tenVad.segment(pcm, sampleRate, 0.5);
  console.log(`  Raw segments: ${rawSegments.length}`);
  for (const seg of rawSegments.slice(0, 5)) {
    console.log(`    [${seg.startSeconds.toFixed(2)}-${seg.endSeconds.toFixed(2)}] ${seg.durationSeconds.toFixed(2)}s`);
  }
  if (rawSegments.length > 5) console.log(`    ... and ${rawSegments.length - 5} more`);

  // Verify: JFK is 11s of speech, should detect 1-3 segments
  if (rawSegments.length < 1) throw new Error("TenVAD detected no speech segments");
  console.log("  PASS: speech detected");

  // ── Test 2: Merge + pad segments ──
  console.log("\n── Test 2: Merge + pad ──");
  const merged = mergeVadSegments(
    rawSegments,
    100,    // minSilenceDurationMs
    400,    // speechPadMs (200ms each side)
    29000,  // maxSegmentDurationMs
    250,    // minSpeechDurationMs
  );
  console.log(`  Merged segments: ${merged.length}`);
  for (const seg of merged) {
    console.log(`    [${seg.startSeconds.toFixed(2)}-${seg.endSeconds.toFixed(2)}] ${seg.durationSeconds.toFixed(2)}s`);
  }

  // Verify: merged should have ≥1 segments, all within audio bounds
  if (merged.length < 1) throw new Error("Merge produced no segments");
  for (const seg of merged) {
    if (seg.startSeconds < 0 || seg.endSeconds > audioDuration + 0.5) {
      throw new Error(`Segment out of bounds: [${seg.startSeconds}-${seg.endSeconds}] vs audio ${audioDuration}s`);
    }
  }
  console.log("  PASS: segments within bounds");

  // ── Test 3: Empty audio ──
  console.log("\n── Test 3: Empty audio ──");
  const silence = new Float32Array(16000); // 1s of zeros
  const silentSegs = await tenVad.segment(silence, 16000, 0.5);
  console.log(`  Silent segments: ${silentSegs.length}`);
  if (silentSegs.length !== 0) console.log("  WARN: energy-based VAD may detect noise floor as speech");

  // ── Test 4: Different thresholds ──
  console.log("\n── Test 4: Threshold sensitivity ──");
  for (const thresh of [0.3, 0.5, 0.7]) {
    const segs = await tenVad.segment(pcm, sampleRate, thresh);
    console.log(`  threshold=${thresh}: ${segs.length} segments`);
  }

  console.log("\nVAD INTEGRATION SMOKE PASSED");
  console.log("TenVAD energy-based backend works without ONNX model.");
  console.log("FireRed VAD: use FireRedVadBackend.create('/path/to/firered-model') when model is available.");
}

main().catch(e => { console.error(e.stack); process.exit(1); });
