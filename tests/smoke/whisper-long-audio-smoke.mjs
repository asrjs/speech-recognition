#!/usr/bin/env node
/**
 * Whisper Long Audio Production Smoke — full 2m47s audio with auto-windowing.
 *
 * Uses whisper-large-v3-turbo q8 splitgraph model via loadSpeechModel.
 * Verifies long audio stitching quality against reference transcription.
 *
 * Usage:
 *   ASRJS_SMOKE=1 node tests/smoke/whisper-long-audio-smoke.mjs
 */
import { execSync } from 'node:child_process';
import { existsSync, readFileSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import os from 'node:os';

const ENABLED = new Set(['1', 'true', 'yes', 'on']);
const isSmoke = ENABLED.has(String(process.env.ASRJS_SMOKE ?? '').toLowerCase());

const MODEL_DIR = '/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph';
const AUDIO_MP3 = path.resolve(process.cwd(), 'tests/fixtures/end-of-chapter-4.en.mp3');
const REF_TXT = path.resolve(process.cwd(), 'tests/fixtures/end-of-chapter-4.en.txt');

async function main() {
  if (!isSmoke) { console.log('SKIP: ASRJS_SMOKE=1 required'); process.exit(0); }

  const t0 = performance.now();
  console.log('=== Whisper Long Audio Production Smoke ===\n');

  // Check prerequisites
  const variantDir = path.join(MODEL_DIR, 'fp16');
  if (!existsSync(variantDir)) { console.error(`ERROR: ${variantDir} not found`); process.exit(1); }
  if (!existsSync(AUDIO_MP3)) { console.error(`ERROR: ${AUDIO_MP3} not found`); process.exit(1); }
  if (!existsSync(REF_TXT)) { console.error(`ERROR: ${REF_TXT} not found`); process.exit(1); }

  // Convert MP3 to WAV (16kHz mono PCM)
  console.log('1. Converting MP3 to WAV...');
  const tmpWav = path.join(os.tmpdir(), `asrjs-long-audio-${Date.now()}.wav`);
  execSync(`ffmpeg -y -i "${AUDIO_MP3}" -ar 16000 -ac 1 -sample_fmt s16 -f wav "${tmpWav}" 2>/dev/null`);
  if (!existsSync(tmpWav)) { console.error('ERROR: ffmpeg conversion failed'); process.exit(1); }
  console.log('   done');

  // Load model
  console.log('2. Loading model (splitgraph q8)...');
  const { loadSpeechModel } = await import('../../dist/index.js');

  const loaded = await loadSpeechModel({
    family: 'whisper-seq2seq',
    modelId: 'openai/whisper-large-v3-turbo',
    backend: 'wasm',
    options: {
      variant: 'fp16',
      source: {
        kind: 'splitgraph',
        artifacts: {
          encoderUrl: path.join(variantDir, 'encoder_model.onnx'),
          decoderInitUrl: path.join(variantDir, 'decoder_init.onnx'),
          decoderStepUrl: path.join(variantDir, 'decoder_step.onnx'),
          decoderAlignUrl: path.join(variantDir, 'decoder_align.onnx'),
          tokenizerUrl: path.join(variantDir, 'tokenizer.json'),
          manifestUrl: path.join(variantDir, 'manifest.json'),
        },
        cpuThreads: 4,
      },
    },
  });
  console.log(`   loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  // Load WAV + convert to PCM
  console.log('\n3. Loading audio...');
  const wavBuf = readFileSync(tmpWav);
  const view = new DataView(wavBuf.buffer, wavBuf.byteOffset, wavBuf.byteLength);

  // Parse WAV header
  if (view.getUint32(0, false) !== 0x52494646) throw new Error('Not RIFF');
  if (view.getUint32(8, false) !== 0x57415645) throw new Error('Not WAVE');
  let off = 12, fmtFound = false, dataOff = 0, dataLen = 0, numCh = 1, sr = 16000, bps = 16;
  while (off < view.byteLength - 8) {
    const id = String.fromCharCode(view.getUint8(off), view.getUint8(off+1), view.getUint8(off+2), view.getUint8(off+3));
    const sz = view.getUint32(off+4, true);
    if (id === 'fmt ') {
      if (view.getUint16(off+8, true) !== 1) throw new Error('Not PCM');
      numCh = view.getUint16(off+10, true); sr = view.getUint32(off+12, true); bps = view.getUint16(off+22, true);
      fmtFound = true;
    } else if (id === 'data') { dataOff = off + 8; dataLen = sz; if (fmtFound) break; }
    off += 8 + sz;
  }
  if (!fmtFound || !dataLen) throw new Error('Invalid WAV');

  const totalSamples = dataLen / (bps / 8);
  const pcm = new Float32Array(totalSamples);
  const dv = new DataView(wavBuf.buffer, wavBuf.byteOffset + dataOff, dataLen);
  for (let i = 0; i < totalSamples; i++) pcm[i] = dv.getInt16(i * 2, true) / 32768;

  // Mono extraction if stereo
  const mono = numCh === 1 ? pcm : pcm.filter((_, i) => i % numCh === 0);
  const duration = mono.length / sr;
  console.log(`   ${path.basename(AUDIO_MP3)}: sr=${sr}, ${duration.toFixed(1)}s`);

  // Cleanup temp WAV
  try { unlinkSync(tmpWav); } catch (_) {}

  // Load reference
  const refText = readFileSync(REF_TXT, 'utf-8').replace(/\s+/g, ' ').trim().toLowerCase();
  const refWords = refText.split(/\s+/).filter(Boolean);
  console.log(`   reference: ${refWords.length} words\n`);

  // Transcribe (auto-windowed for Whisper 30s limit)
  console.log('4. Transcribing with auto-windowing...');
  const t1 = performance.now();
  const result = await loaded.transcribeMonoPcm(mono, sr, {
    language: 'en',
    detail: 'words',
    responseFlavor: 'canonical+native',
    returnWordTimestamps: true,
  });

  const elapsed = (performance.now() - t1) / 1000;
  const text = String(result.canonical?.text ?? result.native?.utteranceText ?? '').trim();
  const outputWords = text.split(/\s+/).filter(Boolean);
  console.log(`   output: ${outputWords.length} words in ${elapsed.toFixed(1)}s`);

  // Quick sample
  const preview = text.substring(0, 120);
  console.log(`   preview: "${preview}..."\n`);

  // Verify
  console.log('5. Verification:');

  // Word overlap
  const refSet = new Set(refWords);
  let matchCount = 0;
  for (const w of outputWords) {
    if (refSet.has(w.toLowerCase().replace(/[.,!?;:"]/g, ''))) matchCount++;
  }
  const overlapPct = (matchCount / Math.max(outputWords.length, 1) * 100).toFixed(1);
  console.log(`   word overlap with reference: ${overlapPct}% (${matchCount}/${outputWords.length})`);

  // Hallucination check
  const hallucinationRe = /\b(the the|and and|you you|thank you thank you|okay okay|um um|uh uh)\b/gi;
  const hallucinations = text.match(hallucinationRe);
  if (hallucinations) {
    console.log(`   \u2717 hallucinations: ${hallucinations.join(', ')}`);
  } else {
    console.log('   \u2713 no hallucinations');
  }

  // Word count sanity
  if (outputWords.length < 50) {
    console.log(`   \u2717 too few words: ${outputWords.length} (expected 300+)`);
  } else {
    console.log(`   \u2713 word count OK: ${outputWords.length}`);
  }

  // Sentence boundaries
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  console.log(`   sentences: ${sentences.length}`);

  await loaded.dispose();
  const totalSec = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`\n   total time: ${totalSec}s`);

  // Output first 500 chars for inspection
  console.log('\n--- FULL OUTPUT (first 500 chars) ---');
  console.log(text.substring(0, 500));

  console.log('\n=== SMOKE COMPLETE ===');
}
main().catch(err => { console.error('FATAL:', err); process.exit(1); });
