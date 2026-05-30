#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const ENABLED = new Set(['1', 'true', 'yes', 'on']);
const isSmoke = ENABLED.has(String(process.env.ASRJS_SMOKE ?? '').toLowerCase());

function parseArgs(argv) {
  const args = { expects: [], language: 'en', variant: 'q8' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--model-dir') args.modelDir = argv[++i];
    else if (a === '--audio') args.audio = argv[++i];
    else if (a === '--expect') args.expects.push(argv[++i]);
    else if (a === '--language') args.language = argv[++i];
    else if (a === '--variant') args.variant = argv[++i];
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!isSmoke) { console.log('SKIP: ASRJS_SMOKE=1 required'); process.exit(0); }

  const modelPath = path.join(args.modelDir, args.variant);
  if (!existsSync(modelPath)) { console.error(`ERROR: not found: ${modelPath}`); process.exit(1); }
  if (!existsSync(args.audio)) { console.error(`ERROR: not found: ${args.audio}`); process.exit(1); }

  const t0 = performance.now();
  console.log('=== Whisper Production Smoke ===\n');

  const { loadSpeechModel } = await import('../../dist/index.js');

  console.log('Loading model...');
  const enc = path.join(modelPath, 'encoder_model.onnx');
  const init = path.join(modelPath, 'decoder_init.onnx');
  const step = path.join(modelPath, 'decoder_step.onnx');
  const align = path.join(modelPath, 'decoder_align.onnx');
  const tok = path.join(modelPath, 'tokenizer.json');

  const loaded = await loadSpeechModel({
    family: 'whisper-seq2seq',
    modelId: 'openai/whisper-large-v3-turbo',
    backend: 'wasm',
    options: {
      variant: args.variant,
      source: {
        kind: 'direct',
        artifacts: { encoderUrl: enc, decoderInitUrl: init, decoderStepUrl: step, decoderAlignUrl: align, tokenizerUrl: tok },
        cpuThreads: 4,
      },
    },
  });
  console.log(`   loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  const wavBuffer = readFileSync(args.audio);
  const WavIo = await import('../../dist/io/node.js');
  const audio = WavIo.decodeWav(wavBuffer.buffer);
  const pcm = new Float32Array(audio.samples);
  console.log(`   audio: ${path.basename(args.audio)}, sr=${audio.sampleRate}, ${(pcm.length / audio.sampleRate).toFixed(1)}s`);

  console.log('Transcribing (greedy)...');
  const t1 = performance.now();
  const result = await loaded.transcribeMonoPcm(pcm, audio.sampleRate, {
    language: args.language,
    detail: 'words',
    responseFlavor: 'canonical+native',
    returnWordTimestamps: true,
    maxNewTokens: 200,
  });

  const elapsed = (performance.now() - t1) / 1000;
  const text = String(result.canonical?.text ?? result.native?.utteranceText ?? '').trim();
  console.log(`   text: "${text}"`);
  console.log(`   time: ${elapsed.toFixed(1)}s\n`);

  let failed = false;
  const lower = text.toLowerCase();
  for (const s of args.expects) {
    if (lower.includes(s.toLowerCase())) { console.log(`   \u2713 "${s}"`); }
    else { console.log(`   \u2717 missing "${s}"`); failed = true; }
  }
  if (/\b(the the|and and|you you)\b/i.test(lower)) { console.log('   \u2717 hallucination'); failed = true; }
  else console.log('   \u2713 no hallucinations');
  const wc = text.split(/\s+/).filter(Boolean).length;
  console.log(`   words: ${wc}`);
  if (wc < 3) { console.log('   \u2717 too few words'); failed = true; }

  await loaded.dispose();
  console.log(`\ntotal: ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  if (failed) { console.log('\n=== SMOKE FAILED ===\n'); process.exit(1); }
  console.log('\n=== SMOKE PASSED ===\n');
}
main().catch(err => { console.error('FATAL:', err); process.exit(1); });
