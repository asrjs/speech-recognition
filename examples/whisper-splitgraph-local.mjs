/**
 * EXAMPLE: Self-exported 4-graph Whisper transcription with word timestamps.
 *
 * Prerequisites:
 *   1. Export a model:
 *      cd tools/whisper-onnx-export
 *      .venv/bin/python export_whisper.py openai/whisper-tiny /tmp/whisper-tiny-4graph
 *
 *   2. Run the example:
 *      WHISPER_MODEL_DIR=/tmp/whisper-tiny-4graph \\
 *        node --experimental-vm-modules examples/whisper-splitgraph-local.mjs
 *
 *   3. For reproducibility testing:
 *      # Generate reference (Python)
 *      .venv/bin/python tools/whisper-onnx-export/generate_hf_reference.py \\
 *        --model-dir /tmp/whisper-tiny-4graph \\
 *        --audio tools/data/fixtures/audio/jfk-short.wav \\
 *        --output /tmp/whisper-ref.json \\
 *        --export-mel
 *
 *      # Compare (TypeScript)
 *      WHISPER_REFERENCE_JSON=/tmp/whisper-ref.json \\
 *        npm test -- tests/whisper-reproducibility-harness.test.ts
 *
 *   4. Smoke test:
 *      WHISPER_SPLITGRAPH_FIXTURE_DIR=/tmp/whisper-tiny-4graph \\
 *        npm test -- tests/whisper-splitgraph-smoke.test.ts
 */

import { createWhisperSeq2SeqModelFamily } from '../src/models/whisper-seq2seq/model.js';
import { loadSplitGraphLocalModel } from '../src/models/whisper-seq2seq/local-file.js';

async function main() {
  const modelDir = process.env.WHISPER_MODEL_DIR;
  if (!modelDir) {
    console.error('Set WHISPER_MODEL_DIR=/path/to/exported/whisper-tiny');
    process.exit(1);
  }

  // 1. Load self-exported 4-graph model from local directory
  const { source, config, modelId } = loadSplitGraphLocalModel(modelDir);
  console.log(`Loaded model: ${modelId} from ${modelDir}`);
  console.log(`  d_model=${config.vocabularySize} decoder_layers=${config.maxTargetPositions}`);

  // 2. Create model family and model
  const factory = createWhisperSeq2SeqModelFamily();
  const context = {
    backend: { id: 'wasm' },
    hooks: {},
  };
  const model = await factory.createModel(
    { modelId, options: { source, config } },
    context,
  );

  // 3. Create session and run transcription
  const session = await model.createSession();

  // Generate 2.5s 440Hz test audio
  const sampleRate = 16000;
  const duration = 2.5;
  const totalSamples = Math.floor(sampleRate * duration);
  const samples = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    samples[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate) * 0.3;
  }

  const audio = {
    sampleRate,
    durationSeconds: duration,
    channels: [samples],
    numberOfChannels: 1,
    numberOfFrames: totalSamples,
  };

  // Text-only (default)
  const textResult = await session.transcribe(audio, { language: 'en' });
  console.log(`\nText: "${textResult.utteranceText}"`);

  // Text + segments
  const segResult = await session.transcribe(audio, {
    language: 'en',
    detail: 'segments',
  });
  console.log(`\nSegments (${segResult.segments?.length ?? 0}):`);
  for (const seg of segResult.segments ?? []) {
    console.log(`  [${seg.startTime.toFixed(2)}-${seg.endTime.toFixed(2)}] ${seg.text}`);
  }

  // Text + word timestamps (requires decoder_align.onnx)
  const wordResult = await session.transcribe(audio, {
    language: 'en',
    detail: 'words',
    returnTimestamps: 'word',
  });
  console.log(`\nWords (${wordResult.words?.length ?? 0}):`);
  for (const word of wordResult.words ?? []) {
    const conf = word.confidence !== undefined ? ` conf=${word.confidence.toFixed(3)}` : '';
    console.log(`  [${word.startTime.toFixed(3)}-${word.endTime.toFixed(3)}] ${word.text}${conf}`);
  }

  await session.dispose();
  await model.dispose();
  console.log('\nDone.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
