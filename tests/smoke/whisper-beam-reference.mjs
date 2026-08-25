#!/usr/bin/env node
/**
 * Artifact-gated native Whisper beam oracle.
 *
 * The committed HF fixture is always checked by Vitest. When a local
 * splitgraph model is available, this smoke compares the library's complete
 * prompt+generated token sequence and text for beam 2 and beam 5.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runAsrPipeline } from './whisperx-runner.mjs';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const fixturePath = path.join(
  repoRoot,
  'tools/data/results/whisper/whisper-large-v3-turbo-jfk2-beams.json',
);
const fixtureAudioPath = path.join(repoRoot, 'tests/fixtures/jfk2.en.wav');
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

function optionValue(name) {
  const index = process.argv.indexOf(name);
  if (index < 0) return undefined;
  const value = process.argv[index + 1];
  if (!value || value.startsWith('--')) throw new Error(`${name} requires a value`);
  return value;
}

const model = optionValue('--model') ?? process.env.WHISPER_BEAM_REFERENCE_MODEL_DIR;
if (!model) {
  console.log(
    'SKIP: set WHISPER_BEAM_REFERENCE_MODEL_DIR or pass --model to run the native beam oracle.',
  );
  process.exit(0);
}
if (!fs.existsSync(model)) throw new Error(`Whisper beam reference model does not exist: ${model}`);

for (const beamSize of [2, 5]) {
  const expected = fixture.beams[String(beamSize)];
  const result = await runAsrPipeline({
    model,
    audioPath: fixtureAudioPath,
    language: fixture.decode.language,
    task: fixture.decode.task,
    beamSize,
    temperature: 0,
    wordTimestamps: false,
    noAlign: true,
    outputFormat: 'txt',
    verbose: true,
  });

  const actualTokens = result.tokenSequences?.[0] ?? [];
  if (JSON.stringify(actualTokens) !== JSON.stringify(expected.tokens)) {
    throw new Error(
      `beam ${beamSize} token mismatch\nexpected=${JSON.stringify(expected.tokens)}\nactual=${JSON.stringify(actualTokens)}`,
    );
  }
  const actualText = result.fullText.trim();
  const expectedText = expected.text.trim();
  if (actualText !== expectedText) {
    throw new Error(
      `beam ${beamSize} text mismatch\nexpected=${expectedText}\nactual=${actualText}`,
    );
  }
  console.log(
    `PASS: beam ${beamSize} exact token/text parity (${result.asrTime.toFixed(3)}s ASR time)`,
  );
}
