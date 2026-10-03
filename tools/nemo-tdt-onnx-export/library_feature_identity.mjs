/**
 * Test whether the two real-browser runs fed the encoder the same features.
 *
 * Both runs used production JS features, but through different code: the hybrid
 * runs the library pipeline, while the encoder-sensitivity replay constructs
 * JSMelProcessor inline with hardcoded options. Equal PCM and equal static
 * option values do not prove equal feature tensors.
 *
 * This computes the library's own production features from the same local PCM
 * both runs hashed (f32le SHA-256 ebd52851...) and compares them, element by
 * element, against the tensor the browser replay actually recorded.
 *
 * Read-only: reads existing fixtures and receipts, writes nothing.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { JSMelProcessor } from '../../dist/audio/js-mel.js';
import { decodeWavToMonoPcm } from '../../dist/io-node.js';

const EXPECTED_PCM_SHA256 =
  'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0';
const REPLAY = 'tools/data/results/orukeet/browser-production-feature-encoder-jfk-2026-10-03.json';
const AUDIO = 'tools/data/fixtures/audio/jfk-short.wav';

const replay = JSON.parse(readFileSync(REPLAY, 'utf8'));
const decoded = decodeWavToMonoPcm(readFileSync(AUDIO));
const pcmBytes = Buffer.from(
  decoded.pcm.buffer,
  decoded.pcm.byteOffset,
  decoded.pcm.byteLength,
);
const pcmSha256 = createHash('sha256').update(pcmBytes).digest('hex');
console.log('== input identity ==');
console.log(`  pcm samples=${decoded.pcm.length} rate=${decoded.sampleRate}`);
console.log(`  pcm f32le sha256=${pcmSha256}`);
console.log(`  matches both browser runs: ${pcmSha256 === EXPECTED_PCM_SHA256}`);
if (pcmSha256 !== EXPECTED_PCM_SHA256) {
  throw new Error('Local fixture is not the PCM both browser runs used.');
}

const recorded = replay.result.productionFeatures;
const recordedLength = Number(replay.result.productionLength);
const melBins = recorded.shape[1];

console.log('== library production features ==');
const processor = new JSMelProcessor({
  nMels: melBins,
  normalization: 'per_feature',
  validLengthMode: 'onnx',
});
const produced = processor.process(decoded.pcm);
console.log(`  shape=${produced.features.length / melBins}x${melBins} frameCount=${produced.frameCount} length=${produced.length}`);

console.log('== comparison against the browser-recorded tensor ==');
console.log(`  recorded shape=${recorded.shape.join('x')} length=${recordedLength}`);

const expected = recorded.data;
const actual = produced.features;
if (expected.length !== actual.length) {
  console.log(`  LENGTH DIFFERS: recorded=${expected.length} library=${actual.length}`);
  console.log('  RESULT: FEATURES DIFFER — the two runs did not feed the same tensor.');
  process.exit(0);
}

let maxAbs = 0;
let firstDivergent = null;
for (let index = 0; index < expected.length; index += 1) {
  const delta = Math.abs(expected[index] - actual[index]);
  if (delta > maxAbs) maxAbs = delta;
  if (firstDivergent === null && delta !== 0) firstDivergent = index;
}
const byteIdentical =
  Buffer.from(new Float32Array(expected).buffer).equals(
    Buffer.from(new Float32Array(actual).buffer),
  );
console.log(`  count=${expected.length} maxAbsError=${maxAbs} firstDivergent=${firstDivergent}`);
console.log(`  byte-identical: ${byteIdentical}`);
console.log(`  lengths agree: ${produced.length === recordedLength}`);
console.log(
  `  RESULT: ${byteIdentical ? 'FEATURES IDENTICAL' : 'FEATURES DIFFER'}`,
);