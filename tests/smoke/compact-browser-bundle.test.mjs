import assert from 'node:assert/strict';
import { test } from 'node:test';
import { inspectCompactOutput, measureCompactConsumers } from './compact-browser-bundle.mjs';

function fixture(path = 'src/runtime/session.ts', bytesInOutput = 1) {
  return {
    outputFiles: [{ contents: new Uint8Array(10) }],
    metafile: {
      outputs: { 'entry.js': { bytes: 10, imports: [], inputs: { [path]: { bytesInOutput } } } },
    },
  };
}

test('retained inputs differ from visited but tree-shaken inputs', () => {
  const result = fixture();
  result.metafile.outputs['entry.js'].inputs['src/models/example/ort.ts'] = { bytesInOutput: 0 };
  assert.equal(inspectCompactOutput(result).retained.length, 1);
});

test('model, preset, engine and native paths fail closed', () => {
  for (const path of [
    'src/models/example/ort.ts',
    'src/presets/example/index.ts',
    'node_modules/onnxruntime-web/dist/ort.mjs',
    'src/io/node.ts',
    'src/io-node.ts',
    'src/inference/backends/native-ort.ts',
    'src\\models\\example\\ort.ts',
  ])
    assert.throws(() => inspectCompactOutput(fixture(path)), /Unexpected retained/);
});

test('external imports, empty evidence, invalid counts and exceeded budgets fail', () => {
  const external = fixture();
  external.metafile.outputs['entry.js'].imports.push({ path: 'node:fs', external: true });
  assert.throws(() => inspectCompactOutput(external), /Unexpected browser imports/);
  assert.throws(() => inspectCompactOutput(fixture(), 9), /exceeds budget/);
  assert.throws(() => inspectCompactOutput(fixture(), NaN), /Invalid bundle budget/);
  assert.throws(() => inspectCompactOutput(fixture('entry.ts', 0)), /Missing retained/);
  assert.throws(() => inspectCompactOutput(fixture('entry.ts', NaN)));
  const mismatch = fixture();
  mismatch.metafile.outputs['entry.js'].bytes = 11;
  assert.throws(() => inspectCompactOutput(mismatch), /byte count mismatch/);
  const absent = fixture();
  absent.outputFiles = [];
  assert.throws(() => inspectCompactOutput(absent), /Expected one browser output/);
});

test('actual public-root consumers remain engine-free within bounded budgets', async () => {
  const receipt = await measureCompactConsumers();
  assert.equal(receipt.status, 'PASS_SOURCE_CONSUMER_BUNDLE');
  assert.deepEqual(
    receipt.cases.map(({ name }) => name),
    ['createSpeechRuntime', 'transcribeWithWindowing'],
  );
  assert(Object.keys(receipt.loadedSourceHashes).length > 0);
});
