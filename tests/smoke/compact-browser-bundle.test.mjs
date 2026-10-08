import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
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
    'node_modules/@asrjs/speech-recognition/dist/models/example/ort.js',
    'node_modules/@asrjs/speech-recognition/dist/presets/example/index.js',
    'node_modules/@asrjs/speech-recognition/dist/io/node.js',
    'node_modules/@asrjs/speech-recognition/dist/io-node.js',
    'node_modules/@asrjs/speech-recognition/dist/inference/backends/native-ort.js',
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

async function consumerFixture(
  entry = 'export const createSpeechRuntime=()=>1; export const transcribeWithWindowing=()=>2;',
) {
  const consumerRoot = await mkdtemp(join(tmpdir(), 'asrjs-compact-contract-'));
  const packageRoot = join(consumerRoot, 'node_modules/@asrjs/speech-recognition');
  await mkdir(join(packageRoot, 'dist'), { recursive: true });
  await writeFile(
    join(packageRoot, 'package.json'),
    JSON.stringify({
      name: '@asrjs/speech-recognition',
      type: 'module',
      exports: { '.': './dist/public.js' },
    }),
  );
  await writeFile(join(packageRoot, 'dist/public.js'), entry);
  return { consumerRoot, packageRoot };
}

test('consumer mode resolves the public package export instead of source checkout', async () => {
  const { consumerRoot, packageRoot } = await consumerFixture();
  const receipt = await measureCompactConsumers({ consumerRoot });
  assert.equal(receipt.status, 'PASS_PACKAGE_CONSUMER_BUNDLE');
  assert.equal(receipt.packageRoot, packageRoot);
  assert(receipt.cases.every(({ bytes }) => bytes < 200));
  assert(Object.keys(receipt.loadedSourceHashes).every((path) => path.startsWith(packageRoot)));
});

test('consumer mode rejects missing package and wrong package identity', async () => {
  const { consumerRoot, packageRoot } = await consumerFixture();
  await writeFile(
    join(packageRoot, 'package.json'),
    JSON.stringify({ name: 'not-asrjs', type: 'module' }),
  );
  await assert.rejects(measureCompactConsumers({ consumerRoot }), /package identity/i);
  await assert.rejects(
    measureCompactConsumers({ consumerRoot: join(consumerRoot, 'missing') }),
    /ENOENT/,
  );
});

test('consumer mode rejects model code retained by an exported consumer', async () => {
  const { consumerRoot, packageRoot } = await consumerFixture(
    "export {createSpeechRuntime,transcribeWithWindowing} from './models/example/ort.js';",
  );
  await mkdir(join(packageRoot, 'dist/models/example'), { recursive: true });
  await writeFile(
    join(packageRoot, 'dist/models/example/ort.js'),
    'export const createSpeechRuntime=()=>1; export const transcribeWithWindowing=()=>2;',
  );
  await assert.rejects(measureCompactConsumers({ consumerRoot }), /Unexpected retained/);
});
