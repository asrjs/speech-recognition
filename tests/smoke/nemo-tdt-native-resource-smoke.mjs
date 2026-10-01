// Real native sessions, resource ownership only; never ASR/tensor-contract acceptance.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, access, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { OrtNemoTdtExecutor } from '../../dist/models/nemo-tdt/executor.js';
import {
  DEFAULT_NEMO_TDT_CLASSIFICATION,
  parseNemoTdtConfig,
} from '../../dist/models/nemo-tdt/config.js';
import { initOrt } from '../../dist/models/nemo-tdt/ort.js';

const { values } = parseArgs({
  options: {
    graph: { type: 'string' },
    vocab: { type: 'string' },
    output: { type: 'string' },
  },
});
for (const name of ['graph', 'vocab', 'output']) {
  assert(values[name]?.trim(), `--${name} is required`);
}
const graph = resolve(values.graph),
  vocab = resolve(values.vocab),
  output = resolve(values.output);
const missingGraph = `${output}.intentionally-missing.onnx`;
async function requireAbsent(path) {
  try {
    await access(path);
  } catch (error) {
    if (error.code === 'ENOENT') return;
    throw error;
  }
  throw new Error(`Refusing existing output/negative-control input: ${path}`);
}
await requireAbsent(output);
await requireAbsent(missingGraph);
async function identity(path) {
  const bytes = await readFile(path);
  assert(bytes.length, `Empty resource fixture: ${path}`);
  return { path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') };
}
const fixtures = { graph: await identity(graph), vocab: await identity(vocab) };
const ort = await initOrt('native-cpu', { cpuThreads: 1 });
const originalCreate = ort.InferenceSession.create;
let activeCase;
// Process-local instrumentation delegates every acquisition/release to the real binding.
ort.InferenceSession.create = async function (...args) {
  const owner = activeCase;
  owner.attemptedGraphs.push(String(args[0]));
  assert.deepEqual(args[1].executionProviders, ['cpu']);
  assert.equal(args[1].intraOpNumThreads, 1);
  assert.equal(args[1].interOpNumThreads, 1);
  const nativeSession = await originalCreate.apply(this, args);
  owner.acquired += 1;
  const release = nativeSession.release.bind(nativeSession);
  return new Proxy(nativeSession, {
    get(target, property, receiver) {
      if (property !== 'release') return Reflect.get(target, property, receiver);
      return async () => {
        owner.releaseAttempts += 1;
        await release();
        owner.released += 1;
      };
    },
  });
};
const cases = [];
try {
  for (const partialFailure of [false, true]) {
    const record = {
      name: partialFailure ? 'partial-acquisition' : 'successful-acquisition',
      attemptedGraphs: [],
      acquired: 0,
      releaseAttempts: 0,
      released: 0,
    };
    activeCase = record;
    const executor = new OrtNemoTdtExecutor(
      'resource-fixture/not-asr',
      DEFAULT_NEMO_TDT_CLASSIFICATION,
      parseNemoTdtConfig('resource-fixture/not-asr'),
      'native-cpu',
      {
        source: {
          kind: 'direct',
          cpuThreads: 1,
          preprocessorBackend: 'js',
          artifacts: {
            encoderUrl: pathToFileURL(graph).href,
            decoderUrl: pathToFileURL(partialFailure ? missingGraph : graph).href,
            tokenizerUrl: pathToFileURL(vocab).href,
          },
        },
      },
    );
    let readinessError;
    try {
      await executor.ready();
    } catch (error) {
      readinessError = error;
    }
    const releasedAtReady = record.released;
    // Always release actual resources before checking results or emitting success.
    await executor.dispose();
    await executor.dispose();
    assert.equal(record.attemptedGraphs.length, 2);
    assert.equal(record.acquired, partialFailure ? 1 : 2);
    assert.equal(record.releaseAttempts, record.acquired);
    assert.equal(record.released, record.acquired);
    if (partialFailure) {
      assert(readinessError instanceof Error, 'Missing decoder must reject initialization');
      assert.equal(releasedAtReady, 1, 'Partial encoder release must finish before ready rejects');
      record.expectedInitializationError = readinessError.message;
    } else {
      assert.equal(readinessError, undefined);
      assert.equal(releasedAtReady, 0);
    }
    record.status = 'PASS_RESOURCE_ONLY';
    cases.push(record);
  }
} finally {
  ort.InferenceSession.create = originalCreate;
}
const receipt = {
  schemaVersion: 1,
  status: 'PASS_RESOURCE_ONLY',
  scope:
    'Actual native CPU acquisition/release through NeMo TDT executor; no inference, model graph contract or numerical parity',
  host: {
    runtime: typeof Bun === 'undefined' ? 'node' : 'bun',
    version: typeof Bun === 'undefined' ? process.version : Bun.version,
  },
  engine: 'onnxruntime-node',
  providers: ['cpu'],
  versions: ort.env.versions,
  fixtures,
  cases,
  commands: process.argv,
  code: await Promise.all(
    [
      'tests/smoke/nemo-tdt-native-resource-smoke.mjs',
      'src/models/nemo-tdt/executor.ts',
      'dist/models/nemo-tdt/executor.js',
      'src/models/nemo-tdt/ort.ts',
      'dist/models/nemo-tdt/ort.js',
    ].map(identity),
  ),
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
console.log(
  JSON.stringify({ output, status: receipt.status, host: receipt.host, cases: cases.length }),
);
