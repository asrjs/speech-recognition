import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { test } from 'node:test';
import { snapshotLocalOrtAssets } from './local-ort-assets.mjs';

const require = createRequire(import.meta.url);
const packageRoot = resolve(dirname(require.resolve('onnxruntime-web')), '..');

test('snapshots only the installed JSEP pair with versioned paths and exact hashes', async () => {
  const engine = await snapshotLocalOrtAssets(packageRoot);
  const metadata = JSON.parse(await readFile(resolve(packageRoot, 'package.json'), 'utf8'));
  assert.equal(engine.version, metadata.version);
  assert.equal(engine.prefix, `/engines/ort-${metadata.version}/`);
  assert.equal(engine.assets.size, 2);
  assert.deepEqual(engine.receipt.map((item) => item.fileName), [
    `engines/ort-${metadata.version}/ort-wasm-simd-threaded.jsep.mjs`,
    `engines/ort-${metadata.version}/ort-wasm-simd-threaded.jsep.wasm`,
  ]);
  for (const item of engine.receipt) {
    const original = await readFile(resolve(packageRoot, 'dist', item.fileName.split('/').at(-1)));
    assert.deepEqual(engine.assets.get(`/${item.fileName}`), original);
    assert.equal(item.bytes, original.length);
    assert.equal(item.sha256, createHash('sha256').update(original).digest('hex'));
  }
  assert(!engine.assets.has('/package.json'));
  assert(!engine.assets.has(`${engine.prefix}ort.node.min.js`));
});

test('a missing requested package fails rather than skipping', async () => {
  await assert.rejects(snapshotLocalOrtAssets(resolve(packageRoot, 'missing-package')), /ENOENT/);
});

test('a different package cannot masquerade as an ORT engine', async () => {
  const esbuildRoot = resolve(dirname(require.resolve('esbuild')), '..');
  await assert.rejects(snapshotLocalOrtAssets(esbuildRoot), /Expected ORT Web package/);
});
