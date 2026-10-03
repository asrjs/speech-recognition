/** Harness-only snapshots of the installed hybrid engine; never serve a directory. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const files = ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'];

export async function snapshotLocalOrtAssets(packageRoot) {
  const metadata = JSON.parse(await readFile(resolve(packageRoot, 'package.json'), 'utf8'));
  assert.equal(metadata.name, 'onnxruntime-web', 'Expected ORT Web package');
  assert.match(metadata.version, /^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/, 'Invalid ORT version');
  const prefix = `/engines/ort-${metadata.version}/`;
  const assets = new Map();
  const receipt = [];
  for (const fileName of files) {
    const path = resolve(packageRoot, 'dist', fileName);
    const info = await lstat(path);
    assert(info.isFile() && !info.isSymbolicLink(), `Not a regular engine file: ${fileName}`);
    assert(info.size > 0 && info.size <= 64 * 1024 * 1024, `Invalid engine size: ${fileName}`);
    const bytes = await readFile(path);
    assert.equal(bytes.length, info.size, `Engine changed while reading: ${fileName}`);
    const urlPath = `${prefix}${fileName}`;
    assets.set(urlPath, bytes);
    receipt.push({
      fileName: urlPath.slice(1),
      bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex'),
    });
  }
  return { assets, receipt, version: metadata.version, prefix };
}
