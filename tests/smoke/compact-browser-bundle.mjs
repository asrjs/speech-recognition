import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, extname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build, version } from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// This budget guards two engine-free source consumers, not arbitrary ASR apps.
export function inspectCompactOutput(result, maxBytes = 32768) {
  assert(Number.isSafeInteger(maxBytes) && maxBytes > 0, 'Invalid bundle budget');
  assert.equal(result.outputFiles.length, 1, 'Expected one browser output');
  const outputs = Object.values(result.metafile.outputs);
  assert.equal(outputs.length, 1, 'Expected one metafile output');
  const output = outputs[0];
  const bytes = result.outputFiles[0].contents.length;
  assert(bytes > 0 && bytes <= maxBytes, `Bundle exceeds budget: ${bytes}/${maxBytes}`);
  assert.equal(output.bytes, bytes, 'Output byte count mismatch');
  assert.equal(output.imports.length, 0, 'Unexpected browser imports');
  const retained = Object.entries(output.inputs).flatMap(([path, entry]) => {
    assert(Number.isSafeInteger(entry.bytesInOutput) && entry.bytesInOutput >= 0);
    if (entry.bytesInOutput === 0) return [];
    const normalized = path.replaceAll('\\', '/');
    assert(
      !/(?:^|\/)src\/(?:models|presets)\/|(?:^|\/)node_modules\/onnxruntime-|(?:^|\/)src\/io-node\.|(?:^|\/)src\/io\/node\.|(?:^|\/)src\/inference\/backends\/native/.test(
        normalized,
      ),
      `Unexpected retained model/engine/native module: ${path}`,
    );
    return [{ path: normalized, bytes: entry.bytesInOutput }];
  });
  assert(retained.length > 0, 'Missing retained input evidence');
  return { bytes, maxBytes, sha256: sha256(result.outputFiles[0].contents), retained };
}

export async function measureCompactConsumers() {
  const loaded = new Map();
  const metadata = await readFile(resolve(root, 'package.json'));
  const toolBytes = await readFile(fileURLToPath(import.meta.url));
  const cases = [];
  for (const name of ['createSpeechRuntime', 'transcribeWithWindowing']) {
    const result = await build({
      absWorkingDir: root,
      stdin: {
        contents: `import {${name}} from './src/index.ts'; globalThis.__compactProbe=${name};`,
        resolveDir: root,
        sourcefile: 'compact-entry.ts',
      },
      bundle: true,
      platform: 'browser',
      format: 'esm',
      minify: true,
      metafile: true,
      write: false,
      plugins: [
        {
          name: 'capture-source-bytes',
          setup(builder) {
            builder.onLoad({ filter: /\.(?:ts|js|mjs|cjs)$/ }, async ({ path }) => {
              const bytes = await readFile(path);
              const hash = sha256(bytes);
              assert(!loaded.has(path) || loaded.get(path) === hash, `Source changed: ${path}`);
              loaded.set(path, hash);
              return {
                contents: bytes.toString('utf8'),
                loader: extname(path) === '.ts' ? 'ts' : 'js',
              };
            });
          },
        },
      ],
    });
    cases.push({ name, ...inspectCompactOutput(result) });
  }
  for (const [path, hash] of loaded) {
    assert.equal(sha256(await readFile(path)), hash, `Source changed after build: ${path}`);
  }
  assert.equal(sha256(await readFile(resolve(root, 'package.json'))), sha256(metadata));
  assert.equal(sha256(await readFile(fileURLToPath(import.meta.url))), sha256(toolBytes));
  return {
    status: 'PASS_SOURCE_CONSUMER_BUNDLE',
    scope:
      'Minified browser-target source bundles only; no inference, startup, install or published-package acceptance',
    esbuildVersion: version,
    packageSha256: sha256(metadata),
    toolSha256: sha256(toolBytes),
    loadedSourceHashes: Object.fromEntries(loaded),
    cases,
  };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  console.log(JSON.stringify(await measureCompactConsumers(), null, 2));
}
