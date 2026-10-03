/** Build an explicitly requested sibling in memory; never certify inference. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, readdir, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { builtinModules, createRequire } from 'node:module';
import { dirname, extname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build as bundle } from 'esbuild';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const requestedRoot = process.argv[2];
const serve = process.argv[3] === '--serve';
assert(
  (process.argv.length === 3 || (process.argv.length === 4 && serve)) && requestedRoot,
  'Usage: node companion-build.mjs <consumer-root> [--serve]',
);
const root = resolve(requestedRoot);
const configPath = resolve(root, 'vite.config.js');
const requireFromConsumer = createRequire(pathToFileURL(configPath));
const original = await readFile(configPath);
const metadata = await readFile(resolve(root, 'package.json'));
const tool = await readFile(fileURLToPath(import.meta.url));
const builtins = new Set(builtinModules);
const bundledConfig = await bundle({
  entryPoints: [configPath],
  bundle: true,
  platform: 'node',
  format: 'esm',
  write: false,
  define: { __dirname: JSON.stringify(root), __filename: JSON.stringify(configPath) },
  plugins: [
    {
      name: 'original-config-resolution',
      setup(builder) {
        builder.onResolve({ filter: /^[^./]/ }, (args) => {
          if (args.kind === 'entry-point') return;
          if (args.path.startsWith('node:') || builtins.has(args.path)) {
            return { path: args.path, external: true };
          }
          return {
            path: pathToFileURL(requireFromConsumer.resolve(args.path)).href,
            external: true,
          };
        });
        builder.onLoad({ filter: /vite\.config\.js$/ }, async ({ path }) => {
          assert.equal(resolve(path), configPath);
          // Supply exactly the captured bytes, rather than hashing only after loading.
          return { contents: original.toString('utf8'), loader: 'js' };
        });
      },
    },
  ],
});
const config = (
  await import(
    `data:text/javascript;base64,${Buffer.from(bundledConfig.outputFiles[0].contents).toString('base64')}`
  )
).default;
assert(config && typeof config === 'object', 'This probe requires an object Vite config');
const vitePath = requireFromConsumer.resolve('vite');
const { build, version } = await import(pathToFileURL(vitePath).href);
const previousCwd = process.cwd();
let result;
try {
  process.chdir(root);
  result = await build({
    ...config,
    root,
    configFile: false,
    cacheDir: resolve(repository, 'tools/data/results/harness/companion-build-cache'),
    build: { ...config.build, write: false, emptyOutDir: false },
  });
} finally {
  process.chdir(previousCwd);
}
assert.equal(hash(await readFile(configPath)), hash(original), 'Config changed during build');
assert.equal(hash(await readFile(resolve(root, 'package.json'))), hash(metadata));
assert.equal(hash(await readFile(fileURLToPath(import.meta.url))), hash(tool));
assert(result && Array.isArray(result.output) && result.output.length > 0, 'Missing build outputs');
const outputs = result.output.map(({ fileName, type, code, source }) => {
  const bytes = Buffer.from(code ?? source ?? '');
  assert(bytes.length > 0, `Empty build output: ${fileName}`);
  return { fileName, type, bytes: bytes.length, sha256: hash(bytes) };
});
console.log(
  JSON.stringify(
    {
      status: 'PASS_IN_MEMORY_COMPANION_BUILD',
      root,
      viteVersion: version,
      configSha256: hash(original),
      packageSha256: hash(metadata),
      toolSha256: hash(tool),
      configEvaluation:
        'In-memory esbuild ESM with original dirname and consumer package resolution',
      scope:
        'Build only; not normal npm build, exhaustive source provenance, browser inference or parity',
      outputs,
    },
    null,
    2,
  ),
);

if (serve) {
  const assets = new Map(
    result.output.map(({ fileName, code, source }) => [
      `/${fileName}`,
      Buffer.from(code ?? source ?? ''),
    ]),
  );
  const publicRoot =
    config.publicDir === false ? null : resolve(root, config.publicDir ?? 'public');
  const publicAssets = [];
  if (publicRoot) {
    const entries = await readdir(publicRoot, { withFileTypes: true }).catch((error) => {
      if (error.code === 'ENOENT') return [];
      throw error;
    });
    for (const entry of entries) {
      // Top-level files only: no symlinks or recursive model directory scan.
      if (!entry.isFile()) continue;
      const path = resolve(publicRoot, entry.name);
      assert((await stat(path)).size <= 10 * 1024 * 1024, 'Preview public asset exceeds 10 MiB');
      const bytes = await readFile(path);
      assert(!assets.has(`/${entry.name}`), 'Preview public/build asset collision');
      assets.set(`/${entry.name}`, bytes);
      publicAssets.push({ fileName: entry.name, bytes: bytes.length, sha256: hash(bytes) });
    }
  }
  const contentTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.wasm': 'application/wasm',
    '.ico': 'image/x-icon',
    '.json': 'application/json',
  };
  const server = createServer((request, response) => {
    let path;
    try {
      path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    } catch {
      response.writeHead(400).end();
      return;
    }
    const bytes = assets.get(path === '/' ? '/index.html' : path);
    if (request.method !== 'GET' || !bytes) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      'Content-Type':
        contentTypes[extname(path)] ?? (path === '/' ? 'text/html' : 'application/octet-stream'),
      'Content-Length': bytes.length,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    response.end(bytes);
  });
  await new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolveListen);
  });
  const address = server.address();
  assert(address && typeof address === 'object');
  console.log(
    JSON.stringify({
      status: 'PREVIEW_READY',
      url: `http://127.0.0.1:${address.port}`,
      publicAssets,
      scope: 'In-memory build and top-level public files only; no inference certification',
    }),
  );
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => server.close());
}
