/** Build an explicitly requested sibling in memory; never certify inference. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { builtinModules, createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build as bundle } from 'esbuild';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const requestedRoot = process.argv[2];
assert(
  process.argv.length === 3 && requestedRoot,
  'Usage: node companion-build.mjs <consumer-root>',
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
