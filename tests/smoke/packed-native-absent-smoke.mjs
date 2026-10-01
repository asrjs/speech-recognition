import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, realpathSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parsePackDryRun } from './package-tarball-smoke.mjs';

const repositoryRoot = fileURLToPath(new URL('../../', import.meta.url));
const consumer = mkdtempSync(join(tmpdir(), 'asrjs-packed-consumer-'));
// Retain the isolated fixture for inspection; never delete user caches/modules.
console.log(`[packed-native-absent] fixture: ${consumer}`);
const env = {
  ...process.env,
  NODE_TLS_REJECT_UNAUTHORIZED: '1',
  npm_config_cache: join(consumer, 'npm-cache'),
  npm_config_update_notifier: 'false',
};
const command = process.platform === 'win32' ? (process.env.ComSpec ?? 'cmd.exe') : 'npm';
const args = ['pack', '--json', '--ignore-scripts', '--offline', '--pack-destination', consumer];
const output = execFileSync(
  command,
  process.platform === 'win32' ? ['/d', '/s', '/c', 'npm', ...args] : args,
  { cwd: repositoryRoot, env, encoding: 'utf8', timeout: 60000 },
);
const packed = parsePackDryRun(output);
assert.equal(typeof packed.filename, 'string');
assert.equal(basename(packed.filename), packed.filename, 'archive filename must be local');
assert.ok(packed.filename.endsWith('.tgz'));
const packageRoot = join(consumer, 'node_modules/@asrjs/speech-recognition');
mkdirSync(packageRoot, { recursive: true });
execFileSync(
  'tar',
  ['-xzf', join(consumer, packed.filename), '--strip-components=1', '-C', packageRoot],
  {
    timeout: 60000,
  },
);

// Link only audited local browser runtime dependencies and consumer type helpers.
// This is archive consumption, not a package-manager installation/size benchmark.
for (const name of ['onnxruntime-web', 'pako', '@types/pako', '@types/node', 'undici-types']) {
  const destination = join(consumer, 'node_modules', name);
  mkdirSync(dirname(destination), { recursive: true });
  symlinkSync(realpathSync(join(repositoryRoot, 'node_modules', name)), destination, 'junction');
}
for (const name of ['native-absent.mjs', 'public-types.mts']) {
  copyFileSync(
    new URL(`./fixtures/package-consumer/${name}`, import.meta.url),
    join(consumer, name),
  );
}
execFileSync(process.execPath, ['native-absent.mjs'], {
  cwd: consumer,
  env,
  stdio: 'inherit',
  timeout: 30000,
});
execFileSync(
  process.execPath,
  [
    join(repositoryRoot, 'node_modules/typescript/bin/tsc'),
    'public-types.mts',
    '--noEmit',
    '--strict',
    '--module',
    'NodeNext',
    '--target',
    'ES2022',
    '--lib',
    'ES2022,DOM,DOM.Iterable',
  ],
  { cwd: consumer, env, stdio: 'inherit', timeout: 60000 },
);
console.log('[packed-native-absent] PASS: strict public declarations without native ORT');
