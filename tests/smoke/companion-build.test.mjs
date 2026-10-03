import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const script = resolve(dirname(fileURLToPath(import.meta.url)), 'companion-build.mjs');
const repository = resolve(dirname(script), '../..');

function reject(args, expected) {
  const result = spawnSync(process.execPath, [script, ...args], {
    cwd: repository,
    encoding: 'utf8',
    timeout: 10000,
  });
  assert.equal(result.error, undefined);
  assert.equal(result.status, 1);
  assert.match(result.stderr, expected);
  assert(!result.stdout.includes('PASS_IN_MEMORY_COMPANION_BUILD'));
}

test('requires an explicit consumer root rather than selecting a demo silently', () => {
  reject([], /Usage:/);
});

test('rejects surplus arguments rather than ignoring them', () => {
  reject([repository, 'unexpected'], /Usage:/);
});

test('a requested root without its Vite config fails, never skips', () => {
  reject([resolve(repository, 'tests/fixtures')], /ENOENT.*vite\.config\.js/s);
});
