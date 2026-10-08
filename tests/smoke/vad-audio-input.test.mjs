import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = readFileSync(new URL('../fixtures/jfk2.en.wav', import.meta.url));
for (const script of ['vad-integration-smoke.mjs', 'vad-pipeline-smoke.mjs']) {
  function run(audio) {
    return spawnSync(process.execPath, [fileURLToPath(new URL(script, import.meta.url)), audio], {
      encoding: 'utf8',
      timeout: 30_000,
      env: { ...process.env, RUN_ASR: '0' },
    });
  }

  test(`${script}: metadata WAV contains exactly the original PCM`, () => {
    const audio = new URL('../fixtures/jfk2.en.wav', import.meta.url);
    const result = run(fileURLToPath(audio));
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout, /Frames: 176000\b/);
    assert.match(
      result.stdout,
      /PCM SHA-256: ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0/,
    );
  });

  test(`${script}: non-RIFF bytes cannot pass as WAV audio`, () => {
    const directory = mkdtempSync(join(tmpdir(), 'asrjs-vad-input-'));
    try {
      const invalid = Buffer.from(source);
      invalid.write('FAKE', 0, 'ascii');
      const audio = join(directory, 'invalid.wav');
      writeFileSync(audio, invalid);
      const result = run(audio);
      assert.notEqual(result.status, 0, result.stdout);
      assert.match(result.stderr, /Invalid WAV/);
      assert.doesNotMatch(result.stdout, /VAD .*PASSED/);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
}
