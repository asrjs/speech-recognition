import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';

const script = 'tests/smoke/transcribe-fixture.mjs';

describe('fixture transcription smoke CLI', () => {
  it('prints usage help', () => {
    const result = spawnSync(process.execPath, [script, '--help'], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('Usage:');
    expect(result.stdout).toContain('--audio');
    expect(result.stdout).toContain('--expect');
  });

  it('fails fast when required arguments are missing', () => {
    const result = spawnSync(process.execPath, [script], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });

    expect(result.status).toBe(2);
    expect(result.stderr).toContain('Missing required --audio');
  });

  it('skips gracefully when fixture smoke is not explicitly enabled', () => {
    const result = spawnSync(
      process.execPath,
      [script, '--audio', 'tests/fixtures/missing.wav', '--model', 'parakeet-tdt-0.6b-v2', '--expect', 'hello'],
      {
        cwd: process.cwd(),
        encoding: 'utf8',
        env: { ...process.env, ASRJS_FIXTURE_SMOKE: '' },
      },
    );

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('fixture transcription smoke skipped');
  });
});
