import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';

const script = 'tests/smoke/benchmark-hot-paths.mjs';

describe('hot-path benchmark CLI', () => {
  it('documents the reproducible baseline and correctness options', () => {
    const result = spawnSync(process.execPath, [script, '--help'], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('benchmark and correctness harness');
    expect(result.stdout).toContain('--runs=N');
    expect(result.stdout).toContain('--json');
    expect(result.stdout).toContain('base and');
  });
});
