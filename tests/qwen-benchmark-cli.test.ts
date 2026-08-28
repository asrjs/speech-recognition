import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';

const script = 'tests/smoke/qwen3-asr-node-wasm-benchmark.mjs';

describe('Qwen benchmark CLI', () => {
  it('documents official graph and forced-window options', () => {
    const result = spawnSync(process.execPath, [script, '--help'], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('--encoder');
    expect(result.stdout).toContain('--dtype');
    expect(result.stdout).toContain('--window-seconds');
    expect(result.stdout).toContain('--reference');
    expect(result.stdout).toContain('official or legacy');
  });

  it('fails before loading when the local artifact directory is missing', () => {
    const result = spawnSync(process.execPath, [script], {
      cwd: process.cwd(),
      encoding: 'utf8',
      env: { ...process.env, QWEN3_ASR_MODEL_DIR: '' },
    });

    expect(result.status).toBe(1);
    expect(result.stderr).toContain('--model-dir is required');
  });
});
