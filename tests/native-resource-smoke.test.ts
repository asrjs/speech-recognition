import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';

const script = resolve('tests/smoke/nemo-tdt-native-resource-smoke.mjs');
const run = (args: string[]) =>
  spawnSync(process.execPath, [script, ...args], {
    encoding: 'utf8',
    timeout: 10_000,
  });

describe('native resource smoke prerequisites (no native inference)', () => {
  it('fails missing required arguments instead of silently skipping', () => {
    const result = run([]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain('--graph is required');
    expect(result.stdout).not.toContain('PASS_RESOURCE_ONLY');
  });

  it('refuses an existing receipt before loading fixtures or native resources', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'asrjs-resource-smoke-'));
    try {
      const output = join(directory, 'receipt.json');
      await writeFile(output, 'preserve original evidence');
      const result = run(['--graph', 'missing.onnx', '--vocab', 'missing.txt', '--output', output]);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain('Refusing existing output/negative-control input');
      expect(await readFile(output, 'utf8')).toBe('preserve original evidence');
    } finally {
      await rm(directory, { recursive: true });
    }
  });

  it('refuses a populated negative-control path rather than claiming partial failure', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'asrjs-resource-smoke-'));
    try {
      const output = join(directory, 'receipt.json');
      const negativeInput = `${output}.intentionally-missing.onnx`;
      await writeFile(negativeInput, 'not missing');
      const result = run(['--graph', 'missing.onnx', '--vocab', 'missing.txt', '--output', output]);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain('Refusing existing output/negative-control input');
      expect(await readFile(negativeInput, 'utf8')).toBe('not missing');
      await expect(readFile(output)).rejects.toMatchObject({ code: 'ENOENT' });
    } finally {
      await rm(directory, { recursive: true });
    }
  });

  it('fails a requested missing fixture without writing a success receipt', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'asrjs-resource-smoke-'));
    try {
      const output = join(directory, 'receipt.json');
      const result = run([
        '--graph',
        join(directory, 'absent.onnx'),
        '--vocab',
        'missing.txt',
        '--output',
        output,
      ]);
      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain('ENOENT');
      expect(result.stdout).not.toContain('PASS_RESOURCE_ONLY');
      await expect(readFile(output)).rejects.toMatchObject({ code: 'ENOENT' });
    } finally {
      await rm(directory, { recursive: true });
    }
  });
});
