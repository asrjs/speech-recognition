import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import { requireFixtureOptIn } from './support/fixture-gates.js';
import {
  resourceGraphSha256,
  writeNativeResourceFixture,
} from './support/native-resource-fixture.js';

const script = resolve('tests/smoke/nemo-tdt-native-resource-smoke.mjs');
const hosts = [
  {
    name: 'node',
    executable: process.execPath,
    enabled: true,
    native: process.env.ASRJS_NATIVE_RESOURCE_NODE === '1',
  },
  {
    name: 'bun',
    executable: process.env.ASRJS_BUN_EXECUTABLE,
    enabled: process.env.ASRJS_NATIVE_RESOURCE_BUN === '1',
    native: process.env.ASRJS_NATIVE_RESOURCE_BUN === '1',
  },
];

describe.each(hosts)('$name native resource smoke prerequisites (no native inference)', (host) => {
  const run = (args: string[]) => {
    if (!host.executable?.trim())
      throw new Error('ASRJS_BUN_EXECUTABLE is required for requested Bun tests');
    const result = spawnSync(host.executable, [script, ...args], {
      encoding: 'utf8',
      timeout: 10_000,
    });
    if (result.error) throw result.error;
    if (result.signal) throw new Error(`Resource smoke terminated: ${result.signal}`);
    return result;
  };
  it('fails missing required arguments instead of silently skipping', (context) => {
    requireFixtureOptIn(context, host.enabled, 'Bun resource checks not requested');
    const result = run([]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain('--graph is required');
    expect(result.stdout).not.toContain('PASS_RESOURCE_ONLY');
  });

  it('refuses an existing receipt before loading fixtures or native resources', async (context) => {
    requireFixtureOptIn(context, host.enabled, 'Bun resource checks not requested');
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

  it('refuses a populated negative-control path rather than claiming partial failure', async (context) => {
    requireFixtureOptIn(context, host.enabled, 'Bun resource checks not requested');
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

  it('fails a requested missing fixture without writing a success receipt', async (context) => {
    requireFixtureOptIn(context, host.enabled, 'Bun resource checks not requested');
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

  it('releases actual native resources using portable non-ASR fixtures', async (context) => {
    requireFixtureOptIn(context, host.native, 'Actual native resource checks not requested');
    const directory = await mkdtemp(join(tmpdir(), 'asrjs-resource-smoke-'));
    try {
      const { graph, vocab } = await writeNativeResourceFixture(directory);
      const output = join(directory, 'receipt.json');
      const result = run(['--graph', graph, '--vocab', vocab, '--output', output]);
      expect(result.status, result.stderr).toBe(0);
      const receipt = JSON.parse(await readFile(output, 'utf8'));
      expect(receipt.status).toBe('PASS_RESOURCE_ONLY');
      expect(receipt.host.runtime).toBe(host.name);
      expect(receipt.engine).toBe('onnxruntime-node');
      expect(receipt.providers).toEqual(['cpu']);
      expect(receipt.fixtures.graph).toMatchObject({ bytes: 125, sha256: resourceGraphSha256 });
      expect(receipt.cases).toHaveLength(2);
      for (const [index, count] of [2, 1].entries()) {
        expect(receipt.cases[index]).toMatchObject({
          status: 'PASS_RESOURCE_ONLY',
          acquired: count,
          releaseAttempts: count,
          released: count,
        });
      }
    } finally {
      await rm(directory, { recursive: true });
    }
  }, 15_000);
});
