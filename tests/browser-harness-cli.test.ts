import { describe, it, expect } from 'vitest';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

// Run the executable in Node: Vite's transformed-module wrapper cannot retain its shebang.
function parseCliArgs(args: string[]) {
  const result = spawnSync(process.execPath, ['--input-type=module', '-e',
    "import {parseCliArgs} from './tools/browser-harness/cli.mjs'; console.log(JSON.stringify(parseCliArgs(JSON.parse(process.argv[1]))));", JSON.stringify(args)], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr);
  return JSON.parse(result.stdout);
}
function writeReceipt(receipt: object, destination: string) {
  const result = spawnSync(process.execPath, ['--input-type=module', '-e',
    "import {writeReceipt} from './tools/browser-harness/cli.mjs'; writeReceipt(JSON.parse(process.argv[1]), process.argv[2]);", JSON.stringify(receipt), destination], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr);
}

describe('verification CLI boundary', () => {
  it.each([
    ['--mode', 'typo'], ['--mode'], ['--unknown'], ['--local-dir', '--expect-jfk'],
    ['--chunk-ms', '-1'], ['--chunk-ms', '0'], ['--chunk-ms', 'NaN'], ['--chunk-ms', '1junk'],
    ['--timeout-ms', '0'], ['--port', '99999'], ['--backend', 'typo'], ['--local-dirs', ','],
  ])('rejects malformed arguments %j before launching', (...args) => {
    expect(() => parseCliArgs(args)).toThrow();
  });
  it('preserves the matrix mode, oracle and warm policy', () => {
    const options = parseCliArgs(['--matrix', '--mode', 'streaming', '--expect-jfk', '--no-warm-rerun']);
    expect(options.mode).toBe('streaming');
    expect(options.oracle.sha256).toHaveLength(64);
    expect(options.warmRerun).toBe(false);
  });
  it('returns a failing process status for invalid input', () => {
    const result = spawnSync(process.execPath, ['tools/browser-harness/cli.mjs', '--chunk-ms', '-1'], { encoding: 'utf8' });
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('Invalid chunkMs');
    expect(result.stdout).not.toContain('Starting in-process');
  });
  it('retains failing receipts without overwriting existing evidence', () => {
    const directory = mkdtempSync(join(tmpdir(), 'asrjs-receipt-'));
    try {
      const target = join(directory, 'result.json');
      writeReceipt({ status: 'FAIL', error: 'oracle mismatch' }, target);
      expect(() => writeReceipt({ status: 'PASS' }, target)).toThrow();
      expect(JSON.parse(readFileSync(target, 'utf8')).status).toBe('FAIL');
    } finally { rmSync(directory, { recursive: true }); }
  });
});
