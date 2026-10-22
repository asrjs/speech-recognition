import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { expect, it } from 'vitest';

it('keeps FP16 variant layout regressions in the standard CI test command', () => {
  const result = spawnSync(process.platform === 'win32' ? 'python' : 'python3', [
    '-m', 'unittest', 'discover', '-s', 'tools/whisper-onnx-export', '-p', 'test_variant_layout.py',
  ], { cwd: fileURLToPath(new URL('..', import.meta.url)), encoding: 'utf8', timeout: 10000 });
  const output = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  expect(result.error, output).toBeUndefined();
  expect(result.status, output).toBe(0);
  expect(output).toContain('Ran 3 tests');
}, 15000);
