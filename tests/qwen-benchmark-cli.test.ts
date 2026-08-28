import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

const script = 'tests/smoke/qwen3-asr-node-wasm-benchmark.mjs';
const audioSha256 = 'a'.repeat(64);

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

  it('selects the matching sample from an official structured reference', async () => {
    const directory = await mkdtemp(path.join(tmpdir(), 'asrjs-qwen-reference-'));
    const referencePath = path.join(directory, 'reference.json');
    try {
      await writeFile(
        referencePath,
        JSON.stringify({
          schema_version: 1,
          reference_kind: 'qwen3-asr-0.6b-native-inference',
          samples: [
            { audio_sha256: 'b'.repeat(64), text: 'wrong sample' },
            { audio_sha256: audioSha256, sample_id: 'matching', text: 'native sample' },
          ],
        }),
        'utf8',
      );

      const { loadFixtureReference } =
        await import('../tests/smoke/qwen3-asr-node-wasm-benchmark.mjs');
      await expect(loadFixtureReference(referencePath, audioSha256)).resolves.toEqual({
        path: referencePath,
        field: 'samples[1].text',
        text: 'native sample',
        kind: 'qwen3-asr-0.6b-native-inference',
        sampleId: 'matching',
      });
      await expect(loadFixtureReference(referencePath, 'c'.repeat(64))).rejects.toThrow(
        'has no sample matching audio SHA-256',
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});
