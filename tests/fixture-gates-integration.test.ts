import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

// Wiring regressions only: no real graphs, inference, downloads or model oracle claims.
function runFixtureTests(files: string[], overrides: Record<string, string> = {}) {
  const env = { ...process.env };
  for (const setting of [
    'ASRJS_FIXTURE_SMOKE',
    'ASRJS_FIXTURE_SMOKE_FORCE',
    'WHISPER_MERGED_FIXTURE_DIR',
    'WHISPER_TIMESTAMPED_FIXTURE_DIR',
    'WHISPER_SPLITGRAPH_FIXTURE_DIR',
    'WHISPER_MEL_REFERENCE_JSON',
    'WHISPER_REFERENCE_JSON',
    'WHISPER_REFERENCE_MEL',
    'WHISPER_REFERENCE_AUDIO',
    'WHISPER_REFERENCE_MODEL_DIR',
    'WHISPER_REFERENCE_ENCODER_DIR',
    'WHISPER_REFERENCE_DECODER_DIR',
    'GIGAAM_CTC_ONNX_SMOKE',
    'GIGAAM_CTC_REFERENCE_JSON',
    'GIGAAM_CTC_ONNX_DIR',
    'GIGAAM_CTC_FP16_ONNX_DIR',
  ])
    delete env[setting];
  const result = spawnSync(
    process.execPath,
    ['node_modules/vitest/vitest.mjs', 'run', ...files, '--reporter=json'],
    { env: { ...env, ...overrides }, encoding: 'utf8', timeout: 30_000, maxBuffer: 2_000_000 },
  );
  if (result.error) throw result.error;
  const report = JSON.parse(result.stdout) as {
    success: boolean;
    numPassedTests: number;
    numPendingTests: number;
  };
  return { ...result, report };
}

describe('cross-model fixture-gate wiring in the actual test runner', () => {
  it('reports disabled Whisper reproducibility inference as SKIP while running the dtype unit test', () => {
    const result = runFixtureTests(['tests/whisper-reproducibility-harness.test.ts']);
    expect(result.status).toBe(0);
    expect(result.report.success).toBe(true);
    expect(result.report.numPassedTests).toBe(3);
    expect(result.report.numPendingTests).toBe(2);
  }, 45_000);

  it('fails explicitly requested Whisper reproducibility with a missing reference', () => {
    const directory = mkdtempSync(join(tmpdir(), 'asr-repro-missing-reference-'));
    try {
      const result = runFixtureTests(['tests/whisper-reproducibility-harness.test.ts'], {
        WHISPER_REFERENCE_JSON: join(directory, 'absent-reference.json'),
      });
      expect(result.status).toBe(1);
      expect(result.report.success).toBe(false);
      expect(result.stdout).toContain('absent-reference.json');
    } finally {
      rmSync(directory, { recursive: true });
    }
  }, 45_000);

  it.each([
    ['empty', '', 'Required fixture must be a nonempty file: WHISPER_REFERENCE_JSON'],
    ['malformed', '{ invalid JSON', 'SyntaxError'],
  ])(
    'fails an explicitly requested %s Whisper reproducibility reference',
    (_name, contents, error) => {
      const directory = mkdtempSync(join(tmpdir(), 'asr-repro-invalid-reference-'));
      try {
        const reference = join(directory, 'reference.json');
        writeFileSync(reference, contents);
        const result = runFixtureTests(['tests/whisper-reproducibility-harness.test.ts'], {
          WHISPER_REFERENCE_JSON: reference,
        });
        expect(result.status).toBe(1);
        expect(result.report.success).toBe(false);
        expect(result.stdout).toContain(error);
      } finally {
        rmSync(directory, { recursive: true });
      }
    },
    45_000,
  );

  it('fails enabled Whisper reproducibility with absent mel and audio before native sessions', () => {
    const directory = mkdtempSync(join(tmpdir(), 'asr-repro-missing-inputs-'));
    try {
      const reference = join(directory, 'reference.json');
      writeFileSync(reference, JSON.stringify({ audio: { path: join(directory, 'absent.wav') } }));
      const result = runFixtureTests(['tests/whisper-reproducibility-harness.test.ts'], {
        WHISPER_REFERENCE_JSON: reference,
      });
      expect(result.status).toBe(1);
      expect(result.report.success).toBe(false);
      expect(result.stdout).toContain('Required fixture setting is missing: WHISPER_REFERENCE_MEL');
      expect(result.stdout).toContain('absent.wav');
    } finally {
      rmSync(directory, { recursive: true });
    }
  }, 45_000);

  it('reports disabled Whisper and GigaAM inference tests as SKIP, including flag=0', () => {
    const result = runFixtureTests(
      [
        'tests/whisper-onnx-smoke.test.ts',
        'tests/whisper-timestamped-decoder.test.ts',
        'tests/whisper-splitgraph-smoke.test.ts',
        'tests/gigaam-ctc-onnx-backends.test.ts',
      ],
      { ASRJS_FIXTURE_SMOKE: '0' },
    );
    expect(result.status).toBe(0);
    expect(result.report.success).toBe(true);
    expect(result.report.numPassedTests).toBe(0);
    expect(result.report.numPendingTests).toBe(9);
  }, 45_000);

  it.each([
    ['tests/whisper-onnx-smoke.test.ts', 'WHISPER_MERGED_FIXTURE_DIR'],
    ['tests/whisper-timestamped-decoder.test.ts', 'WHISPER_TIMESTAMPED_FIXTURE_DIR'],
  ])(
    'fails a forced %s run with absent configuration',
    (file, setting) => {
      const result = runFixtureTests([file], { ASRJS_FIXTURE_SMOKE_FORCE: '1' });
      expect(result.status).toBe(1);
      expect(result.report.success).toBe(false);
      expect(result.stdout).toContain(`Required fixture setting is missing: ${setting}`);
    },
    45_000,
  );

  it('fails an explicitly requested split-graph run with an empty fixture directory', () => {
    const directory = mkdtempSync(join(tmpdir(), 'asr-empty-splitgraph-'));
    try {
      const result = runFixtureTests(['tests/whisper-splitgraph-smoke.test.ts'], {
        WHISPER_SPLITGRAPH_FIXTURE_DIR: directory,
      });
      expect(result.status).toBe(1);
      expect(result.report.success).toBe(false);
      expect(result.stdout).toContain('encoder_model.onnx');
    } finally {
      rmSync(directory, { recursive: true });
    }
  }, 45_000);

  it('fails an enabled GigaAM suite with a missing reference instead of skipping it', () => {
    const directory = mkdtempSync(join(tmpdir(), 'asr-missing-reference-'));
    try {
      const result = runFixtureTests(['tests/gigaam-ctc-onnx-backends.test.ts'], {
        GIGAAM_CTC_ONNX_SMOKE: '1',
        GIGAAM_CTC_REFERENCE_JSON: join(directory, 'absent-reference.json'),
      });
      expect(result.status).toBe(1);
      expect(result.report.success).toBe(false);
      expect(result.stdout).toContain('absent-reference.json');
    } finally {
      rmSync(directory, { recursive: true });
    }
  }, 45_000);

  it('fails a malformed Whisper mel reference instead of swallowing it as absent', () => {
    const directory = mkdtempSync(join(tmpdir(), 'asr-malformed-reference-'));
    try {
      const reference = join(directory, 'reference.json');
      writeFileSync(reference, '{ invalid JSON');
      const result = runFixtureTests(['tests/whisper-mel-validation.test.ts'], {
        WHISPER_MEL_REFERENCE_JSON: reference,
      });
      expect(result.status).toBe(1);
      expect(result.report.success).toBe(false);
      expect(result.stdout).toContain('SyntaxError');
    } finally {
      rmSync(directory, { recursive: true });
    }
  }, 45_000);
});
