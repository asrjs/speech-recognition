import {
  compareStageCaptures,
  runCli,
} from '../tools/model-debugging/scripts/node-compare-stage-captures.mjs';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

function capture(logit = 1, sampleId = 'clip-a') {
  return {
    schema_version: 1,
    samples: [
      {
        sample_id: sampleId,
        audio: { sha256: 'audio-a' },
        tokens: [10, 11, 12],
        transcript: 'hello world',
        eos: 99,
        stages: {
          features: { data: [0.25, 0.5], shape: [1, 2], dtype: 'float32' },
          logits: { data: [logit, 0.2, -0.1], shape: [1, 3], dtype: 'float32' },
        },
      },
    ],
  };
}

describe('stage capture comparator', () => {
  it.each([
    { label: 'different dtype', metadata: { shape: [1, 2], dtype: 'float16' }, both: false },
    { label: 'missing dtype', metadata: { shape: [1, 2] }, both: true },
    { label: 'missing shape', metadata: { dtype: 'float32' }, both: true },
    { label: 'negative dimensions', metadata: { shape: [-1, -2], dtype: 'float32' }, both: true },
    { label: 'coerced dimensions', metadata: { shape: ['1', 2], dtype: 'float32' }, both: true },
    { label: 'wrong element count', metadata: { shape: [2, 2], dtype: 'float32' }, both: true },
  ])('does not certify tensor contracts with $label', ({ metadata, both }) => {
    const sample = capture().samples[0];
    const invalid = {
      ...capture(),
      samples: [
        { ...sample, stages: { ...sample.stages, features: { data: [0.25, 0.5], ...metadata } } },
      ],
    };
    const report = compareStageCaptures(both ? invalid : capture(), invalid);
    expect(report.comparison.pass).toBe(false);
    expect(report.samples[0]?.first_failed_stage).toBe('features');
  });

  it('rejects samples with no stages and stages with no numerical values', () => {
    const emptyStages = { ...capture(), samples: [{ ...capture().samples[0], stages: {} }] };
    expect(() => compareStageCaptures(emptyStages, emptyStages)).toThrow('no captured stages');
    const emptyTensor = capture();
    emptyTensor.samples[0].stages.features.data = [];
    emptyTensor.samples[0].stages.features.shape = [0];
    expect(() => compareStageCaptures(emptyTensor, emptyTensor)).toThrow(
      'at least one captured value',
    );
  });

  it('fails when the reference audio hash is omitted by the candidate', () => {
    const candidate = { ...capture(), samples: [{ ...capture().samples[0], audio: {} }] };
    expect(compareStageCaptures(capture(), candidate).samples[0]?.first_failed_stage).toBe(
      'audio_identity',
    );
    expect(compareStageCaptures(candidate, candidate).comparison.pass).toBe(false);
  });

  it('compares compressed captures through the real CLI', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-stage-capture-'));
    try {
      const reference = path.join(directory, 'reference.json.gz');
      const candidate = path.join(directory, 'candidate.json');
      const output = path.join(directory, 'report.json');
      fs.writeFileSync(reference, gzipSync(JSON.stringify(capture())));
      fs.writeFileSync(candidate, JSON.stringify(capture()));
      expect(runCli(['--reference', reference, '--candidate', candidate, '--output', output])).toBe(
        0,
      );
      fs.writeFileSync(candidate, JSON.stringify(capture(9)));
      expect(runCli(['--reference', reference, '--candidate', candidate, '--output', output])).toBe(
        1,
      );
      expect(JSON.parse(fs.readFileSync(output, 'utf8')).samples[0].first_failed_stage).toBe(
        'logits',
      );
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it('fails on changed or missing TDT durations and token frame indices', () => {
    const reference = capture();
    const timed = {
      ...reference,
      samples: [{ ...reference.samples[0], token_duration: [1, 0, 2], token_frames: [0, 1, 1] }],
    };
    const changedDuration = {
      ...timed,
      samples: [{ ...timed.samples[0], token_duration: [1, 1, 2] }],
    };
    const changedFrame = { ...timed, samples: [{ ...timed.samples[0], token_frames: [0, 1, 2] }] };
    expect(compareStageCaptures(timed, changedDuration).samples[0]?.first_failed_stage).toBe(
      'token_duration',
    );
    expect(compareStageCaptures(timed, changedFrame).samples[0]?.first_failed_stage).toBe(
      'token_frames',
    );
    expect(compareStageCaptures(timed, reference).comparison.pass).toBe(false);
    expect(compareStageCaptures(timed, timed).comparison.pass).toBe(true);
  });

  it('rejects duration/frame metadata with the wrong token count', () => {
    const reference = capture();
    for (const field of ['token_duration', 'token_frames']) {
      const malformed = { ...reference, samples: [{ ...reference.samples[0], [field]: [1] }] };
      expect(() => compareStageCaptures(malformed, malformed)).toThrow('length must match');
    }
  });

  it('reports divergence in the reference pipeline order rather than alphabetical order', () => {
    const reference = capture();
    reference.samples[0].stages = {
      features: { data: [0.25, 0.5], shape: [1, 2], dtype: 'float32' },
      logits: { data: [1, 0.2, -0.1], shape: [1, 3], dtype: 'float32' },
    };
    const candidate = capture(9);
    candidate.samples[0].stages.features.data[0] = 9;
    const ordered = { ...reference, stage_order: ['features', 'logits'] };
    expect(compareStageCaptures(ordered, candidate).samples[0]?.first_failed_stage).toBe(
      'features',
    );
    expect(
      compareStageCaptures({ ...reference, stage_order: ['logits', 'features'] }, candidate)
        .samples[0]?.first_failed_stage,
    ).toBe('logits');
  });

  it('fails when a declared required stage is absent from both captures', () => {
    const report = compareStageCaptures({ ...capture(), stage_order: ['encoder'] }, capture());
    expect(report.comparison.pass).toBe(false);
    expect(report.samples[0]?.first_failed_stage).toBe('encoder');
  });

  it('rejects empty captures and ambiguous stage orders instead of returning parity PASS', () => {
    expect(() => compareStageCaptures({ samples: [] }, { samples: [] })).toThrow('at least one');
    expect(() =>
      compareStageCaptures({ ...capture(), stage_order: ['features', 'features'] }, capture()),
    ).toThrow('unique non-empty');
  });

  it('aligns by sample_id and reports numerical metrics and argmax', () => {
    const report = compareStageCaptures(capture(), capture(1.000001));
    expect(report.comparison.pass).toBe(true);
    expect(report.samples[0]?.first_failed_stage).toBeNull();
    expect(report.samples[0]?.stages.logits.stats.argmax.match).toBe(true);
    expect(report.samples[0]?.stages.logits.stats.cosine).toBeCloseTo(1, 6);
    expect(report.samples[0]?.outputs.tokens.firstMismatch).toBeNull();
    expect(report.samples[0]?.outputs.transcript.match).toBe(true);
    expect(report.samples[0]?.outputs.eos.match).toBe(true);
  });

  it('identifies the earliest failed stage without relying on row order or text', () => {
    const candidate = {
      ...capture(),
      samples: [
        {
          ...capture(1, 'clip-b').samples[0],
          sample_id: 'clip-a',
          stages: {
            features: { data: [0.25, 0.9], shape: [1, 2], dtype: 'float32' },
            logits: { data: [0.1, 0.8, -0.1], shape: [1, 3], dtype: 'float32' },
          },
        },
      ],
    };
    const report = compareStageCaptures(capture(), candidate, {
      absTolerance: 1e-6,
      relTolerance: 0,
    });
    expect(report.comparison.pass).toBe(false);
    expect(report.samples[0]?.first_failed_stage).toBe('features');
    expect(report.samples[0]?.stages.features.stats.firstMismatch?.index).toBe(1);
    expect(report.samples[0]?.stages.logits.stats.argmax.match).toBe(false);
  });

  it('reports the first token divergence after tensor stages pass', () => {
    const candidate = {
      ...capture(),
      samples: [{ ...capture().samples[0], tokens: [10, 77, 12] }],
    };
    const report = compareStageCaptures(capture(), candidate);
    expect(report.comparison.pass).toBe(false);
    expect(report.samples[0]?.first_failed_stage).toBe('tokens');
    expect(report.samples[0]?.outputs.tokens.firstMismatch).toEqual({
      index: 1,
      reference: 11,
      candidate: 77,
    });
  });

  it('makes missing and extra stable sample ids visible', () => {
    const reference = {
      ...capture(),
      samples: [capture().samples[0], { ...capture(2).samples[0], sample_id: 'clip-b' }],
    };
    const report = compareStageCaptures(reference, capture());
    expect(report.comparison.pass).toBe(false);
    expect(report.comparison.reference_only_sample_ids).toEqual(['clip-b']);
    expect(report.samples.find((sample) => sample.sample_id === 'clip-b')?.failure).toBe(
      'missing_candidate_sample',
    );
  });

  it('rejects identical sample ids when their audio identities differ', () => {
    const candidate = {
      ...capture(),
      samples: [{ ...capture().samples[0], audio: { sha256: 'audio-b' } }],
    };
    const report = compareStageCaptures(capture(), candidate);
    expect(report.comparison.pass).toBe(false);
    expect(report.samples[0]?.failure).toBe('audio_identity_mismatch');
    expect(report.samples[0]?.first_failed_stage).toBe('audio_identity');
  });
});
