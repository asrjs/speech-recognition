import { compareStageCaptures } from '../tools/model-debugging/scripts/node-compare-stage-captures.mjs';
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
