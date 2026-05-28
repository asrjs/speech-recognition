import {
  createPipelineContext,
  createSubtitleSidecarStage,
  PipelineAbortedError,
  PipelineStageError,
  runPipelineStages,
  type PipelineStage,
  type TranscriptResult,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function transcript(text = 'Hello world.'): TranscriptResult {
  return {
    text,
    warnings: [],
    meta: {
      detailLevel: 'sentences',
      isFinal: true,
      sentenceCount: 1,
    },
    sentences: [
      {
        index: 0,
        text,
        startTime: 0,
        endTime: 1.2,
      },
    ],
  };
}

describe('pluggable pipeline composition', () => {
  it('runs user stages sequentially and stores sidecar outputs', async () => {
    const calls: string[] = [];
    const decodeStage: PipelineStage = {
      id: 'decode',
      async run() {
        calls.push('decode');
        return { transcript: transcript() };
      },
    };
    const metadataStage: PipelineStage = {
      id: 'metadata',
      async run(context) {
        calls.push(`metadata:${context.transcript?.text}`);
        return {
          sidecars: {
            json: context.transcript,
          },
        };
      },
    };

    const result = await runPipelineStages(createPipelineContext(), [
      decodeStage,
      metadataStage,
      createSubtitleSidecarStage({ formats: ['srt', 'vtt'] }),
    ]);

    expect(calls).toEqual(['decode', 'metadata:Hello world.']);
    expect(result.transcript?.text).toBe('Hello world.');
    expect(result.sidecars.srt).toContain('00:00:00,000 --> 00:00:01,200');
    expect(result.sidecars.vtt).toMatch(/^WEBVTT\n\n/);
    expect(result.completedStageIds).toEqual(['decode', 'metadata', 'subtitle-sidecars']);
  });

  it('preserves immutable-ish context snapshots between stages', async () => {
    const initial = createPipelineContext({
      sidecars: { existing: 'keep' },
      transcript: transcript('Original.'),
    });
    const stage: PipelineStage = {
      id: 'replace-transcript',
      run() {
        return {
          transcript: transcript('Replacement.'),
          sidecars: { extra: 42 },
        };
      },
    };

    const result = await runPipelineStages(initial, [stage]);

    expect(initial.transcript?.text).toBe('Original.');
    expect(initial.sidecars).toEqual({ existing: 'keep' });
    expect(result.transcript?.text).toBe('Replacement.');
    expect(result.sidecars).toEqual({ existing: 'keep', extra: 42 });
  });

  it('wraps failing stages with stage id and cause', async () => {
    const cause = new Error('boom');
    const stage: PipelineStage = {
      id: 'bad-stage',
      run() {
        throw cause;
      },
    };

    await expect(runPipelineStages(createPipelineContext(), [stage])).rejects.toMatchObject({
      constructor: PipelineStageError,
      stageId: 'bad-stage',
      cause,
    });
  });

  it('checks abort signals before running each stage', async () => {
    const signal = { aborted: false };
    const first: PipelineStage = {
      id: 'first',
      run(context) {
        if (context.signal) {
          signal.aborted = true;
        }
        return {};
      },
    };
    const second: PipelineStage = {
      id: 'second',
      run() {
        throw new Error('must not run');
      },
    };

    await expect(
      runPipelineStages(createPipelineContext({ signal }), [first, second]),
    ).rejects.toBeInstanceOf(PipelineAbortedError);
  });
});
