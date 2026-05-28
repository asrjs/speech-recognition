import {
  createPipelineContext,
  createTranscriptOutputStage,
  runPipelineStages,
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

describe('transcript output sidecar stage', () => {
  it('emits requested JSON, SRT, and VTT sidecars', async () => {
    const result = await runPipelineStages(
      createPipelineContext({ transcript: transcript() }),
      [createTranscriptOutputStage({ formats: ['json', 'srt', 'vtt'] })],
    );

    expect(result.sidecars.json).toMatchObject({ text: 'Hello world.' });
    expect(result.sidecars.srt).toContain('00:00:00,000 --> 00:00:01,200');
    expect(result.sidecars.vtt).toMatch(/^WEBVTT\n\n/);
    expect(result.completedStageIds).toEqual(['transcript-output-sidecars']);
  });

  it('emits no sidecars when transcript is missing', async () => {
    const result = await runPipelineStages(
      createPipelineContext({ sidecars: { existing: 'keep' } }),
      [createTranscriptOutputStage({ formats: ['json', 'srt', 'vtt'] })],
    );

    expect(result.sidecars).toEqual({ existing: 'keep' });
    expect(result.completedStageIds).toEqual(['transcript-output-sidecars']);
  });

  it('overwrites only explicitly requested sidecar keys', async () => {
    const result = await runPipelineStages(
      createPipelineContext({
        transcript: transcript('Replacement.'),
        sidecars: {
          json: { text: 'old' },
          srt: 'old srt',
          vtt: 'old vtt',
          custom: 'keep',
        },
      }),
      [createTranscriptOutputStage({ formats: ['json', 'vtt'] })],
    );

    expect(result.sidecars.json).toMatchObject({ text: 'Replacement.' });
    expect(result.sidecars.vtt).toContain('Replacement.');
    expect(result.sidecars.srt).toBe('old srt');
    expect(result.sidecars.custom).toBe('keep');
  });

  it('can be configured with a custom stage id', async () => {
    const result = await runPipelineStages(
      createPipelineContext({ transcript: transcript() }),
      [createTranscriptOutputStage({ id: 'exports', formats: ['json'] })],
    );

    expect(result.completedStageIds).toEqual(['exports']);
    expect(result.sidecars.json).toMatchObject({ meta: { isFinal: true } });
  });
});
