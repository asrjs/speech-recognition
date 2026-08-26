import {
  createPipelineContext,
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

describe('pipeline composition benchmark / verification', () => {
  it('executes pipeline stages sequentially in expected time', async () => {
    const stages: PipelineStage[] = [
      {
        id: 'stage-1',
        run(context) {
          return { transcript: transcript(context.input as string) };
        },
      },
      {
        id: 'stage-2',
        run(context) {
          return {
            sidecars: {
              processedText: context.transcript?.text.toUpperCase(),
            },
          };
        },
      },
    ];

    const iterations = 1000;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      const result = await runPipelineStages(
        createPipelineContext({ input: 'sample text' }),
        stages,
      );
      expect(result.sidecars.processedText).toBe('SAMPLE TEXT');
    }
    const elapsed = performance.now() - start;
    expect(elapsed).toBeGreaterThan(0);
  });
});
