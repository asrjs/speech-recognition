import {
  createPipelineContext,
  createWindowingStage,
  PipelineStageError,
  runPipelineStages,
  type ModelInferenceLimits,
  type TranscriptResult,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function word(index: number, text: string, startTime: number, endTime: number) {
  return { index, text, startTime, endTime };
}

function transcript(words: ReturnType<typeof word>[], text = words.map((item) => item.text).join(' ')): TranscriptResult {
  return {
    text,
    warnings: [],
    meta: {
      detailLevel: 'words',
      isFinal: true,
      metrics: {
        totalMs: 10,
        audioDurationSec: 1,
      },
    },
    words,
  };
}

const inference: ModelInferenceLimits = {
  sampleRate: 16000,
  maxInputDurationSec: 2,
  recommendedWindowDurationSec: 2,
  minWindowDurationSec: 1,
  maxWindowDurationSec: 2,
  autoWindowThresholdSec: 2,
  defaultOverlapSec: 0.5,
  supportsWordTimestamps: true,
  supportsSegmentTimestamps: true,
  defaultSegmentationStrategy: 'word-punctuation',
  defaultMergeStrategy: 'word-dedupe',
};

describe('windowing pipeline stage', () => {
  it('routes long audio through windowing and returns a merged transcript', async () => {
    const audio = new Float32Array(4 * 16000);
    let calls = 0;
    const result = await runPipelineStages(
      createPipelineContext({ input: audio, options: { detail: 'words' } }),
      [
        createWindowingStage({
          inference,
          async transcribeWindow(_windowAudio, options) {
            calls += 1;
            expect(options.detail).toBe('words');
            if (calls === 1) {
              return transcript([word(0, 'Hello', 0, 0.4), word(1, 'world.', 0.5, 1.0)]);
            }
            return transcript([word(0, 'Again', 0, 0.4), word(1, 'done.', 0.5, 1.0)]);
          },
        }),
      ],
    );

    expect(calls).toBeGreaterThan(1);
    expect(result.transcript?.text).toContain('Hello world.');
    expect(result.transcript?.words?.length).toBeGreaterThan(2);
    expect(result.completedStageIds).toEqual(['windowing']);
  });

  it('uses direct fallback when windowing is disabled', async () => {
    const audio = new Float32Array(4 * 16000);
    let windowCalls = 0;
    let directCalls = 0;
    const result = await runPipelineStages(
      createPipelineContext({ input: audio, options: { detail: 'words', windowing: 'disabled' } }),
      [
        createWindowingStage({
          inference,
          async transcribeWindow() {
            windowCalls += 1;
            return transcript([], 'window');
          },
          async transcribeDirect(input, options) {
            directCalls += 1;
            expect(input).toBe(audio);
            expect(options.windowing).toBe('disabled');
            return transcript([word(0, 'direct', 0, 0.3)], 'direct');
          },
        }),
      ],
    );

    expect(windowCalls).toBe(0);
    expect(directCalls).toBe(1);
    expect(result.transcript?.text).toBe('direct');
  });

  it('passes context signal into transcription options', async () => {
    const signal = { aborted: false };
    let seenSignal: unknown;
    await runPipelineStages(
      createPipelineContext({ input: new Float32Array(16000), options: {}, signal }),
      [
        createWindowingStage({
          inference,
          async transcribeWindow(_windowAudio, options) {
            seenSignal = options.signal;
            return transcript([word(0, 'ok', 0, 0.2)]);
          },
        }),
      ],
    );

    expect(seenSignal).toBe(signal);
  });

  it('throws a useful stage error when input is missing', async () => {
    const stage = createWindowingStage({
      inference,
      async transcribeWindow() {
        return transcript([], 'must not run');
      },
    });

    await expect(runPipelineStages(createPipelineContext(), [stage])).rejects.toMatchObject({
      constructor: PipelineStageError,
      stageId: 'windowing',
      cause: expect.objectContaining({ message: expect.stringContaining('requires context.input') }),
    });
  });
});
