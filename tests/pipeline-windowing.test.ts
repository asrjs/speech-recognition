import {
  createDefaultModelInferenceLimits,
  addWindowMetrics,
  buildWindowedMetrics,
  createWindowedMetricsAccumulator,
  dedupeWindowWords,
  partitionWordsIntoSegments,
  resolveWindowPolicy,
  resolveTranscriptDetail,
  transcribeWithWindowing,
  type TranscriptResult,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

function word(index: number, text: string, startTime: number, endTime: number) {
  return { index, text, startTime, endTime };
}

function result(words: ReturnType<typeof word>[], offsetText = ''): TranscriptResult {
  return {
    text: offsetText || words.map((item) => item.text).join(' '),
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

function segmentOnlyResult(text: string): TranscriptResult {
  return {
    text,
    warnings: [],
    meta: {
      detailLevel: 'segments',
      isFinal: true,
      metrics: {
        totalMs: 10,
        audioDurationSec: 2,
      },
    },
    segments: [{ index: 0, text, startTime: 0, endTime: 2 }],
  };
}

describe('pipeline windowing primitives', () => {
  it('preserves additive decoder and encoder telemetry across windows', () => {
    const accumulator = createWindowedMetricsAccumulator(20);
    accumulator.windowCount = 2;
    addWindowMetrics(accumulator, {
      decoderStepMs: 10,
      decoderStepCount: 2,
      decoderGpuTensorDownloads: 3,
      encoderFrameCount: 100,
      decodeIterations: 2,
      decoderKvCacheLocation: 'cpu',
      preprocessorBackend: 'js',
      totalMs: 100,
    });
    addWindowMetrics(accumulator, {
      decoderStepMs: 20,
      decoderStepCount: 4,
      decoderGpuTensorDownloads: 5,
      encoderFrameCount: 200,
      decodeIterations: 4,
      decoderKvCacheLocation: 'cpu',
      preprocessorBackend: 'js',
      totalMs: 200,
    });

    expect(buildWindowedMetrics(accumulator)).toMatchObject({
      decoderStepMs: 30,
      decoderStepCount: 6,
      decoderStepAvgMs: 5,
      decoderGpuTensorDownloads: 8,
      encoderFrameCount: 300,
      decodeIterations: 6,
      decoderKvCacheLocation: 'cpu',
      preprocessorBackend: 'js',
      windowCount: 2,
      totalMs: 300,
    });

    const mixedAccumulator = createWindowedMetricsAccumulator(2);
    mixedAccumulator.windowCount = 2;
    addWindowMetrics(mixedAccumulator, { decoderKvCacheLocation: 'cpu' });
    addWindowMetrics(mixedAccumulator, { decoderKvCacheLocation: 'gpu' });
    expect(buildWindowedMetrics(mixedAccumulator)?.decoderKvCacheLocation).toBe('mixed');
  });

  it('maps transformers-style return options onto canonical detail levels', () => {
    expect(resolveTranscriptDetail({ returnTimestamps: true })).toBe('segments');
    expect(resolveTranscriptDetail({ returnTimestamps: 'word' })).toBe('words');
    expect(resolveTranscriptDetail({ returnTimestamps: 'sentences' })).toBe('sentences');
    expect(resolveTranscriptDetail({ returnWords: true })).toBe('words');
    expect(resolveTranscriptDetail({ returnTokens: true })).toBe('detailed');
    expect(resolveTranscriptDetail({ detail: 'text', returnTokens: true })).toBe('text');
  });

  it('resolves model-specific Parakeet and Whisper defaults', () => {
    const parakeet = createDefaultModelInferenceLimits({ family: 'nemo-tdt', modelId: 'parakeet' });
    const whisper = createDefaultModelInferenceLimits({
      family: 'whisper-seq2seq',
      modelId: 'whisper',
    });

    expect(resolveWindowPolicy({ inference: parakeet }).windowDurationSec).toBe(90);
    expect(resolveWindowPolicy({ inference: parakeet }).maxWindowDurationSec).toBe(180);
    expect(resolveWindowPolicy({ inference: whisper }).windowDurationSec).toBe(30);
    expect(
      resolveWindowPolicy({ inference: whisper, windowDurationSeconds: 90 }).windowDurationSec,
    ).toBe(30);
    expect(
      resolveWindowPolicy({
        inference: whisper,
        windowDurationSeconds: 90,
        unsafeAllowOverMaxWindow: true,
      }).windowDurationSec,
    ).toBe(90);
  });

  it('segments timestamped words conservatively', () => {
    const segments = partitionWordsIntoSegments([
      word(0, 'Dr.', 0, 0.2),
      word(1, 'Smith', 0.3, 0.8),
      word(2, 'arrived.', 0.9, 1.4),
      word(3, 'He', 1.5, 1.7),
      word(4, 'left?', 1.8, 2.1),
      word(5, 'Yes', 5.5, 5.8),
    ]);

    expect(segments.map((segment) => segment.text)).toEqual([
      'Dr. Smith arrived.',
      'He left?',
      'Yes',
    ]);
  });

  it('dedupes overlapping words by normalized text', () => {
    expect(
      dedupeWindowWords([
        word(0, 'hello', 0, 0.5),
        word(1, 'Hello,', 0.2, 1.0),
        word(2, 'world', 1.1, 1.5),
      ]).map((item) => item.text),
    ).toEqual(['Hello,', 'world']);
  });

  it('routes long audio through window transcription and merges words', async () => {
    const audio = new Float32Array(4 * 16000);
    let calls = 0;
    const transcript = await transcribeWithWindowing({
      input: audio,
      inference: {
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
      },
      options: { detail: 'words' },
      async transcribeWindow(_windowAudio) {
        calls += 1;
        if (calls === 1) {
          return result([word(0, 'Hello', 0, 0.4), word(1, 'world.', 0.5, 1.0)]);
        }
        return result([word(0, 'Again', 0, 0.4), word(1, 'done.', 0.5, 1.0)]);
      },
    });

    expect(calls).toBeGreaterThan(1);
    expect(transcript.meta.metrics?.windowCount).toBe(calls);
    expect(transcript.text).toContain('Hello world.');
    expect(transcript.words?.length).toBeGreaterThan(2);
    expect(transcript.meta.metrics?.rtf).toBeGreaterThan(0);
  });

  it('suppresses token overlap for segment-only long-audio results', async () => {
    const audio = new Float32Array(4 * 16000);
    let calls = 0;
    const transcript = await transcribeWithWindowing({
      input: audio,
      inference: {
        sampleRate: 16000,
        maxInputDurationSec: 2,
        recommendedWindowDurationSec: 2,
        minWindowDurationSec: 1,
        maxWindowDurationSec: 2,
        autoWindowThresholdSec: 2,
        defaultOverlapSec: 0.5,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: true,
        defaultSegmentationStrategy: 'word-punctuation',
        defaultMergeStrategy: 'word-dedupe',
      },
      options: { detail: 'segments' },
      async transcribeWindow() {
        const texts = ['alpha beta gamma', 'beta gamma delta', 'gamma delta epsilon'];
        const text = texts[Math.min(calls, texts.length - 1)]!;
        calls += 1;
        return segmentOnlyResult(text);
      },
    });

    expect(calls).toBe(3);
    expect(transcript.text).toBe('alpha beta gamma delta epsilon');
    expect(transcript.segments?.map((segment) => segment.text)).toEqual([
      'alpha beta gamma delta epsilon',
    ]);
  });

  it('uses temporal overlap to trim divergent segment-only window prefixes', async () => {
    const audio = new Float32Array(9 * 16000);
    let calls = 0;
    const texts = [
      'alpha beta gamma delta epsilon zeta',
      'noise artifact theta iota kappa lambda',
      'other artifact mu nu xi omicron',
      'tail artifact pi rho sigma tau',
    ];
    const transcript = await transcribeWithWindowing({
      input: audio,
      inference: {
        sampleRate: 16000,
        maxInputDurationSec: 3,
        recommendedWindowDurationSec: 3,
        minWindowDurationSec: 1,
        maxWindowDurationSec: 3,
        autoWindowThresholdSec: 3,
        defaultOverlapSec: 1,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: true,
        defaultSegmentationStrategy: 'word-punctuation',
        defaultMergeStrategy: 'concat',
      },
      options: { detail: 'segments' },
      async transcribeWindow(windowAudio) {
        const text = texts[Math.min(calls, texts.length - 1)]!;
        calls += 1;
        return {
          text,
          warnings: [],
          meta: {
            detailLevel: 'segments' as const,
            isFinal: true,
            metrics: { totalMs: 10, audioDurationSec: windowAudio.durationSeconds },
          },
          segments: [{ index: 0, text, startTime: 0, endTime: windowAudio.durationSeconds }],
        };
      },
    });

    expect(calls).toBe(4);
    expect(transcript.text).toBe(
      'alpha beta gamma delta epsilon zeta theta iota kappa lambda mu nu xi omicron pi rho sigma tau',
    );
  });

  it('removes an exact overlap that starts after a divergent temporal prefix', async () => {
    const audio = new Float32Array(5 * 16000);
    let calls = 0;
    const texts = ['alpha beta gamma delta epsilon zeta', 'noise artifact epsilon zeta theta iota'];
    const transcript = await transcribeWithWindowing({
      input: audio,
      inference: {
        sampleRate: 16000,
        maxInputDurationSec: 3,
        recommendedWindowDurationSec: 3,
        minWindowDurationSec: 1,
        maxWindowDurationSec: 3,
        autoWindowThresholdSec: 3,
        defaultOverlapSec: 1,
        supportsWordTimestamps: false,
        supportsSegmentTimestamps: true,
        defaultSegmentationStrategy: 'word-punctuation',
        defaultMergeStrategy: 'concat',
      },
      options: { detail: 'segments' },
      async transcribeWindow(windowAudio) {
        const text = texts[Math.min(calls, texts.length - 1)]!;
        calls += 1;
        return {
          text,
          warnings: [],
          meta: {
            detailLevel: 'segments' as const,
            isFinal: true,
            metrics: { totalMs: 10, audioDurationSec: windowAudio.durationSeconds },
          },
          segments: [{ index: 0, text, startTime: 0, endTime: windowAudio.durationSeconds }],
        };
      },
    });

    expect(calls).toBe(2);
    expect(transcript.text).toBe('alpha beta gamma delta epsilon zeta theta iota');
  });
});
