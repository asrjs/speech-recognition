import {
  createDefaultModelInferenceLimits,
  addWindowMetrics,
  buildWindowedMetrics,
  createWindowedMetricsAccumulator,
  dedupeWindowWords,
  partitionWordsIntoSegments,
  planWindowedTranscription,
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
  it('preserves actual component providers and reports mixed window placement', () => {
    const accumulator = createWindowedMetricsAccumulator(40);
    const hybrid = { encoderBackend: 'webgpu', decoderBackend: 'wasm', jointBackend: 'wasm' };
    addWindowMetrics(accumulator, hybrid);
    addWindowMetrics(accumulator, hybrid);
    expect(buildWindowedMetrics(accumulator)).toMatchObject(hybrid);
    addWindowMetrics(accumulator, {
      encoderBackend: 'wasm', decoderBackend: 'webgpu', jointBackend: 'webgpu',
    });
    expect(buildWindowedMetrics(accumulator)).toMatchObject({
      encoderBackend: 'mixed', decoderBackend: 'mixed', jointBackend: 'mixed',
    });
  });

  it('resolves token aliases at the public windowing boundary while honoring explicit detail', () => {
    const input = new Float32Array(16000);
    expect(planWindowedTranscription(input, { returnTokens: true }, undefined).options.detail).toBe(
      'detailed',
    );
    expect(
      planWindowedTranscription(input, { returnTokens: true, detail: 'text' }, undefined).options
        .detail,
    ).toBe('text');
  });

  it.each(['id', 'rawText', 'interval'] as const)(
    'does not collapse tokens with different %s',
    async (difference) => {
      let calls = 0;
      const transcript = await transcribeWithWindowing({
        input: new Float32Array(3.5 * 16000),
        inference: {
          ...createDefaultModelInferenceLimits({ family: 'nemo-tdt' }),
          maxInputDurationSec: 2,
          recommendedWindowDurationSec: 2,
          minWindowDurationSec: 1,
          maxWindowDurationSec: 2,
          autoWindowThresholdSec: 2,
          defaultOverlapSec: 0.5,
        },
        options: { detail: 'detailed' },
        async transcribeWindow() {
          const first = calls++ === 0;
          return {
            ...result([]),
            tokens: [
              {
                index: 0,
                text: 'same',
                id: !first && difference === 'id' ? 2 : 1,
                rawText: !first && difference === 'rawText' ? '▁same' : 'same',
                startTime: first ? 1.6 : 0.1,
                endTime: first ? 1.9 : difference === 'interval' ? 0.5 : 0.4,
              },
            ],
          };
        },
      });
      expect(calls).toBe(2);
      expect(transcript.tokens).toHaveLength(2);
      expect(transcript.tokens?.map(({ index }) => index)).toEqual([0, 1]);
      expect(transcript.warnings.map(({ code }) => code)).toContain(
        'pipeline.windowing.token-overlap-unresolved',
      );
    },
  );

  it.each(['words', 'detailed'] as const)(
    'does not invent token references for %s windows',
    async (detail) => {
      let calls = 0;
      const transcript = await transcribeWithWindowing({
        input: new Float32Array(4 * 16000),
        inference: {
          ...createDefaultModelInferenceLimits({ family: 'nemo-tdt' }),
          maxInputDurationSec: 2,
          recommendedWindowDurationSec: 2,
          minWindowDurationSec: 1,
          maxWindowDurationSec: 2,
          autoWindowThresholdSec: 2,
          defaultOverlapSec: 0,
        },
        options: { detail },
        async transcribeWindow() {
          calls++;
          return { ...result([]), words: [{ ...word(0, 'hello', 0.1, 0.4), tokenIndices: [99] }] };
        },
      });
      expect(calls).toBe(2);
      expect(transcript.tokens).toBeUndefined();
      if (detail === 'detailed') {
        expect(transcript.words?.every((item) => item.tokenIndices === undefined)).toBe(true);
        expect(transcript.warnings.map(({ code }) => code)).toContain(
          'pipeline.windowing.token-overlap-unresolved',
        );
      }
    },
  );

  it.each([0, 0.5])(
    'preserves detailed tokens and word references with overlap %s',
    async (overlap) => {
      const locals = [
        {
          ...result([word(0, 'alpha', 0.1, 0.4), word(1, 'shared', 1.6, 1.9)]),
          tokens: [
            { index: 7, id: 11, text: 'alpha', startTime: 0.1, endTime: 0.4, confidence: 0.9 },
            { index: 8, id: 12, text: 'shared', startTime: 1.6, endTime: 1.9 },
          ],
        },
        {
          ...result([word(0, 'shared', 0.1, 0.4), word(1, 'omega', 0.8, 1.2)]),
          tokens: [
            { index: 7, id: 12, text: 'shared', startTime: 0.1, endTime: 0.4 },
            { index: 8, id: 13, text: 'omega', startTime: 0.8, endTime: 1.2 },
          ],
        },
      ].map((local) => ({
        ...local,
        words: local.words!.map((item, index) => ({ ...item, tokenIndices: [7 + index] })),
      }));
      let calls = 0;
      const transcript = await transcribeWithWindowing({
        input: new Float32Array((4 - overlap) * 16000),
        inference: {
          ...createDefaultModelInferenceLimits({ family: 'nemo-tdt' }),
          maxInputDurationSec: 2,
          recommendedWindowDurationSec: 2,
          minWindowDurationSec: 1,
          maxWindowDurationSec: 2,
          autoWindowThresholdSec: 2,
          defaultOverlapSec: overlap,
        },
        options: { detail: 'detailed' },
        async transcribeWindow(_audio, options) {
          expect(options.detail).toBe('detailed');
          return locals[calls++]!;
        },
      });
      expect(calls).toBe(2);
      expect(
        transcript.tokens?.map(({ index, id, startTime, endTime }) => [
          index,
          id,
          startTime,
          endTime,
        ]),
      ).toEqual(
        overlap
          ? [
              [0, 11, 0.1, 0.4],
              [1, 12, 1.6, 1.9],
              [2, 13, 2.3, 2.7],
            ]
          : [
              [0, 11, 0.1, 0.4],
              [1, 12, 1.6, 1.9],
              [2, 12, 2.1, 2.4],
              [3, 13, 2.8, 3.2],
            ],
      );
      expect(transcript.words?.map(({ tokenIndices }) => tokenIndices)).toEqual(
        overlap ? [[0], [1], [2]] : [[0], [1], [2], [3]],
      );
      expect(transcript.tokens?.[0]?.confidence).toBe(0.9);
      expect(locals[1]?.tokens[0]).toMatchObject({ index: 7, startTime: 0.1 });
    },
  );

  it('retains untimed overlapping tokens with an explicit ambiguity warning', async () => {
    let calls = 0;
    const transcript = await transcribeWithWindowing({
      input: new Float32Array(3.5 * 16000),
      inference: {
        ...createDefaultModelInferenceLimits({ family: 'nemo-tdt' }),
        maxInputDurationSec: 2,
        recommendedWindowDurationSec: 2,
        minWindowDurationSec: 1,
        maxWindowDurationSec: 2,
        autoWindowThresholdSec: 2,
        defaultOverlapSec: 0.5,
      },
      options: { detail: 'detailed' },
      async transcribeWindow() {
        calls++;
        return {
          ...result([word(0, 'repeat', 0.1, 0.4)]),
          tokens: [{ index: 0, id: 1, text: 'repeat' }],
        };
      },
    });
    expect(calls).toBe(2);
    expect(transcript.tokens).toEqual([
      { index: 0, id: 1, text: 'repeat' },
      { index: 1, id: 1, text: 'repeat' },
    ]);
    expect(transcript.warnings.map(({ code }) => code)).toContain(
      'pipeline.windowing.token-overlap-unresolved',
    );
  });

  it.each([2, 3])('retains overlap timestamps and history across %i windows', async (count) => {
    const windows = [
      result([word(0, 'alpha', 0.1, 0.4), word(1, 'shared', 1.6, 1.9)]),
      result([word(0, 'shared', 0.1, 0.5), word(1, 'omega', 0.8, 1.2)]),
      result([word(0, 'tail', 0.1, 0.5)]),
    ];
    let calls = 0;
    const transcript = await transcribeWithWindowing({
      input: new Float32Array((count === 2 ? 3.5 : 5) * 16000),
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
      async transcribeWindow() {
        return windows[calls++]!;
      },
    });
    expect(calls).toBe(count);
    expect(transcript.words?.map(({ text, startTime, endTime }) => [text, startTime, endTime])).toEqual([
      ['alpha', 0.1, 0.4], ['shared', 1.6, 2], ['omega', 2.3, 2.7],
      ...(count === 3 ? [['tail', 3.1, 3.5]] : []),
    ]);
    expect(transcript.words?.map(({ index }) => index)).toEqual(count === 2 ? [0, 1, 2] : [0, 1, 2, 3]);
    expect(transcript.segments?.[0]).toMatchObject({ startTime: 0.1, endTime: count === 2 ? 2.7 : 3.5 });
    expect(windows[1]?.words?.[0]).toMatchObject({ startTime: 0.1, endTime: 0.5 });
  });

  it.each(['words', 'segments'] as const)(
    'globalizes %s timestamps without mutating window-local results',
    async (detail) => {
      const locals: TranscriptResult[] = [];
      const transcript = await transcribeWithWindowing({
        input: new Float32Array(4 * 16000),
        inference: {
          sampleRate: 16000,
          maxInputDurationSec: 2,
          recommendedWindowDurationSec: 2,
          minWindowDurationSec: 1,
          maxWindowDurationSec: 2,
          autoWindowThresholdSec: 2,
          defaultOverlapSec: 0,
          supportsWordTimestamps: detail === 'words',
          supportsSegmentTimestamps: true,
          defaultSegmentationStrategy: 'word-punctuation',
          defaultMergeStrategy: 'concat',
        },
        options: { detail },
        async transcribeWindow(audio) {
          expect(audio.durationSeconds).toBe(2);
          const text = locals.length === 0 ? 'First.' : 'Second.';
          const local: TranscriptResult = {
            ...result(detail === 'words' ? [word(0, text, 0.25, 1.25)] : [], text),
            segments: [{ index: 0, text, startTime: 0.25, endTime: 1.25 }],
          };
          locals.push(local);
          return local;
        },
      });

      expect(locals).toHaveLength(2);
      expect(transcript.segments?.map(({ startTime, endTime }) => [startTime, endTime])).toEqual([
        [0.25, 1.25], [2.25, 3.25],
      ]);
      if (detail === 'words') {
        expect(transcript.words?.map(({ startTime, endTime }) => [startTime, endTime])).toEqual([
          [0.25, 1.25], [2.25, 3.25],
        ]);
      } else {
        expect(transcript.words).toBeUndefined();
      }
      expect(locals[1]?.segments?.[0]?.startTime).toBe(0.25);
      expect(locals[1]?.words?.[0]?.startTime).toBe(detail === 'words' ? 0.25 : undefined);
    },
  );

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
