import {
  BENCHMARK_RUN_CSV_COLUMNS,
  benchmarkRunRecordsToCsv,
  calcRtfx,
  flattenBenchmarkRunRecord,
  levenshteinDistance,
  normalizeBenchmarkText,
  summarizeNumericSeries,
  textSimilarity,
  toCsv,
} from '@asrjs/speech-recognition/bench';
import {
  extractAudioUrl,
  fetchRandomRows,
  getConfigsAndSplits,
  normalizeDatasetRow,
  normalizeReferenceText,
} from '@asrjs/speech-recognition/datasets';
import { describe, expect, it, vi } from 'vitest';

describe('benchmark and dataset helpers', () => {
  it('normalizes text and computes similarity metrics for benchmark comparisons', () => {
    expect(normalizeBenchmarkText('Hello,   World!')).toBe('hello world');
    expect(levenshteinDistance('kitten', 'sitting')).toBe(3);
    expect(textSimilarity('Hello world', 'hello, world!')).toBeCloseTo(1, 5);
  });

  it('summarizes numeric series and exports flat run csv', () => {
    const summary = summarizeNumericSeries([1, 2, 3, 4, 5]);
    expect(summary.mean).toBe(3);
    expect(summary.median).toBe(3);
    expect(summary.p90).toBe(5);
    expect(calcRtfx(10, 2000)).toBe(5);

    const flattened = flattenBenchmarkRunRecord({
      id: 'run-1',
      sampleKey: 'sample-a',
      audioDurationSec: 12,
      transcription: 'hello',
      metrics: {
        encode_ms: 1000,
        decode_ms: 500,
        total_ms: 1700,
      },
    });
    expect(flattened.encode_rtfx).toBe(12);
    expect(flattened.decode_rtfx).toBe(24);

    const csv = benchmarkRunRecordsToCsv([
      {
        id: 'run-1',
        sampleKey: 'sample-a',
        audioDurationSec: 12,
        transcription: 'hello',
        metrics: {
          encode_ms: 1000,
          decode_ms: 500,
        },
      },
    ]);
    expect(csv.startsWith(BENCHMARK_RUN_CSV_COLUMNS.join(','))).toBe(true);
    expect(toCsv([{ alpha: 'a', beta: 2 }], ['alpha', 'beta'])).toContain('alpha,beta');
  });

  it('normalizes dataset rows and extracts audio urls from nested shapes', () => {
    expect(extractAudioUrl({ src: 'https://example/audio.wav' })).toBe('https://example/audio.wav');
    expect(extractAudioUrl([{ url: 'https://example/array.wav' }])).toBe(
      'https://example/array.wav',
    );
    expect(normalizeReferenceText('A PARAGRAPH B NEWLINE C')).toBe('A\nB\nC');

    const row = normalizeDatasetRow({
      row_idx: 4,
      row: {
        audio: { src: 'https://example/audio.wav' },
        transcription: 'hello world',
        speaker: 'speaker-a',
        gender: 'f',
        sample_rate: 22050,
      },
    });
    expect(row.rowIndex).toBe(4);
    expect(row.audioUrl).toBe('https://example/audio.wav');
    expect(row.referenceText).toBe('hello world');
    expect(row.sampleRate).toBe(22050);
  });

  it('groups dataset configs and supports deterministic random row sampling', async () => {
    const configs = getConfigsAndSplits([
      { config: 'en', split: 'train' },
      { config: 'en', split: 'validation' },
      { config: 'fr', split: 'train' },
    ]);
    expect(configs.get('en')).toEqual(['train', 'validation']);

    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({
        rows: Array.from({ length: 100 }, (_, index) => ({
          row_idx: index,
          row: { audio: { src: `https://example/${index}.wav` }, transcription: `sample-${index}` },
        })),
      }),
      text: async () => '',
      headers: new Headers(),
    } as Response);

    try {
      const result = await fetchRandomRows({
        dataset: 'demo',
        config: 'en',
        split: 'train',
        totalRows: 100,
        sampleCount: 3,
        seed: 'fixed-seed',
      });

      expect(result.rows).toHaveLength(3);
      expect(result.offsets).toHaveLength(3);
      expect(result.seedUsed).not.toBeNull();
    } finally {
      fetchSpy.mockRestore();
    }
  });
});
