import {
  BENCHMARK_LIFECYCLE_CSV_COLUMNS,
  BENCHMARK_RUN_CSV_COLUMNS,
  benchmarkLifecycleRecordsToCsv,
  benchmarkMemoryDeltaBytes,
  benchmarkRunRecordsToCsv,
  calcRtfx,
  createBenchmarkStageMetrics,
  flattenBenchmarkRunRecord,
  flattenBenchmarkLifecycleRecord,
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
      backend: 'webgpu-hybrid',
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
      metrics: {
        encode_ms: 1000,
        decode_ms: 500,
        total_ms: 1700,
        window_count: 3,
        decoder_step_count: 6,
        encoder_frame_count: 300,
      },
    });
    expect(flattened.encode_rtfx).toBe(12);
    expect(flattened.decode_rtfx).toBe(24);
    expect(flattened.backend).toBe('webgpu-hybrid');
    expect(flattened.encoder_backend).toBe('webgpu');
    expect(flattened.decoder_backend).toBe('wasm');
    expect(flattened.window_count).toBe(3);
    expect(flattened.decoder_step_count).toBe(6);
    expect(flattened.encoder_frame_count).toBe(300);

    const csv = benchmarkRunRecordsToCsv([
      {
        id: 'run-1',
        sampleKey: 'sample-a',
        audioDurationSec: 12,
        transcription: 'hello',
        metrics: {
          encode_ms: 1000,
          decode_ms: 500,
          window_count: 2,
          decoder_gpu_tensor_downloads: 8,
        },
      },
    ]);
    expect(csv.startsWith(BENCHMARK_RUN_CSV_COLUMNS.join(','))).toBe(true);
    expect(csv).toContain('window_count');
    expect(csv).toContain('decoder_gpu_tensor_downloads');
    expect(csv).toContain(',2,');
    expect(csv).toContain(',8,');
    expect(toCsv([{ alpha: 'a', beta: 2 }], ['alpha', 'beta'])).toContain('alpha,beta');
  });

  it('projects canonical long-audio telemetry into benchmark rows', () => {
    const metrics = createBenchmarkStageMetrics({
      totalMs: 300,
      rtf: 0.15,
      windowCount: 2,
      decoderStepMs: 30,
      decoderStepCount: 6,
      decoderStepAvgMs: 5,
      decoderGpuTensorDownloads: 8,
      decoderKvCacheLocation: 'cpu',
      encoderRunMs: 120,
      encoderTotalMs: 125,
      encoderFrameCount: 300,
      decodeIterations: 6,
      preprocessorBackend: 'js',
    });

    expect(metrics).toMatchObject({
      window_count: 2,
      decoder_step_ms: 30,
      decoder_step_count: 6,
      decoder_step_avg_ms: 5,
      decoder_gpu_tensor_downloads: 8,
      decoder_kv_cache_location: 'cpu',
      encoder_run_ms: 120,
      encoder_total_ms: 125,
      encoder_frame_count: 300,
      decode_iterations: 6,
      preprocessor_backend: 'js',
    });
  });

  it('exports model lifecycle timing and comparable memory deltas', () => {
    const before = {
      capturedAt: '2026-08-26T00:00:00.000Z',
      source: 'measure-user-agent-specific-memory' as const,
      scope: 'process' as const,
      bytes: 100,
    };
    const after = {
      capturedAt: '2026-08-26T00:00:01.000Z',
      source: 'measure-user-agent-specific-memory' as const,
      scope: 'process' as const,
      bytes: 175,
    };
    const record = {
      id: 'load-1',
      phase: 'model-load' as const,
      totalMs: 900,
      initialArtifactResolutionMs: 600,
      initializationAndRetryMs: 300,
      attemptCount: 1,
      retryUsed: false,
      completedAssetCount: 4,
      reportedAssetBytes: 1024,
      cacheStatus: 'unknown' as const,
      memoryBefore: before,
      memoryAfter: after,
      modelKey: 'parakeet-tdt-0.6b-v3',
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    };

    expect(benchmarkMemoryDeltaBytes(before, after)).toBe(75);
    expect(
      benchmarkMemoryDeltaBytes(before, { ...after, source: 'unavailable', bytes: null }),
    ).toBeNull();

    const flattened = flattenBenchmarkLifecycleRecord(record);
    expect(flattened.memory_delta_bytes).toBe(75);
    expect(flattened.cache_status).toBe('unknown');
    expect(flattened.initialization_and_retry_ms).toBe(300);

    const csv = benchmarkLifecycleRecordsToCsv([record]);
    expect(csv.startsWith(BENCHMARK_LIFECYCLE_CSV_COLUMNS.join(','))).toBe(true);
    expect(csv).toContain('model-load');
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
