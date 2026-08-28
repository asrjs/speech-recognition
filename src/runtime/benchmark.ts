import type { TranscriptMetrics } from '../types/index.js';

export interface NumericSummary {
  readonly count: number;
  readonly min: number | null;
  readonly max: number | null;
  readonly mean: number | null;
  readonly median: number | null;
  readonly p90: number | null;
  readonly stddev: number | null;
}

export interface BenchmarkStageMetrics {
  readonly preprocess_ms?: number;
  readonly encode_ms?: number;
  readonly decode_ms?: number;
  readonly tokenize_ms?: number;
  readonly total_ms?: number;
  readonly rtf?: number;
  /** Number of model inference windows used to compose this benchmark run. */
  readonly window_count?: number;
  readonly decoder_step_ms?: number;
  readonly decoder_step_count?: number;
  readonly decoder_step_avg_ms?: number;
  readonly decoder_gpu_tensor_downloads?: number;
  readonly decoder_kv_cache_location?: string;
  readonly encoder_run_ms?: number;
  readonly encoder_total_ms?: number;
  readonly encoder_frame_count?: number;
  readonly decode_iterations?: number;
  readonly preprocessor_backend?: string;
}

/**
 * Projects canonical transcript telemetry into the stable benchmark row shape.
 * Derived percentile fields are intentionally not represented because they
 * cannot be reconstructed from window-level totals.
 */
export function createBenchmarkStageMetrics(
  metrics: TranscriptMetrics | undefined,
): BenchmarkStageMetrics | undefined {
  if (!metrics) {
    return undefined;
  }
  return {
    preprocess_ms: metrics.preprocessMs,
    encode_ms: metrics.encodeMs,
    decode_ms: metrics.decodeMs,
    tokenize_ms: metrics.tokenizeMs,
    total_ms: metrics.totalMs,
    rtf: metrics.rtf,
    window_count: metrics.windowCount,
    decoder_step_ms: metrics.decoderStepMs,
    decoder_step_count: metrics.decoderStepCount,
    decoder_step_avg_ms: metrics.decoderStepAvgMs,
    decoder_gpu_tensor_downloads: metrics.decoderGpuTensorDownloads,
    decoder_kv_cache_location: metrics.decoderKvCacheLocation,
    encoder_run_ms: metrics.encoderRunMs,
    encoder_total_ms: metrics.encoderTotalMs,
    encoder_frame_count: metrics.encoderFrameCount,
    decode_iterations: metrics.decodeIterations,
    preprocessor_backend: metrics.preprocessorBackend,
  };
}

export type BenchmarkLifecyclePhase = 'model-load' | 'model-dispose';

export type BenchmarkCacheStatus = 'hit' | 'miss' | 'mixed' | 'disabled' | 'unknown';

export interface BenchmarkMemorySnapshot {
  readonly capturedAt: string;
  readonly source: 'measure-user-agent-specific-memory' | 'unavailable';
  readonly scope: 'process' | 'unavailable';
  readonly bytes: number | null;
  readonly reason?: 'unsupported' | 'measurement-failed' | 'invalid-result';
}

export interface BenchmarkLifecycleRecord {
  readonly id?: string;
  readonly phase: BenchmarkLifecyclePhase;
  readonly startedAt?: string;
  readonly finishedAt?: string;
  readonly totalMs?: number;
  /** Time from load start until the first runtime initialization attempt begins. */
  readonly initialArtifactResolutionMs?: number;
  /** Time after first artifact resolution through final initialization, including any retry path. */
  readonly initializationAndRetryMs?: number;
  readonly attemptCount?: number;
  readonly retryUsed?: boolean;
  readonly completedAssetCount?: number;
  readonly reportedAssetBytes?: number;
  readonly cacheStatus?: BenchmarkCacheStatus;
  readonly memoryBefore?: BenchmarkMemorySnapshot;
  readonly memoryAfter?: BenchmarkMemorySnapshot;
  readonly error?: string;
  readonly modelKey?: string;
  readonly backend?: string;
  readonly encoderBackend?: string;
  readonly decoderBackend?: string;
  readonly encoderQuant?: string;
  readonly decoderQuant?: string;
  readonly preprocessorBackend?: string;
}

export interface BenchmarkRunRecord {
  readonly batchId?: string;
  readonly startedAt?: string;
  readonly finishedAt?: string;
  readonly id?: string;
  readonly sampleKey?: string;
  readonly sampleOrder?: number;
  readonly rowIndex?: number;
  readonly repeatIndex?: number;
  readonly audioDurationSec?: number;
  readonly speaker?: string;
  readonly gender?: string;
  readonly speed?: number;
  readonly volume?: number;
  readonly transcription?: string;
  readonly referenceText?: string;
  readonly exactMatchToFirst?: boolean;
  readonly similarityToFirst?: number;
  readonly metrics?: BenchmarkStageMetrics;
  readonly error?: string;
  readonly modelKey?: string;
  readonly backend?: string;
  readonly encoderBackend?: string;
  readonly decoderBackend?: string;
  readonly encoderQuant?: string;
  readonly decoderQuant?: string;
  readonly preprocessor?: string;
  readonly preprocessorBackend?: string;
  readonly hardwareCpu?: string;
  readonly hardwareGpu?: string;
  readonly hardwareGpuModel?: string;
  readonly hardwareGpuCores?: number;
  readonly hardwareVram?: number;
  readonly hardwareMemory?: number;
  readonly hardwareWebgpu?: boolean;
}

export const BENCHMARK_RUN_CSV_COLUMNS = [
  'batch_id',
  'started_at',
  'finished_at',
  'run_id',
  'sample_key',
  'sample_order',
  'sample_row_index',
  'repeat_index',
  'audio_duration_sec',
  'speaker',
  'gender',
  'speed',
  'volume',
  'transcription',
  'reference_text',
  'exact_match_first',
  'similarity_first',
  'preprocess_ms',
  'encode_ms',
  'decode_ms',
  'tokenize_ms',
  'total_ms',
  'rtf',
  'window_count',
  'decoder_step_ms',
  'decoder_step_count',
  'decoder_step_avg_ms',
  'decoder_gpu_tensor_downloads',
  'decoder_kv_cache_location',
  'encoder_run_ms',
  'encoder_total_ms',
  'encoder_frame_count',
  'decode_iterations',
  'encode_rtfx',
  'decode_rtfx',
  'preprocessor_backend',
  'error',
  'model_key',
  'backend',
  'encoder_backend',
  'decoder_backend',
  'encoder_quant',
  'decoder_quant',
  'preprocessor',
  'preprocessor_backend_setting',
  'hardware_cpu',
  'hardware_gpu',
  'hardware_gpu_model',
  'hardware_gpu_cores',
  'hardware_vram',
  'hardware_memory',
  'hardware_webgpu',
] as const;

export const BENCHMARK_LIFECYCLE_CSV_COLUMNS = [
  'lifecycle_id',
  'phase',
  'started_at',
  'finished_at',
  'total_ms',
  'initial_artifact_resolution_ms',
  'initialization_and_retry_ms',
  'attempt_count',
  'retry_used',
  'completed_asset_count',
  'reported_asset_bytes',
  'cache_status',
  'memory_before_bytes',
  'memory_after_bytes',
  'memory_delta_bytes',
  'memory_source',
  'memory_scope',
  'memory_before_reason',
  'memory_after_reason',
  'error',
  'model_key',
  'backend',
  'encoder_backend',
  'decoder_backend',
  'encoder_quant',
  'decoder_quant',
  'preprocessor_backend_setting',
] as const;

export function normalizeBenchmarkText(value: string | null | undefined): string {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function levenshteinDistance(
  left: string | null | undefined,
  right: string | null | undefined,
): number {
  const a = left || '';
  const b = right || '';
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;

  const previous = new Array<number>(b.length + 1);
  const current = new Array<number>(b.length + 1);

  for (let index = 0; index <= b.length; index += 1) {
    previous[index] = index;
  }

  for (let row = 1; row <= a.length; row += 1) {
    current[0] = row;
    for (let column = 1; column <= b.length; column += 1) {
      const substitutionCost = a[row - 1] === b[column - 1] ? 0 : 1;
      current[column] = Math.min(
        previous[column]! + 1,
        current[column - 1]! + 1,
        previous[column - 1]! + substitutionCost,
      );
    }
    for (let column = 0; column <= b.length; column += 1) {
      previous[column] = current[column]!;
    }
  }

  return previous[b.length]!;
}

export function textSimilarity(
  left: string | null | undefined,
  right: string | null | undefined,
): number {
  const normalizedLeft = normalizeBenchmarkText(left);
  const normalizedRight = normalizeBenchmarkText(right);
  const maxLength = Math.max(normalizedLeft.length, normalizedRight.length);
  if (maxLength === 0) {
    return 1;
  }
  return 1 - levenshteinDistance(normalizedLeft, normalizedRight) / maxLength;
}

export function mean(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

export function percentile(values: readonly number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index]!;
}

export function stddev(values: readonly number[]): number {
  if (values.length < 2) return 0;
  const avg = mean(values)!;
  const variance = values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

export function summarizeNumericSeries(values: readonly number[]): NumericSummary {
  const numeric = values.filter(Number.isFinite);
  if (numeric.length === 0) {
    return {
      count: 0,
      min: null,
      max: null,
      mean: null,
      median: null,
      p90: null,
      stddev: null,
    };
  }

  return {
    count: numeric.length,
    min: Math.min(...numeric),
    max: Math.max(...numeric),
    mean: mean(numeric),
    median: median(numeric),
    p90: percentile(numeric, 90),
    stddev: stddev(numeric),
  };
}

export function safeNumber(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function calcRtfx(
  audioDurationSec: number | null | undefined,
  stageMs: number | null | undefined,
): number | null {
  const duration = Number(audioDurationSec);
  const latencyMs = Number(stageMs);
  if (
    !Number.isFinite(duration) ||
    !Number.isFinite(latencyMs) ||
    duration <= 0 ||
    latencyMs <= 0
  ) {
    return null;
  }
  return (duration * 1000) / latencyMs;
}

function escapeCsv(value: unknown): string {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function toCsv(
  rows: readonly Record<string, unknown>[],
  columns: readonly string[],
): string {
  const header = columns.join(',');
  const lines = rows.map((row) => columns.map((column) => escapeCsv(row[column])).join(','));
  return [header, ...lines].join('\n');
}

export function benchmarkMemoryDeltaBytes(
  before: BenchmarkMemorySnapshot | null | undefined,
  after: BenchmarkMemorySnapshot | null | undefined,
): number | null {
  if (
    before?.bytes === null ||
    before?.bytes === undefined ||
    after?.bytes === null ||
    after?.bytes === undefined ||
    before.source !== after.source ||
    before.scope !== after.scope
  ) {
    return null;
  }
  return after.bytes - before.bytes;
}

export function flattenBenchmarkLifecycleRecord(
  record: BenchmarkLifecycleRecord,
): Record<string, unknown> {
  return {
    lifecycle_id: record.id,
    phase: record.phase,
    started_at: record.startedAt,
    finished_at: record.finishedAt,
    total_ms: record.totalMs,
    initial_artifact_resolution_ms: record.initialArtifactResolutionMs,
    initialization_and_retry_ms: record.initializationAndRetryMs,
    attempt_count: record.attemptCount,
    retry_used: record.retryUsed,
    completed_asset_count: record.completedAssetCount,
    reported_asset_bytes: record.reportedAssetBytes,
    cache_status: record.cacheStatus,
    memory_before_bytes: record.memoryBefore?.bytes,
    memory_after_bytes: record.memoryAfter?.bytes,
    memory_delta_bytes: benchmarkMemoryDeltaBytes(record.memoryBefore, record.memoryAfter),
    memory_source: record.memoryAfter?.source ?? record.memoryBefore?.source,
    memory_scope: record.memoryAfter?.scope ?? record.memoryBefore?.scope,
    memory_before_reason: record.memoryBefore?.reason,
    memory_after_reason: record.memoryAfter?.reason,
    error: record.error || '',
    model_key: record.modelKey,
    backend: record.backend,
    encoder_backend: record.encoderBackend,
    decoder_backend: record.decoderBackend,
    encoder_quant: record.encoderQuant,
    decoder_quant: record.decoderQuant,
    preprocessor_backend_setting: record.preprocessorBackend,
  };
}

export function benchmarkLifecycleRecordsToCsv(
  records: readonly BenchmarkLifecycleRecord[],
): string {
  return toCsv(
    records.map((record) => flattenBenchmarkLifecycleRecord(record)),
    [...BENCHMARK_LIFECYCLE_CSV_COLUMNS],
  );
}

export function flattenBenchmarkRunRecord(run: BenchmarkRunRecord): Record<string, unknown> {
  const metrics = run.metrics || {};
  return {
    batch_id: run.batchId,
    started_at: run.startedAt,
    finished_at: run.finishedAt,
    run_id: run.id,
    sample_key: run.sampleKey,
    sample_order: run.sampleOrder,
    sample_row_index: run.rowIndex,
    repeat_index: run.repeatIndex,
    audio_duration_sec: run.audioDurationSec,
    speaker: run.speaker,
    gender: run.gender,
    speed: run.speed,
    volume: run.volume,
    transcription: run.transcription,
    reference_text: run.referenceText,
    exact_match_first: run.exactMatchToFirst,
    similarity_first: run.similarityToFirst,
    preprocess_ms: metrics.preprocess_ms,
    encode_ms: metrics.encode_ms,
    decode_ms: metrics.decode_ms,
    tokenize_ms: metrics.tokenize_ms,
    total_ms: metrics.total_ms,
    rtf: metrics.rtf,
    window_count: metrics.window_count,
    decoder_step_ms: metrics.decoder_step_ms,
    decoder_step_count: metrics.decoder_step_count,
    decoder_step_avg_ms: metrics.decoder_step_avg_ms,
    decoder_gpu_tensor_downloads: metrics.decoder_gpu_tensor_downloads,
    decoder_kv_cache_location: metrics.decoder_kv_cache_location,
    encoder_run_ms: metrics.encoder_run_ms,
    encoder_total_ms: metrics.encoder_total_ms,
    encoder_frame_count: metrics.encoder_frame_count,
    decode_iterations: metrics.decode_iterations,
    encode_rtfx: calcRtfx(run.audioDurationSec, metrics.encode_ms),
    decode_rtfx: calcRtfx(run.audioDurationSec, metrics.decode_ms),
    preprocessor_backend: metrics.preprocessor_backend,
    error: run.error || '',
    model_key: run.modelKey,
    backend: run.backend,
    encoder_backend: run.encoderBackend,
    decoder_backend: run.decoderBackend,
    encoder_quant: run.encoderQuant,
    decoder_quant: run.decoderQuant,
    preprocessor: run.preprocessor,
    preprocessor_backend_setting: run.preprocessorBackend,
    hardware_cpu: run.hardwareCpu,
    hardware_gpu: run.hardwareGpu,
    hardware_gpu_model: run.hardwareGpuModel,
    hardware_gpu_cores: run.hardwareGpuCores,
    hardware_vram: run.hardwareVram,
    hardware_memory: run.hardwareMemory,
    hardware_webgpu: run.hardwareWebgpu,
  };
}

export function benchmarkRunRecordsToCsv(runs: readonly BenchmarkRunRecord[]): string {
  return toCsv(
    runs.map((run) => flattenBenchmarkRunRecord(run)),
    [...BENCHMARK_RUN_CSV_COLUMNS],
  );
}
