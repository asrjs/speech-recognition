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
  readonly preprocessor_backend?: string;
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
  'encode_rtfx',
  'decode_rtfx',
  'preprocessor_backend',
  'error',
  'model_key',
  'backend',
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
    encode_rtfx: calcRtfx(run.audioDurationSec, metrics.encode_ms),
    decode_rtfx: calcRtfx(run.audioDurationSec, metrics.decode_ms),
    preprocessor_backend: metrics.preprocessor_backend,
    error: run.error || '',
    model_key: run.modelKey,
    backend: run.backend,
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
