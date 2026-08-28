import type { TranscriptMetrics } from '../types/index.js';

export interface WindowedMetricsAccumulator {
  readonly audioDurationSec: number;
  windowCount: number;
  preprocessMs: number;
  encodeMs: number;
  decodeMs: number;
  tokenizeMs: number;
  postprocessMs: number;
  totalMs: number;
  wallMs: number;
  emittedTokenCount: number;
  emittedWordCount: number;
  hasMetrics: boolean;
}

export function createWindowedMetricsAccumulator(audioDurationSec: number): WindowedMetricsAccumulator {
  return {
    audioDurationSec,
    windowCount: 0,
    preprocessMs: 0,
    encodeMs: 0,
    decodeMs: 0,
    tokenizeMs: 0,
    postprocessMs: 0,
    totalMs: 0,
    wallMs: 0,
    emittedTokenCount: 0,
    emittedWordCount: 0,
    hasMetrics: false,
  };
}

function addMetricValue(
  accumulator: WindowedMetricsAccumulator,
  key: keyof Omit<WindowedMetricsAccumulator, 'audioDurationSec' | 'hasMetrics' | 'windowCount'>,
  value: number | undefined,
): void {
  if (value !== undefined && Number.isFinite(value)) {
    accumulator[key] += value;
    accumulator.hasMetrics = true;
  }
}

export function addWindowMetrics(
  accumulator: WindowedMetricsAccumulator,
  metrics: TranscriptMetrics | undefined,
): void {
  if (!metrics) {
    return;
  }
  addMetricValue(accumulator, 'preprocessMs', metrics.preprocessMs);
  addMetricValue(accumulator, 'encodeMs', metrics.encodeMs);
  addMetricValue(accumulator, 'decodeMs', metrics.decodeMs);
  addMetricValue(accumulator, 'tokenizeMs', metrics.tokenizeMs);
  addMetricValue(accumulator, 'postprocessMs', metrics.postprocessMs);
  addMetricValue(accumulator, 'totalMs', metrics.totalMs);
  addMetricValue(accumulator, 'wallMs', metrics.wallMs);
  addMetricValue(accumulator, 'emittedTokenCount', metrics.emittedTokenCount);
  addMetricValue(accumulator, 'emittedWordCount', metrics.emittedWordCount);
}

export function buildWindowedMetrics(
  accumulator: WindowedMetricsAccumulator,
): TranscriptMetrics | undefined {
  if (!accumulator.hasMetrics && accumulator.windowCount === 0) {
    return undefined;
  }

  const totalMs = accumulator.totalMs || accumulator.wallMs;
  const rtf = totalMs > 0 ? totalMs / 1000 / accumulator.audioDurationSec : undefined;
  return {
    preprocessMs: accumulator.preprocessMs || undefined,
    encodeMs: accumulator.encodeMs || undefined,
    decodeMs: accumulator.decodeMs || undefined,
    tokenizeMs: accumulator.tokenizeMs || undefined,
    postprocessMs: accumulator.postprocessMs || undefined,
    totalMs: totalMs || undefined,
    wallMs: accumulator.wallMs || undefined,
    audioDurationSec: accumulator.audioDurationSec,
    windowCount: accumulator.windowCount,
    rtf,
    rtfx: rtf && rtf > 0 ? 1 / rtf : undefined,
    emittedTokenCount: accumulator.emittedTokenCount || undefined,
    emittedWordCount: accumulator.emittedWordCount || undefined,
  };
}
