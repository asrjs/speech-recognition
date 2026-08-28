import type { TranscriptMetrics } from '../types/index.js';

export interface WindowedMetricsAccumulator {
  readonly audioDurationSec: number;
  windowCount: number;
  preprocessMs: number;
  encodeMs: number;
  decodeMs: number;
  tokenizeMs: number;
  postprocessMs: number;
  languageDetectionMs: number;
  decoderInitMs: number;
  decoderInitInputMs: number;
  decoderInitRunMs: number;
  decoderInitOutputMs: number;
  decoderStepMs: number;
  decoderStepFeedBuildMs: number;
  decoderStepTensorCloneMs: number;
  decoderStepRunMs: number;
  decoderStepOutputMs: number;
  decoderLogitProcessMs: number;
  decoderStepCount: number;
  decoderGpuTensorInputs: number;
  decoderCpuTensorInputs: number;
  decoderGpuTensorOutputs: number;
  decoderCpuTensorOutputs: number;
  decoderGpuTensorDownloads: number;
  decoderInitTensorCreateMs: number;
  decoderInitLogitReadMs: number;
  decoderInitKvExtractMs: number;
  decoderStepTensorCreateMs: number;
  decoderStepLogitReadMs: number;
  decoderStepKvMergeMs: number;
  decoderEncoderKvTensorReuses: number;
  decoderEncoderKvTensorCreates: number;
  sessionCreateMs: number;
  encoderRunMs: number;
  encoderOutputMs: number;
  encoderOutputCastMs: number;
  encoderBufferRewrapMs: number;
  encoderGpuFlushMs: number;
  encoderGpuDrainMs: number;
  encoderTotalMs: number;
  wordAlignmentReferenceMs: number;
  decodeAudioMs: number;
  downmixMs: number;
  resampleMs: number;
  audioPreparationMs: number;
  encoderFrameCount: number;
  decodeIterations: number;
  totalMs: number;
  wallMs: number;
  emittedTokenCount: number;
  emittedWordCount: number;
  decoderKvCacheLocation?: string;
  encoderOutputLocation?: string;
  encoderOutputDtype?: string;
  requestedPreprocessorBackend?: string;
  preprocessorBackend?: string;
  wordAlignmentSource?: string;
  resampler?: string;
  resamplerQuality?: string;
  hasMetrics: boolean;
}

export function createWindowedMetricsAccumulator(
  audioDurationSec: number,
): WindowedMetricsAccumulator {
  return {
    audioDurationSec,
    windowCount: 0,
    preprocessMs: 0,
    encodeMs: 0,
    decodeMs: 0,
    tokenizeMs: 0,
    postprocessMs: 0,
    languageDetectionMs: 0,
    decoderInitMs: 0,
    decoderInitInputMs: 0,
    decoderInitRunMs: 0,
    decoderInitOutputMs: 0,
    decoderStepMs: 0,
    decoderStepFeedBuildMs: 0,
    decoderStepTensorCloneMs: 0,
    decoderStepRunMs: 0,
    decoderStepOutputMs: 0,
    decoderLogitProcessMs: 0,
    decoderStepCount: 0,
    decoderGpuTensorInputs: 0,
    decoderCpuTensorInputs: 0,
    decoderGpuTensorOutputs: 0,
    decoderCpuTensorOutputs: 0,
    decoderGpuTensorDownloads: 0,
    decoderInitTensorCreateMs: 0,
    decoderInitLogitReadMs: 0,
    decoderInitKvExtractMs: 0,
    decoderStepTensorCreateMs: 0,
    decoderStepLogitReadMs: 0,
    decoderStepKvMergeMs: 0,
    decoderEncoderKvTensorReuses: 0,
    decoderEncoderKvTensorCreates: 0,
    sessionCreateMs: 0,
    encoderRunMs: 0,
    encoderOutputMs: 0,
    encoderOutputCastMs: 0,
    encoderBufferRewrapMs: 0,
    encoderGpuFlushMs: 0,
    encoderGpuDrainMs: 0,
    encoderTotalMs: 0,
    wordAlignmentReferenceMs: 0,
    decodeAudioMs: 0,
    downmixMs: 0,
    resampleMs: 0,
    audioPreparationMs: 0,
    encoderFrameCount: 0,
    decodeIterations: 0,
    totalMs: 0,
    wallMs: 0,
    emittedTokenCount: 0,
    emittedWordCount: 0,
    hasMetrics: false,
  };
}

function addMetricValue(
  accumulator: WindowedMetricsAccumulator,
  key: WindowedNumericMetricKey,
  value: number | undefined,
): void {
  if (value !== undefined && Number.isFinite(value)) {
    accumulator[key] += value;
    accumulator.hasMetrics = true;
  }
}

type WindowedNumericMetricKey =
  | 'preprocessMs'
  | 'encodeMs'
  | 'decodeMs'
  | 'tokenizeMs'
  | 'postprocessMs'
  | 'languageDetectionMs'
  | 'decoderInitMs'
  | 'decoderInitInputMs'
  | 'decoderInitRunMs'
  | 'decoderInitOutputMs'
  | 'decoderStepMs'
  | 'decoderStepFeedBuildMs'
  | 'decoderStepTensorCloneMs'
  | 'decoderStepRunMs'
  | 'decoderStepOutputMs'
  | 'decoderLogitProcessMs'
  | 'decoderStepCount'
  | 'decoderGpuTensorInputs'
  | 'decoderCpuTensorInputs'
  | 'decoderGpuTensorOutputs'
  | 'decoderCpuTensorOutputs'
  | 'decoderGpuTensorDownloads'
  | 'decoderInitTensorCreateMs'
  | 'decoderInitLogitReadMs'
  | 'decoderInitKvExtractMs'
  | 'decoderStepTensorCreateMs'
  | 'decoderStepLogitReadMs'
  | 'decoderStepKvMergeMs'
  | 'decoderEncoderKvTensorReuses'
  | 'decoderEncoderKvTensorCreates'
  | 'sessionCreateMs'
  | 'encoderRunMs'
  | 'encoderOutputMs'
  | 'encoderOutputCastMs'
  | 'encoderBufferRewrapMs'
  | 'encoderGpuFlushMs'
  | 'encoderGpuDrainMs'
  | 'encoderTotalMs'
  | 'wordAlignmentReferenceMs'
  | 'decodeAudioMs'
  | 'downmixMs'
  | 'resampleMs'
  | 'audioPreparationMs'
  | 'encoderFrameCount'
  | 'decodeIterations'
  | 'totalMs'
  | 'wallMs'
  | 'emittedTokenCount'
  | 'emittedWordCount';

function addStringMetric(
  accumulator: WindowedMetricsAccumulator,
  key: keyof Pick<
    WindowedMetricsAccumulator,
    | 'decoderKvCacheLocation'
    | 'encoderOutputLocation'
    | 'encoderOutputDtype'
    | 'requestedPreprocessorBackend'
    | 'preprocessorBackend'
    | 'wordAlignmentSource'
    | 'resampler'
    | 'resamplerQuality'
  >,
  value: string | null | undefined,
): void {
  if (typeof value !== 'string' || value.length === 0) {
    return;
  }
  const previous = accumulator[key];
  accumulator[key] = previous === undefined || previous === value ? value : 'mixed';
  accumulator.hasMetrics = true;
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
  addMetricValue(accumulator, 'languageDetectionMs', metrics.languageDetectionMs);
  addMetricValue(accumulator, 'decoderInitMs', metrics.decoderInitMs);
  addMetricValue(accumulator, 'decoderInitInputMs', metrics.decoderInitInputMs);
  addMetricValue(accumulator, 'decoderInitRunMs', metrics.decoderInitRunMs);
  addMetricValue(accumulator, 'decoderInitOutputMs', metrics.decoderInitOutputMs);
  addMetricValue(accumulator, 'decoderStepMs', metrics.decoderStepMs);
  addMetricValue(accumulator, 'decoderStepFeedBuildMs', metrics.decoderStepFeedBuildMs);
  addMetricValue(accumulator, 'decoderStepTensorCloneMs', metrics.decoderStepTensorCloneMs);
  addMetricValue(accumulator, 'decoderStepRunMs', metrics.decoderStepRunMs);
  addMetricValue(accumulator, 'decoderStepOutputMs', metrics.decoderStepOutputMs);
  addMetricValue(accumulator, 'decoderLogitProcessMs', metrics.decoderLogitProcessMs);
  addMetricValue(accumulator, 'decoderStepCount', metrics.decoderStepCount);
  addMetricValue(accumulator, 'decoderGpuTensorInputs', metrics.decoderGpuTensorInputs);
  addMetricValue(accumulator, 'decoderCpuTensorInputs', metrics.decoderCpuTensorInputs);
  addMetricValue(accumulator, 'decoderGpuTensorOutputs', metrics.decoderGpuTensorOutputs);
  addMetricValue(accumulator, 'decoderCpuTensorOutputs', metrics.decoderCpuTensorOutputs);
  addMetricValue(accumulator, 'decoderGpuTensorDownloads', metrics.decoderGpuTensorDownloads);
  addMetricValue(accumulator, 'decoderInitTensorCreateMs', metrics.decoderInitTensorCreateMs);
  addMetricValue(accumulator, 'decoderInitLogitReadMs', metrics.decoderInitLogitReadMs);
  addMetricValue(accumulator, 'decoderInitKvExtractMs', metrics.decoderInitKvExtractMs);
  addMetricValue(accumulator, 'decoderStepTensorCreateMs', metrics.decoderStepTensorCreateMs);
  addMetricValue(accumulator, 'decoderStepLogitReadMs', metrics.decoderStepLogitReadMs);
  addMetricValue(accumulator, 'decoderStepKvMergeMs', metrics.decoderStepKvMergeMs);
  addMetricValue(accumulator, 'decoderEncoderKvTensorReuses', metrics.decoderEncoderKvTensorReuses);
  addMetricValue(
    accumulator,
    'decoderEncoderKvTensorCreates',
    metrics.decoderEncoderKvTensorCreates,
  );
  addMetricValue(accumulator, 'sessionCreateMs', metrics.sessionCreateMs);
  addMetricValue(accumulator, 'encoderRunMs', metrics.encoderRunMs);
  addMetricValue(accumulator, 'encoderOutputMs', metrics.encoderOutputMs);
  addMetricValue(accumulator, 'encoderOutputCastMs', metrics.encoderOutputCastMs);
  addMetricValue(accumulator, 'encoderBufferRewrapMs', metrics.encoderBufferRewrapMs);
  addMetricValue(accumulator, 'encoderGpuFlushMs', metrics.encoderGpuFlushMs);
  addMetricValue(accumulator, 'encoderGpuDrainMs', metrics.encoderGpuDrainMs);
  addMetricValue(accumulator, 'encoderTotalMs', metrics.encoderTotalMs);
  addMetricValue(accumulator, 'wordAlignmentReferenceMs', metrics.wordAlignmentReferenceMs);
  addMetricValue(accumulator, 'decodeAudioMs', metrics.decodeAudioMs);
  addMetricValue(accumulator, 'downmixMs', metrics.downmixMs);
  addMetricValue(accumulator, 'resampleMs', metrics.resampleMs);
  addMetricValue(accumulator, 'audioPreparationMs', metrics.audioPreparationMs);
  addMetricValue(accumulator, 'encoderFrameCount', metrics.encoderFrameCount);
  addMetricValue(accumulator, 'decodeIterations', metrics.decodeIterations);
  addMetricValue(accumulator, 'totalMs', metrics.totalMs);
  addMetricValue(accumulator, 'wallMs', metrics.wallMs);
  addMetricValue(accumulator, 'emittedTokenCount', metrics.emittedTokenCount);
  addMetricValue(accumulator, 'emittedWordCount', metrics.emittedWordCount);
  addStringMetric(accumulator, 'decoderKvCacheLocation', metrics.decoderKvCacheLocation);
  addStringMetric(accumulator, 'encoderOutputLocation', metrics.encoderOutputLocation);
  addStringMetric(accumulator, 'encoderOutputDtype', metrics.encoderOutputDtype);
  addStringMetric(
    accumulator,
    'requestedPreprocessorBackend',
    metrics.requestedPreprocessorBackend,
  );
  addStringMetric(accumulator, 'preprocessorBackend', metrics.preprocessorBackend);
  addStringMetric(accumulator, 'wordAlignmentSource', metrics.wordAlignmentSource);
  addStringMetric(accumulator, 'resampler', metrics.resampler);
  addStringMetric(accumulator, 'resamplerQuality', metrics.resamplerQuality);
}

export function buildWindowedMetrics(
  accumulator: WindowedMetricsAccumulator,
): TranscriptMetrics | undefined {
  if (!accumulator.hasMetrics && accumulator.windowCount === 0) {
    return undefined;
  }

  const totalMs = accumulator.totalMs || accumulator.wallMs;
  const rtf = totalMs > 0 ? totalMs / 1000 / accumulator.audioDurationSec : undefined;
  const decoderStepAvgMs =
    accumulator.decoderStepMs > 0 && accumulator.decoderStepCount > 0
      ? accumulator.decoderStepMs / accumulator.decoderStepCount
      : undefined;
  return {
    preprocessMs: accumulator.preprocessMs || undefined,
    encodeMs: accumulator.encodeMs || undefined,
    decodeMs: accumulator.decodeMs || undefined,
    tokenizeMs: accumulator.tokenizeMs || undefined,
    postprocessMs: accumulator.postprocessMs || undefined,
    languageDetectionMs: accumulator.languageDetectionMs || undefined,
    decoderInitMs: accumulator.decoderInitMs || undefined,
    decoderInitInputMs: accumulator.decoderInitInputMs || undefined,
    decoderInitRunMs: accumulator.decoderInitRunMs || undefined,
    decoderInitOutputMs: accumulator.decoderInitOutputMs || undefined,
    decoderStepMs: accumulator.decoderStepMs || undefined,
    decoderStepFeedBuildMs: accumulator.decoderStepFeedBuildMs || undefined,
    decoderStepTensorCloneMs: accumulator.decoderStepTensorCloneMs || undefined,
    decoderStepRunMs: accumulator.decoderStepRunMs || undefined,
    decoderStepOutputMs: accumulator.decoderStepOutputMs || undefined,
    decoderStepAvgMs,
    decoderLogitProcessMs: accumulator.decoderLogitProcessMs || undefined,
    decoderStepCount: accumulator.decoderStepCount || undefined,
    decoderGpuTensorInputs: accumulator.decoderGpuTensorInputs || undefined,
    decoderCpuTensorInputs: accumulator.decoderCpuTensorInputs || undefined,
    decoderGpuTensorOutputs: accumulator.decoderGpuTensorOutputs || undefined,
    decoderCpuTensorOutputs: accumulator.decoderCpuTensorOutputs || undefined,
    decoderGpuTensorDownloads: accumulator.decoderGpuTensorDownloads || undefined,
    decoderKvCacheLocation: accumulator.decoderKvCacheLocation,
    decoderInitTensorCreateMs: accumulator.decoderInitTensorCreateMs || undefined,
    decoderInitLogitReadMs: accumulator.decoderInitLogitReadMs || undefined,
    decoderInitKvExtractMs: accumulator.decoderInitKvExtractMs || undefined,
    decoderStepTensorCreateMs: accumulator.decoderStepTensorCreateMs || undefined,
    decoderStepLogitReadMs: accumulator.decoderStepLogitReadMs || undefined,
    decoderStepKvMergeMs: accumulator.decoderStepKvMergeMs || undefined,
    decoderEncoderKvTensorReuses: accumulator.decoderEncoderKvTensorReuses || undefined,
    decoderEncoderKvTensorCreates: accumulator.decoderEncoderKvTensorCreates || undefined,
    sessionCreateMs: accumulator.sessionCreateMs || undefined,
    encoderRunMs: accumulator.encoderRunMs || undefined,
    encoderOutputMs: accumulator.encoderOutputMs || undefined,
    encoderOutputCastMs: accumulator.encoderOutputCastMs || undefined,
    encoderOutputLocation: accumulator.encoderOutputLocation,
    encoderOutputDtype: accumulator.encoderOutputDtype,
    encoderBufferRewrapMs: accumulator.encoderBufferRewrapMs || undefined,
    encoderGpuFlushMs: accumulator.encoderGpuFlushMs || undefined,
    encoderGpuDrainMs: accumulator.encoderGpuDrainMs || undefined,
    encoderTotalMs: accumulator.encoderTotalMs || undefined,
    wordAlignmentReferenceMs: accumulator.wordAlignmentReferenceMs || undefined,
    wordAlignmentSource: accumulator.wordAlignmentSource,
    totalMs: totalMs || undefined,
    wallMs: accumulator.wallMs || undefined,
    audioDurationSec: accumulator.audioDurationSec,
    windowCount: accumulator.windowCount,
    rtf,
    rtfx: rtf && rtf > 0 ? 1 / rtf : undefined,
    requestedPreprocessorBackend: accumulator.requestedPreprocessorBackend,
    preprocessorBackend: accumulator.preprocessorBackend,
    decodeAudioMs: accumulator.decodeAudioMs || undefined,
    downmixMs: accumulator.downmixMs || undefined,
    resampleMs: accumulator.resampleMs || undefined,
    audioPreparationMs: accumulator.audioPreparationMs || undefined,
    resampler: accumulator.resampler,
    resamplerQuality: accumulator.resamplerQuality,
    encoderFrameCount: accumulator.encoderFrameCount || undefined,
    decodeIterations: accumulator.decodeIterations || undefined,
    emittedTokenCount: accumulator.emittedTokenCount || undefined,
    emittedWordCount: accumulator.emittedWordCount || undefined,
  };
}
