import type { TranscriptNormalizationContext, TranscriptResult, TranscriptWarning, TranscriptWord, TranscriptToken, TranscriptMetrics } from '../../types/index.js';
import type { LegacyParakeetTranscript, LegacyParakeetMetrics } from './compat.js';

export function mapLegacyParakeetMetricsToCanonical(
  metrics?: LegacyParakeetMetrics
): TranscriptMetrics | undefined {
  if (!metrics) return undefined;
  return {
    preprocessMs: metrics.preprocess_ms,
    encodeMs: metrics.encode_ms,
    decodeMs: metrics.decode_ms,
    tokenizeMs: metrics.tokenize_ms,
    postprocessMs: metrics.tokenize_ms,
    totalMs: metrics.total_ms,
    wallMs: metrics.wall_ms,
    audioDurationSec: metrics.audio_duration_sec,
    rtf: metrics.rtf,
    rtfx: metrics.rtfx,
    preprocessorBackend: metrics.preprocessor_backend,
    decodeAudioMs: metrics.audio_decode_ms,
    downmixMs: metrics.downmix_ms,
    resampleMs: metrics.resample_ms,
    audioPreparationMs: metrics.audio_preparation_ms,
    inputSampleRate: metrics.input_sample_rate,
    outputSampleRate: metrics.output_sample_rate,
    resampler: metrics.resampler,
    resamplerQuality: metrics.resampler_quality,
    encoderFrameCount: metrics.encoder_frame_count,
    decodeIterations: metrics.decode_iterations,
    emittedTokenCount: metrics.emitted_token_count,
    emittedWordCount: metrics.emitted_word_count,
  };
}

export function mapLegacyParakeetNativeToCanonical(
  native: LegacyParakeetTranscript,
  context: TranscriptNormalizationContext = {}
): TranscriptResult {
  const detail = context.detailLevel ?? 'segments';
  const words: TranscriptWord[] = (native.words ?? []).map((word, index) => ({
    index,
    text: word.text,
    startTime: word.start_time,
    endTime: word.end_time,
    confidence: word.confidence,
  }));

  const tokens: TranscriptToken[] = (native.tokens ?? []).map((token, index) => ({
    index,
    id: token.id,
    text: token.token,
    rawText: token.raw_text,
    startTime: token.start_time,
    endTime: token.end_time,
    confidence: token.confidence,
    frameIndex: token.frame_index,
    logProb: token.log_prob,
    tdtStep: token.tdt_step,
  }));

  const segments = words.length > 0
    ? [
        {
          index: 0,
          text: native.utterance_text,
          startTime: words[0]!.startTime,
          endTime: words[words.length - 1]!.endTime,
          confidence: native.confidence_scores?.utterance ?? undefined,
          wordIndices: words.map((word) => word.index),
        },
      ]
    : undefined;

  const result: TranscriptResult = {
    text: native.utterance_text,
    warnings: [] satisfies readonly TranscriptWarning[],
    meta: {
      ...context,
      detailLevel: detail,
      isFinal: native.is_final,
      modelFamily: context.modelFamily ?? 'parakeet',
      tokenCount: tokens.length || undefined,
      wordCount: words.length || undefined,
      segmentCount: segments?.length,
      averageConfidence: native.confidence_scores?.utterance ?? undefined,
      averageWordConfidence: native.confidence_scores?.word_avg ?? undefined,
      averageTokenConfidence: native.confidence_scores?.token_avg ?? undefined,
      nativeAvailable: true,
      metrics: native.metrics
        ? mapLegacyParakeetMetricsToCanonical(native.metrics)
        : context.metrics,
    },
  };

  if (detail !== 'text' && segments) {
    Object.assign(result, { segments });
  }
  if ((detail === 'words' || detail === 'detailed') && words.length > 0) {
    Object.assign(result, { words });
  }
  if (detail === 'detailed' && tokens.length > 0) {
    Object.assign(result, { tokens });
  }

  return result;
}
