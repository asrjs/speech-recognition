import type {
  ModelClassification,
  TranscriptNormalizationContext,
  TranscriptNormalizer,
  TranscriptResult,
  TranscriptWarning,
  TranscriptionEnvelope,
} from '../types/index.js';
import {
  createModelClassification,
  defaultNemoConfidenceReconstructor,
  defaultNemoTimestampReconstructor,
  mapNemoNativeToCanonical,
  type NemoTimestampReconstructor,
} from '../models/nemo-common/index.js';
import {
  DEFAULT_LASR_CTC_CLASSIFICATION,
  type LasrCtcNativeTranscript,
  mapLasrCtcNativeToCanonical,
} from '../models/lasr-ctc/index.js';
import {
  DEFAULT_NEMO_AED_CLASSIFICATION,
  type NemoAedNativeTranscript,
} from '../models/nemo-aed/index.js';
import {
  DEFAULT_NEMO_RNNT_CLASSIFICATION,
  type NemoRnntNativeTranscript,
} from '../models/nemo-rnnt/index.js';
import {
  DEFAULT_NEMO_TDT_CLASSIFICATION,
  type NemoTdtNativeTranscript,
} from '../models/nemo-tdt/index.js';
import {
  DEFAULT_WHISPER_CLASSIFICATION,
  type WhisperNativeTranscript,
  mapWhisperNativeToCanonical,
} from '../models/whisper-seq2seq/index.js';
import type { LegacyParakeetTranscript } from '../presets/parakeet/index.js';

function withTranscriptDefaults(
  context: TranscriptNormalizationContext = {},
): Required<Pick<TranscriptNormalizationContext, 'detailLevel'>> & TranscriptNormalizationContext {
  return {
    detailLevel: context.detailLevel ?? 'segments',
    ...context,
  };
}

function createTranscriptNormalizer<TNative>(
  id: string,
  map: (native: TNative, context: TranscriptNormalizationContext) => TranscriptResult,
): TranscriptNormalizer<TNative> {
  return {
    id,
    toCanonical(native, context = {}) {
      return map(native, withTranscriptDefaults(context));
    },
    toEnvelope(native, context = {}) {
      return {
        canonical: map(native, withTranscriptDefaults(context)),
        native,
      };
    },
  };
}

function normalizeClassification(
  base: ModelClassification,
  override: Partial<ModelClassification> = {},
): ModelClassification {
  return createModelClassification(base, override);
}

const nemoAedTimestampReconstructor: NemoTimestampReconstructor<NemoAedNativeTranscript> = {
  reconstruct(nativeTranscript, detail) {
    const defaultReconstructed = defaultNemoTimestampReconstructor.reconstruct(
      nativeTranscript,
      detail,
    );
    if (
      defaultReconstructed.segments?.length ||
      defaultReconstructed.words?.length ||
      detail === 'text'
    ) {
      return defaultReconstructed;
    }

    const duration = nativeTranscript.metrics?.audioDurationSec ?? 0;
    const segments =
      nativeTranscript.utteranceText.length > 0
        ? [
            {
              index: 0,
              text: nativeTranscript.utteranceText,
              startTime: 0,
              endTime: duration,
              confidence: nativeTranscript.confidence?.utterance,
            },
          ]
        : [];

    if (detail === 'segments' || detail === 'words') {
      return { segments };
    }

    return {
      segments,
      tokens: (nativeTranscript.tokens ?? []).map((token) => ({
        index: token.index,
        id: token.id,
        text: token.text,
        rawText: token.rawText,
        isWordStart: token.isWordStart,
        confidence: token.confidence,
        logProb: token.logProb,
      })),
    };
  },
};

export function createNemoTdtTranscriptNormalizer(
  classification: Partial<ModelClassification> = {},
): TranscriptNormalizer<NemoTdtNativeTranscript> {
  const normalizedClassification = normalizeClassification(
    DEFAULT_NEMO_TDT_CLASSIFICATION,
    classification,
  );
  return createTranscriptNormalizer('nemo-tdt', (native, context) =>
    mapNemoNativeToCanonical(
      native,
      normalizedClassification,
      {
        ...context,
        detailLevel: context.detailLevel ?? 'segments',
        metrics: native.metrics
          ? {
              preprocessMs: native.metrics.preprocessMs,
              encodeMs: native.metrics.encodeMs,
              decodeMs: native.metrics.decodeMs,
              tokenizeMs: native.metrics.tokenizeMs,
              postprocessMs: native.metrics.tokenizeMs,
              totalMs: native.metrics.totalMs,
              wallMs: native.metrics.wallMs,
              audioDurationSec: native.metrics.audioDurationSec,
              rtf: native.metrics.rtf,
              rtfx: native.metrics.rtfx,
              requestedPreprocessorBackend: native.metrics.requestedPreprocessorBackend,
              preprocessorBackend: native.metrics.preprocessorBackend,
              decodeAudioMs: native.metrics.decodeAudioMs,
              downmixMs: native.metrics.downmixMs,
              resampleMs: native.metrics.resampleMs,
              audioPreparationMs: native.metrics.audioPreparationMs,
              inputSampleRate: native.metrics.inputSampleRate,
              outputSampleRate: native.metrics.outputSampleRate,
              resampler: native.metrics.resampler,
              resamplerQuality: native.metrics.resamplerQuality,
              encoderFrameCount: native.metrics.encoderFrameCount,
              decodeIterations: native.metrics.decodeIterations,
              emittedTokenCount: native.metrics.emittedTokenCount,
              emittedWordCount: native.metrics.emittedWordCount,
            }
          : context.metrics,
      },
      defaultNemoTimestampReconstructor,
      defaultNemoConfidenceReconstructor,
    ),
  );
}

export function createNemoRnntTranscriptNormalizer(
  classification: Partial<ModelClassification> = {},
): TranscriptNormalizer<NemoRnntNativeTranscript> {
  const normalizedClassification = normalizeClassification(
    DEFAULT_NEMO_RNNT_CLASSIFICATION,
    classification,
  );
  return createTranscriptNormalizer('nemo-rnnt', (native, context) =>
    mapNemoNativeToCanonical(
      native,
      normalizedClassification,
      {
        ...context,
        detailLevel: context.detailLevel ?? 'segments',
        metrics: native.metrics
          ? {
              preprocessMs: native.metrics.preprocessMs,
              encodeMs: native.metrics.encodeMs,
              decodeMs: native.metrics.decodeMs,
              tokenizeMs: native.metrics.tokenizeMs,
              postprocessMs: native.metrics.tokenizeMs,
              totalMs: native.metrics.totalMs,
              wallMs: native.metrics.wallMs,
              audioDurationSec: native.metrics.audioDurationSec,
              rtf: native.metrics.rtf,
              rtfx: native.metrics.rtfx,
              requestedPreprocessorBackend: native.metrics.requestedPreprocessorBackend,
              preprocessorBackend: native.metrics.preprocessorBackend,
              decodeAudioMs: native.metrics.decodeAudioMs,
              downmixMs: native.metrics.downmixMs,
              resampleMs: native.metrics.resampleMs,
              audioPreparationMs: native.metrics.audioPreparationMs,
              inputSampleRate: native.metrics.inputSampleRate,
              outputSampleRate: native.metrics.outputSampleRate,
              resampler: native.metrics.resampler,
              resamplerQuality: native.metrics.resamplerQuality,
              encoderFrameCount: native.metrics.encoderFrameCount,
              decodeIterations: native.metrics.decodeIterations,
              emittedTokenCount: native.metrics.emittedTokenCount,
              emittedWordCount: native.metrics.emittedWordCount,
            }
          : context.metrics,
      },
      defaultNemoTimestampReconstructor,
      defaultNemoConfidenceReconstructor,
    ),
  );
}

export function createNemoAedTranscriptNormalizer(
  classification: Partial<ModelClassification> = {},
): TranscriptNormalizer<NemoAedNativeTranscript> {
  const normalizedClassification = normalizeClassification(
    DEFAULT_NEMO_AED_CLASSIFICATION,
    classification,
  );
  return createTranscriptNormalizer('nemo-aed', (native, context) =>
    mapNemoNativeToCanonical(
      native,
      normalizedClassification,
      {
        ...context,
        detailLevel: context.detailLevel ?? 'segments',
        language: native.language ?? context.language,
        metrics: native.metrics
          ? {
              preprocessMs: native.metrics.preprocessMs,
              encodeMs: native.metrics.encodeMs,
              decodeMs: native.metrics.decodeMs,
              tokenizeMs: native.metrics.tokenizeMs,
              postprocessMs: native.metrics.tokenizeMs,
              totalMs: native.metrics.totalMs,
              wallMs: native.metrics.wallMs,
              audioDurationSec: native.metrics.audioDurationSec,
              rtf: native.metrics.rtf,
              rtfx: native.metrics.rtfx,
              requestedPreprocessorBackend: native.metrics.requestedPreprocessorBackend,
              preprocessorBackend: native.metrics.preprocessorBackend,
              decodeAudioMs: native.metrics.decodeAudioMs,
              downmixMs: native.metrics.downmixMs,
              resampleMs: native.metrics.resampleMs,
              audioPreparationMs: native.metrics.audioPreparationMs,
              inputSampleRate: native.metrics.inputSampleRate,
              outputSampleRate: native.metrics.outputSampleRate,
              resampler: native.metrics.resampler,
              resamplerQuality: native.metrics.resamplerQuality,
              encoderFrameCount: native.metrics.encoderFrameCount,
              decodeIterations: native.metrics.decodeIterations,
              emittedTokenCount: native.metrics.emittedTokenCount,
              emittedWordCount: native.metrics.emittedWordCount,
            }
          : context.metrics,
      },
      nemoAedTimestampReconstructor,
      defaultNemoConfidenceReconstructor,
    ),
  );
}

export function createLasrCtcTranscriptNormalizer(
  classification: Partial<ModelClassification> = {},
): TranscriptNormalizer<LasrCtcNativeTranscript> {
  const normalizedClassification = normalizeClassification(
    DEFAULT_LASR_CTC_CLASSIFICATION,
    classification,
  );
  return createTranscriptNormalizer('lasr-ctc', (native, context) =>
    mapLasrCtcNativeToCanonical(native, normalizedClassification, {
      ...context,
      detailLevel: context.detailLevel ?? 'segments',
      metrics: native.metrics ?? context.metrics,
    }),
  );
}

export function createWhisperTranscriptNormalizer(
  classification: Partial<ModelClassification> = {},
): TranscriptNormalizer<WhisperNativeTranscript> {
  const normalizedClassification = normalizeClassification(
    DEFAULT_WHISPER_CLASSIFICATION,
    classification,
  );
  return createTranscriptNormalizer('whisper-seq2seq', (native, context) =>
    mapWhisperNativeToCanonical(native, normalizedClassification, {
      ...context,
      detailLevel: context.detailLevel ?? 'segments',
    }),
  );
}

export function createLegacyParakeetTranscriptNormalizer(): TranscriptNormalizer<LegacyParakeetTranscript> {
  return createTranscriptNormalizer('parakeet-legacy', (native, context) => {
    const detail = context.detailLevel ?? 'segments';
    const words = (native.words ?? []).map((word, index) => ({
      index,
      text: word.text,
      startTime: word.start_time,
      endTime: word.end_time,
      confidence: word.confidence,
    }));
    const tokens = (native.tokens ?? []).map((token, index) => ({
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
    const segments =
      words.length > 0
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
          ? {
              preprocessMs: native.metrics.preprocess_ms,
              encodeMs: native.metrics.encode_ms,
              decodeMs: native.metrics.decode_ms,
              tokenizeMs: native.metrics.tokenize_ms,
              postprocessMs: native.metrics.tokenize_ms,
              totalMs: native.metrics.total_ms,
              wallMs: native.metrics.wall_ms,
              audioDurationSec: native.metrics.audio_duration_sec,
              rtf: native.metrics.rtf,
              rtfx: native.metrics.rtfx,
              preprocessorBackend: native.metrics.preprocessor_backend,
              decodeAudioMs: native.metrics.audio_decode_ms,
              downmixMs: native.metrics.downmix_ms,
              resampleMs: native.metrics.resample_ms,
              audioPreparationMs: native.metrics.audio_preparation_ms,
              inputSampleRate: native.metrics.input_sample_rate,
              outputSampleRate: native.metrics.output_sample_rate,
              resampler: native.metrics.resampler,
              resamplerQuality: native.metrics.resampler_quality,
              encoderFrameCount: native.metrics.encoder_frame_count,
              decodeIterations: native.metrics.decode_iterations,
              emittedTokenCount: native.metrics.emitted_token_count,
              emittedWordCount: native.metrics.emitted_word_count,
            }
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
  });
}

export const nemoTdtTranscriptNormalizer = createNemoTdtTranscriptNormalizer();
export const nemoRnntTranscriptNormalizer = createNemoRnntTranscriptNormalizer();
export const nemoAedTranscriptNormalizer = createNemoAedTranscriptNormalizer();
export const lasrCtcTranscriptNormalizer = createLasrCtcTranscriptNormalizer();
export const whisperTranscriptNormalizer = createWhisperTranscriptNormalizer();
export const legacyParakeetTranscriptNormalizer = createLegacyParakeetTranscriptNormalizer();

export function normalizeNemoTdtTranscript(
  native: NemoTdtNativeTranscript,
  context: TranscriptNormalizationContext = {},
): TranscriptResult {
  return nemoTdtTranscriptNormalizer.toCanonical(native, context);
}

export function normalizeNemoRnntTranscript(
  native: NemoRnntNativeTranscript,
  context: TranscriptNormalizationContext = {},
): TranscriptResult {
  return nemoRnntTranscriptNormalizer.toCanonical(native, context);
}

export function normalizeNemoAedTranscript(
  native: NemoAedNativeTranscript,
  context: TranscriptNormalizationContext = {},
): TranscriptResult {
  return nemoAedTranscriptNormalizer.toCanonical(native, context);
}

export function normalizeLasrCtcTranscript(
  native: LasrCtcNativeTranscript,
  context: TranscriptNormalizationContext = {},
): TranscriptResult {
  return lasrCtcTranscriptNormalizer.toCanonical(native, context);
}

export function normalizeWhisperTranscript(
  native: WhisperNativeTranscript,
  context: TranscriptNormalizationContext = {},
): TranscriptResult {
  return whisperTranscriptNormalizer.toCanonical(native, context);
}

export function normalizeLegacyParakeetTranscript(
  native: LegacyParakeetTranscript,
  context: TranscriptNormalizationContext = {},
): TranscriptResult {
  return legacyParakeetTranscriptNormalizer.toCanonical(native, context);
}

export function isTranscriptionEnvelope<TNative = unknown>(
  value: unknown,
): value is TranscriptionEnvelope<TNative> {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  if (!candidate.canonical || typeof candidate.canonical !== 'object') {
    return false;
  }

  const canonical = candidate.canonical as Record<string, unknown>;
  return (
    typeof canonical.text === 'string' &&
    Array.isArray(canonical.warnings) &&
    typeof canonical.meta === 'object' &&
    canonical.meta !== null
  );
}

export function getCanonicalTranscript<TNative = unknown>(
  value: TranscriptResult | TranscriptionEnvelope<TNative>,
): TranscriptResult {
  return isTranscriptionEnvelope<TNative>(value) ? value.canonical : value;
}
