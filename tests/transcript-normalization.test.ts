import {
  createLasrCtcTranscriptNormalizer,
  createLegacyParakeetTranscriptNormalizer,
  createNemoTdtTranscriptNormalizer,
  createWhisperTranscriptNormalizer,
  getCanonicalTranscript,
  isTranscriptionEnvelope,
  type LasrCtcNativeTranscript,
  type LegacyParakeetTranscript,
  type NemoTdtNativeTranscript,
  type WhisperNativeTranscript,
} from '@asrjs/speech-recognition';
import { describe, expect, it } from 'vitest';

describe('transcript normalization helpers', () => {
  it('normalizes NeMo TDT native output into canonical transcript data', () => {
    const native: NemoTdtNativeTranscript = {
      utteranceText: 'hello world',
      isFinal: true,
      words: [
        {
          index: 0,
          text: 'hello',
          startTime: 0,
          endTime: 0.5,
          confidence: 0.9,
        },
        {
          index: 1,
          text: 'world',
          startTime: 0.5,
          endTime: 1,
          confidence: 0.8,
        },
      ],
      tokens: [
        {
          index: 0,
          id: 1,
          text: 'hello',
          startTime: 0,
          endTime: 0.5,
          confidence: 0.9,
        },
      ],
      confidence: {
        utterance: 0.85,
        wordAverage: 0.85,
        tokenAverage: 0.9,
      },
    };

    const canonical = createNemoTdtTranscriptNormalizer().toCanonical(native, {
      detailLevel: 'detailed',
      modelId: 'parakeet-tdt-0.6b-v3',
      backendId: 'webgpu',
    });

    expect(canonical.text).toBe('hello world');
    expect(canonical.meta.modelFamily).toBe('nemo-tdt');
    expect(canonical.meta.averageConfidence).toBe(0.85);
    expect(canonical.words).toHaveLength(2);
    expect(canonical.tokens).toHaveLength(1);
  });

  it('normalizes LASR CTC and Whisper native outputs with the same interface', () => {
    const lasrNative: LasrCtcNativeTranscript = {
      utteranceText: 'medical scaffold',
      isFinal: true,
      words: [
        {
          index: 0,
          text: 'medical',
          startTime: 0,
          endTime: 0.4,
          confidence: 0.9,
        },
      ],
      tokens: [
        {
          index: 0,
          id: 12,
          text: 'medical',
          startTime: 0,
          endTime: 0.4,
          confidence: 0.9,
        },
      ],
      confidence: {
        utterance: 0.9,
        wordAverage: 0.9,
        tokenAverage: 0.9,
      },
    };
    const whisperNative: WhisperNativeTranscript = {
      utteranceText: 'translated whisper scaffold',
      isFinal: true,
      language: 'en',
      segments: [
        {
          index: 0,
          text: 'translated whisper scaffold',
          startTime: 0,
          endTime: 1.2,
          confidence: 0.93,
        },
      ],
      tokens: [
        {
          index: 0,
          id: 99,
          text: 'translated',
          startTime: 0,
          endTime: 0.3,
          confidence: 0.94,
        },
      ],
    };

    const lasrCanonical = createLasrCtcTranscriptNormalizer().toCanonical(lasrNative, {
      detailLevel: 'detailed',
    });
    const whisperCanonical = createWhisperTranscriptNormalizer().toCanonical(whisperNative, {
      detailLevel: 'detailed',
    });

    expect(lasrCanonical.meta.modelFamily).toBe('lasr-ctc');
    expect(lasrCanonical.tokens?.[0]?.text).toBe('medical');
    expect(whisperCanonical.meta.modelFamily).toBe('whisper');
    expect(whisperCanonical.segments?.[0]?.text).toBe('translated whisper scaffold');
  });

  it('normalizes legacy Parakeet JSON and exposes canonical/envelope helpers', () => {
    const legacy: LegacyParakeetTranscript = {
      utterance_text: 'legacy parakeet',
      words: [
        {
          text: 'legacy',
          start_time: 0,
          end_time: 0.5,
          confidence: 0.8,
        },
      ],
      tokens: [
        {
          id: 1,
          token: 'legacy',
          start_time: 0,
          end_time: 0.5,
          confidence: 0.8,
        },
      ],
      confidence_scores: {
        utterance: 0.8,
        word_avg: 0.8,
        token_avg: 0.8,
      },
      metrics: {
        preprocess_ms: 1,
        encode_ms: 2,
        decode_ms: 3,
        tokenize_ms: 1,
        total_ms: 7,
        wall_ms: 7.5,
        audio_duration_sec: 3.2,
        rtf: 0.1,
        rtfx: 10,
        preprocessor_backend: 'js',
        resample_ms: 0.8,
        encoder_frame_count: 42,
        decode_iterations: 11,
        emitted_token_count: 6,
        emitted_word_count: 2,
      },
      is_final: true,
    };

    const normalizer = createLegacyParakeetTranscriptNormalizer();
    const envelope = normalizer.toEnvelope(legacy, {
      detailLevel: 'detailed',
      modelId: 'parakeet-tdt-0.6b-v3',
    });

    expect(isTranscriptionEnvelope(envelope)).toBe(true);
    expect(getCanonicalTranscript(envelope).text).toBe('legacy parakeet');
    expect(envelope.canonical.meta.modelFamily).toBe('parakeet');
    expect(envelope.canonical.meta.metrics?.totalMs).toBe(7);
    expect(envelope.canonical.meta.metrics?.wallMs).toBe(7.5);
    expect(envelope.canonical.meta.metrics?.audioDurationSec).toBe(3.2);
    expect(envelope.canonical.meta.metrics?.rtfx).toBe(10);
    expect(envelope.canonical.meta.metrics?.preprocessorBackend).toBe('js');
    expect(envelope.canonical.meta.metrics?.encoderFrameCount).toBe(42);
    expect(envelope.canonical.tokens?.[0]?.text).toBe('legacy');
  });
});
