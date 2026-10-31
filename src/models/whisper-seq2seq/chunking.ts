import { planWhisperChunks } from '../../pipeline/whisper-chunking.js';
import type { AudioBufferLike, TranscriptMetrics } from '../../types/index.js';
import type {
  WhisperNativeSegment,
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperNativeWord,
  WhisperSeq2SeqTranscriptionOptions,
} from './types.js';

export interface WhisperChunkTranscriptInput {
  readonly chunkStartTime: number;
  readonly transcript: WhisperNativeTranscript;
}

export function mergeWhisperChunkTranscripts(chunks: readonly WhisperChunkTranscriptInput[]): WhisperNativeTranscript {
  const segments: WhisperNativeSegment[] = [];
  const words: WhisperNativeWord[] = [];
  const tokens: WhisperNativeToken[] = [];
  const warnings = chunks.flatMap((chunk) => chunk.transcript.warnings ?? []);
  const language = chunks.find((chunk) => chunk.transcript.language)?.transcript.language;
  const metrics = mergeWhisperChunkMetrics(chunks.map((chunk) => chunk.transcript.metrics));

  for (const chunk of chunks) {
    const offset = chunk.chunkStartTime;
    for (const segment of chunk.transcript.segments ?? []) {
      segments.push({
        ...segment,
        index: segments.length,
        startTime: offsetTime(segment.startTime, offset),
        endTime: offsetTime(segment.endTime, offset),
      });
    }
    for (const word of chunk.transcript.words ?? []) {
      const adjusted: WhisperNativeWord = {
        ...word,
        startTime: offsetTime(word.startTime, offset),
        endTime: offsetTime(word.endTime, offset),
      };
      const previous = words[words.length - 1];
      if (
        previous &&
        normalizeWhisperWordText(previous.text) === normalizeWhisperWordText(adjusted.text) &&
        adjusted.startTime < previous.endTime
      ) {
        if ((adjusted.confidence ?? 0) > (previous.confidence ?? 0)) {
          words[words.length - 1] = adjusted;
        }
        continue;
      }
      words.push(adjusted);
    }
    for (const token of chunk.transcript.tokens ?? []) {
      tokens.push({
        ...token,
        index: tokens.length,
        startTime: token.startTime === undefined ? undefined : offsetTime(token.startTime, offset),
        endTime: token.endTime === undefined ? undefined : offsetTime(token.endTime, offset),
      });
    }
  }

  const indexedWords = words.map((word, index) => ({ ...word, index }));

  const utteranceText = segments.length > 0
    ? segments.map((segment) => segment.text).join(' ').trim()
    : chunks.map((chunk) => chunk.transcript.utteranceText).filter(Boolean).join(' ').trim();

  return {
    utteranceText,
    isFinal: chunks.every((chunk) => chunk.transcript.isFinal),
    ...(language ? { language } : {}),
    ...(segments.length > 0 ? { segments } : {}),
    ...(indexedWords.length > 0 ? { words: indexedWords } : {}),
    ...(tokens.length > 0 ? { tokens } : {}),
    ...(metrics ? { metrics } : {}),
    ...(warnings.length > 0 ? { warnings } : {}),
  };
}

function normalizeWhisperWordText(text: string): string {
  return text.trim().toLowerCase();
}

function offsetTime(time: number, offset: number): number {
  return Math.round((time + offset) * 1000) / 1000;
}

function sumMetrics(
  metrics: readonly TranscriptMetrics[],
  key: keyof TranscriptMetrics,
): number | undefined {
  let total = 0;
  let seen = false;
  for (const metric of metrics) {
    const value = metric[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      total += value;
      seen = true;
    }
  }
  return seen ? total : undefined;
}

function commonStringMetric(
  metrics: readonly TranscriptMetrics[],
  key: keyof TranscriptMetrics,
): string | undefined {
  const values = metrics
    .map((metric) => metric[key])
    .filter((value): value is string => typeof value === 'string');
  if (values.length === 0) return undefined;
  const first = values[0];
  return values.every((value) => value === first) ? first : 'mixed';
}

function roundMetric(value: number, digits = 3): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function mergeWhisperChunkMetrics(
  inputMetrics: readonly (TranscriptMetrics | undefined)[],
): TranscriptMetrics | undefined {
  const metrics = inputMetrics.filter((metric): metric is TranscriptMetrics => Boolean(metric));
  if (metrics.length === 0) {
    return undefined;
  }

  const totalMs = sumMetrics(metrics, 'totalMs');
  const audioDurationSec = sumMetrics(metrics, 'audioDurationSec');
  const rtf =
    totalMs !== undefined && audioDurationSec !== undefined && audioDurationSec > 0
      ? totalMs / (audioDurationSec * 1000)
      : undefined;

  return {
    preprocessMs: sumMetrics(metrics, 'preprocessMs'),
    encodeMs: sumMetrics(metrics, 'encodeMs'),
    decodeMs: sumMetrics(metrics, 'decodeMs'),
    tokenizeMs: sumMetrics(metrics, 'tokenizeMs'),
    postprocessMs: sumMetrics(metrics, 'postprocessMs'),
    languageDetectionMs: sumMetrics(metrics, 'languageDetectionMs'),
    decoderInitMs: sumMetrics(metrics, 'decoderInitMs'),
    decoderInitInputMs: sumMetrics(metrics, 'decoderInitInputMs'),
    decoderInitRunMs: sumMetrics(metrics, 'decoderInitRunMs'),
    decoderInitOutputMs: sumMetrics(metrics, 'decoderInitOutputMs'),
    decoderStepMs: sumMetrics(metrics, 'decoderStepMs'),
    decoderStepFeedBuildMs: sumMetrics(metrics, 'decoderStepFeedBuildMs'),
    decoderStepTensorCloneMs: sumMetrics(metrics, 'decoderStepTensorCloneMs'),
    decoderStepRunMs: sumMetrics(metrics, 'decoderStepRunMs'),
    decoderStepOutputMs: sumMetrics(metrics, 'decoderStepOutputMs'),
    decoderLogitProcessMs: sumMetrics(metrics, 'decoderLogitProcessMs'),
    decoderStepCount: sumMetrics(metrics, 'decoderStepCount'),
    decoderGpuTensorInputs: sumMetrics(metrics, 'decoderGpuTensorInputs'),
    decoderCpuTensorInputs: sumMetrics(metrics, 'decoderCpuTensorInputs'),
    decoderGpuTensorOutputs: sumMetrics(metrics, 'decoderGpuTensorOutputs'),
    decoderCpuTensorOutputs: sumMetrics(metrics, 'decoderCpuTensorOutputs'),
    decoderGpuTensorDownloads: sumMetrics(metrics, 'decoderGpuTensorDownloads'),
    decoderKvCacheLocation: commonStringMetric(metrics, 'decoderKvCacheLocation'),
    totalMs,
    wallMs: sumMetrics(metrics, 'wallMs'),
    audioDurationSec,
    rtf: rtf !== undefined ? roundMetric(rtf, 4) : undefined,
    rtfx:
      totalMs !== undefined && audioDurationSec !== undefined && totalMs > 0
        ? roundMetric(audioDurationSec / (totalMs / 1000), 4)
        : undefined,
    requestedPreprocessorBackend: commonStringMetric(metrics, 'requestedPreprocessorBackend'),
    preprocessorBackend: commonStringMetric(metrics, 'preprocessorBackend'),
    encoderFrameCount: sumMetrics(metrics, 'encoderFrameCount'),
    decodeIterations: sumMetrics(metrics, 'decodeIterations'),
    emittedTokenCount: sumMetrics(metrics, 'emittedTokenCount'),
    emittedWordCount: sumMetrics(metrics, 'emittedWordCount'),
  };
}

export function shouldChunkWhisperAudio(
  audio: AudioBufferLike,
  options: WhisperSeq2SeqTranscriptionOptions,
): boolean {
  if (options.windowing === 'disabled' || options.unsafeAllowOverMaxWindow) return false;
  const maxDuration = options.chunkLengthSeconds ?? options.maxInputDurationSeconds ?? 30;
  return audio.durationSeconds > maxDuration;
}

export async function transcribeWhisperLongAudio(
  audio: AudioBufferLike,
  options: WhisperSeq2SeqTranscriptionOptions,
  transcribeChunk: (
    chunkAudio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
  ) => Promise<WhisperNativeTranscript>,
): Promise<WhisperNativeTranscript> {
  const pcmData = audio.channels?.[0] ?? new Float32Array(0);
  const chunkLengthSeconds = options.chunkLengthSeconds ?? 30;
  const chunks = planWhisperChunks(
    pcmData.length,
    audio.sampleRate,
    chunkLengthSeconds,
    options.strideLengthSeconds,
  );

  const chunkTranscripts: WhisperChunkTranscriptInput[] = [];
  for (const chunk of chunks) {
    const samples = pcmData.slice(chunk.startSample, chunk.endSample);
    const chunkAudio: AudioBufferLike = {
      sampleRate: audio.sampleRate,
      durationSeconds: samples.length / audio.sampleRate,
      channels: [samples],
      numberOfChannels: 1,
      numberOfFrames: samples.length,
    };
    const transcript = await transcribeChunk(chunkAudio, {
      ...options,
      unsafeAllowOverMaxWindow: true,
    });
    chunkTranscripts.push({ chunkStartTime: chunk.startTime, transcript });
  }

  return mergeWhisperChunkTranscripts(chunkTranscripts);
}

