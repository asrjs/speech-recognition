import type {
  SubtitleCue,
  TranscriptResult,
  TranscriptSegment,
  TranscriptSentence,
  TranscriptWord,
} from '../types/index.js';
import { joinTranscriptWords, partitionWordsIntoSentences } from './sentence-segmenter.js';

export type SubtitleFormat = 'srt' | 'vtt';
export type SubtitleCueSource = 'sentences' | 'segments' | 'words';

export interface SubtitleFormattingOptions {
  readonly source?: SubtitleCueSource;
  readonly maxGapSeconds?: number;
  readonly includeSpeaker?: boolean;
}

type TimedTextSpan = TranscriptSentence | TranscriptSegment;

export function formatSubtitleTimestamp(seconds: number, format: SubtitleFormat): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const totalMilliseconds = Math.round(safeSeconds * 1000);
  const milliseconds = totalMilliseconds % 1000;
  const totalSeconds = Math.floor(totalMilliseconds / 1000);
  const second = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minute = totalMinutes % 60;
  const hour = Math.floor(totalMinutes / 60);
  const decimal = format === 'srt' ? ',' : '.';
  return `${pad2(hour)}:${pad2(minute)}:${pad2(second)}${decimal}${pad3(milliseconds)}`;
}

export function transcriptToSrt(
  transcript: TranscriptResult,
  options: SubtitleFormattingOptions = {},
): string {
  return cuesToSrt(transcriptToSubtitleCues(transcript, options), options);
}

export function transcriptToVtt(
  transcript: TranscriptResult,
  options: SubtitleFormattingOptions = {},
): string {
  return cuesToVtt(transcriptToSubtitleCues(transcript, options), options);
}

export function transcriptToSubtitleCues(
  transcript: TranscriptResult,
  options: SubtitleFormattingOptions = {},
): SubtitleCue[] {
  const source = options.source ?? (transcript.sentences ? 'sentences' : 'segments');
  if (source === 'words') {
    return wordsToCues(transcript.words ?? [], options);
  }

  const spans = source === 'sentences' ? transcript.sentences : transcript.segments;
  return (spans ?? []).map((span, index) => spanToCue(index, span));
}

export function cuesToSrt(
  cues: readonly SubtitleCue[],
  options: SubtitleFormattingOptions = {},
): string {
  return cues
    .map((cue, index) => [
      String(index + 1),
      `${formatSubtitleTimestamp(cue.startTime, 'srt')} --> ${formatSubtitleTimestamp(cue.endTime, 'srt')}`,
      formatCueText(cue, options),
      '',
    ].join('\n'))
    .join('\n');
}

export function cuesToVtt(
  cues: readonly SubtitleCue[],
  options: SubtitleFormattingOptions = {},
): string {
  const body = cues
    .map((cue) => [
      `${formatSubtitleTimestamp(cue.startTime, 'vtt')} --> ${formatSubtitleTimestamp(cue.endTime, 'vtt')}`,
      formatCueText(cue, options),
      '',
    ].join('\n'))
    .join('\n');
  return `WEBVTT\n\n${body}`;
}

function wordsToCues(
  words: readonly TranscriptWord[],
  options: SubtitleFormattingOptions,
): SubtitleCue[] {
  const sentences = partitionWordsIntoSentences(words, {
    gapThresholdSeconds: options.maxGapSeconds,
  });
  return sentences.map((sentence, index) => spanToCue(index, sentence));
}

function spanToCue(index: number, span: TimedTextSpan): SubtitleCue {
  return {
    index,
    text: span.text,
    startTime: span.startTime,
    endTime: span.endTime,
    speaker: span.speaker,
  };
}

function formatCueText(cue: SubtitleCue, options: SubtitleFormattingOptions): string {
  const text = sanitizeSubtitleText(cue.text || joinTranscriptWords([]));
  if (options.includeSpeaker === false || !cue.speaker) {
    return text;
  }
  return `[${cue.speaker}]: ${text}`;
}

function sanitizeSubtitleText(text: string): string {
  return text.trim().replace(/-->/g, '->');
}

function pad2(value: number): string {
  return String(value).padStart(2, '0');
}

function pad3(value: number): string {
  return String(value).padStart(3, '0');
}
