import type { TranscriptResult } from '../types/index.js';
import type { PipelineStage } from './composition.js';
import { transcriptToSrt, transcriptToVtt } from './subtitles.js';

export type TranscriptOutputFormat = 'json' | 'srt' | 'vtt';

export interface TranscriptOutputStageOptions {
  readonly id?: string;
  readonly formats?: readonly TranscriptOutputFormat[];
}

export function createTranscriptOutputStage(
  options: TranscriptOutputStageOptions = {},
): PipelineStage {
  const formats = options.formats ?? ['json'];
  return {
    id: options.id ?? 'transcript-output-sidecars',
    run(context) {
      if (!context.transcript) {
        return {};
      }
      return {
        sidecars: createTranscriptSidecars(context.transcript, formats),
      };
    },
  };
}

export function createTranscriptSidecars(
  transcript: TranscriptResult,
  formats: readonly TranscriptOutputFormat[],
): Record<string, unknown> {
  const sidecars: Record<string, unknown> = {};
  for (const format of formats) {
    if (format === 'json') {
      sidecars.json = transcript;
    } else if (format === 'srt') {
      sidecars.srt = transcriptToSrt(transcript);
    } else if (format === 'vtt') {
      sidecars.vtt = transcriptToVtt(transcript);
    }
  }
  return sidecars;
}
