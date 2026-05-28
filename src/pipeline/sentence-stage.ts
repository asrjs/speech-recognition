import type {
  BaseTranscriptionOptions,
  TranscriptResult,
  TranscriptSegment,
  TranscriptSentence,
} from '../types/index.js';
import type { PipelineStage } from './composition.js';
import {
  partitionWordsIntoSentences,
  type SentenceSegmentationOptions,
} from './sentence-segmenter.js';

export interface SentenceSegmentationStageOptions extends SentenceSegmentationOptions {
  readonly id?: string;
  readonly source?: 'words';
  readonly updateSegments?: boolean;
}

export function createSentenceSegmentationStage<
  TOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
>(options: SentenceSegmentationStageOptions = {}): PipelineStage<TOptions> {
  return {
    id: options.id ?? 'sentence-segmentation',
    run(context) {
      const transcript = context.transcript;
      if (!transcript?.words || transcript.words.length === 0) {
        return transcript ? { transcript } : undefined;
      }

      const segmented = addSentenceSegmentation(transcript, options);
      return { transcript: segmented };
    },
  };
}

export function addSentenceSegmentation(
  transcript: TranscriptResult,
  options: SentenceSegmentationStageOptions = {},
): TranscriptResult {
  if (!transcript.words || transcript.words.length === 0) {
    return transcript;
  }

  const sentences = partitionWordsIntoSentences(transcript.words, {
    gapThresholdSeconds: options.gapThresholdSeconds,
    nonBreakingPeriodWords: options.nonBreakingPeriodWords,
  });
  const segments = options.updateSegments ? sentences.map(sentenceToSegment) : transcript.segments;

  return {
    ...transcript,
    sentences,
    segments,
    meta: {
      ...transcript.meta,
      sentenceCount: sentences.length,
      ...(options.updateSegments ? { segmentCount: sentences.length } : {}),
    },
  };
}

function sentenceToSegment(sentence: TranscriptSentence): TranscriptSegment {
  return {
    index: sentence.index,
    text: sentence.text,
    startTime: sentence.startTime,
    endTime: sentence.endTime,
    confidence: sentence.confidence,
    wordIndices: sentence.wordIndices,
    speaker: sentence.speaker,
  };
}
