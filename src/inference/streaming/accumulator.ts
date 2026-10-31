import type {
  PartialTranscript,
  TranscriptResult,
  TranscriptSegment,
  TranscriptToken,
  TranscriptWord,
} from '../../types/index.js';
import { joinTranscriptFragments } from './merge.js';

export class TranscriptAccumulator {
  private revision = 0;
  private committedText = '';
  private previewText = '';
  private committedWords: TranscriptWord[] = [];
  private committedSegments: TranscriptSegment[] = [];
  private committedTokens: TranscriptToken[] = [];

  commit(
    text: string,
    words: readonly TranscriptWord[] = [],
    segments: readonly TranscriptSegment[] = [],
    tokens: readonly TranscriptToken[] = [],
  ): void {
    if (text.trim().length > 0) {
      this.committedText = joinTranscriptFragments(this.committedText, text);
    }
    if (words.length > 0) {
      this.committedWords.push(...words);
    }
    if (segments.length > 0) {
      this.committedSegments.push(...segments);
    }
    if (tokens.length > 0) {
      this.committedTokens.push(...tokens);
    }
  }

  getCommittedWords(): readonly TranscriptWord[] {
    return this.committedWords;
  }

  getCommittedSegments(): readonly TranscriptSegment[] {
    return this.committedSegments;
  }

  getCommittedTokens(): readonly TranscriptToken[] {
    return this.committedTokens;
  }

  getCommittedText(): string {
    return this.committedText;
  }

  getPreviewText(): string {
    return this.previewText;
  }

  update(result: TranscriptResult, kind: 'partial' | 'final'): PartialTranscript {
    this.revision += 1;

    if (kind === 'final') {
      if (result.text?.trim()) {
        this.committedText = joinTranscriptFragments(this.committedText, result.text);
      }
      if (result.words?.length) {
        this.committedWords.push(...result.words);
      }
      if (result.segments?.length) {
        this.committedSegments.push(...result.segments);
      }
      if (result.tokens?.length) {
        this.committedTokens.push(...result.tokens);
      }
      this.previewText = '';
    } else {
      this.previewText = result.text ?? '';
    }

    const allWords = [
      ...this.committedWords,
      ...(kind === 'final' ? [] : (result.words ?? [])),
    ].map((word, index) => ({ ...word, index }));

    const allSegments = [
      ...this.committedSegments,
      ...(kind === 'final' ? [] : (result.segments ?? [])),
    ].map((segment, index) => ({ ...segment, index }));

    const allTokens = [
      ...this.committedTokens,
      ...(kind === 'final' ? [] : (result.tokens ?? [])),
    ].map((token, index) => ({ ...token, index }));

    return {
      kind,
      revision: this.revision,
      text:
        kind === 'final'
          ? this.committedText
          : joinTranscriptFragments(this.committedText, this.previewText),
      committedText: this.committedText,
      previewText: this.previewText,
      warnings: result.warnings ?? [],
      meta: {
        ...result.meta,
        isFinal: kind === 'final',
        wordCount: allWords.length > 0 ? allWords.length : result.meta?.wordCount,
        segmentCount: allSegments.length > 0 ? allSegments.length : result.meta?.segmentCount,
        tokenCount: allTokens.length > 0 ? allTokens.length : result.meta?.tokenCount,
      },
      segments: allSegments.length > 0 ? allSegments : undefined,
      words: allWords.length > 0 ? allWords : undefined,
      tokens: allTokens.length > 0 ? allTokens : undefined,
    };
  }

  reset(): void {
    this.revision = 0;
    this.committedText = '';
    this.previewText = '';
    this.committedWords = [];
    this.committedSegments = [];
    this.committedTokens = [];
  }

  getState(): { revision: number; committedText: string; previewText: string } {
    return {
      revision: this.revision,
      committedText: this.committedText,
      previewText: this.previewText,
    };
  }
}

