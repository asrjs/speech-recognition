import type { PartialTranscript, TranscriptResult } from '../../types/index.js';
import { joinTranscriptFragments } from './merge.js';

export class TranscriptAccumulator {
  private revision = 0;
  private committedText = '';
  private previewText = '';

  update(result: TranscriptResult, kind: 'partial' | 'final'): PartialTranscript {
    this.revision += 1;

    if (kind === 'final') {
      this.committedText = joinTranscriptFragments(this.committedText, result.text);
      this.previewText = '';
    } else {
      this.previewText = result.text;
    }

    return {
      kind,
      revision: this.revision,
      text:
        kind === 'final'
          ? this.committedText
          : joinTranscriptFragments(this.committedText, this.previewText),
      committedText: this.committedText,
      previewText: this.previewText,
      warnings: result.warnings,
      meta: {
        ...result.meta,
        isFinal: kind === 'final',
      },
      segments: result.segments,
      words: result.words,
      tokens: result.tokens,
    };
  }

  reset(): void {
    this.revision = 0;
    this.committedText = '';
    this.previewText = '';
  }

  getState(): { revision: number; committedText: string; previewText: string } {
    return {
      revision: this.revision,
      committedText: this.committedText,
      previewText: this.previewText,
    };
  }
}
