export type TranscriptDetailLevel = 'text' | 'segments' | 'words' | 'detailed';
export type TranscriptResponseFlavor = 'canonical' | 'canonical+native' | 'native';

export interface TranscriptWarning {
  readonly code: string;
  readonly message: string;
  readonly recoverable?: boolean;
}

export interface TranscriptToken {
  readonly index: number;
  readonly text: string;
  readonly rawText?: string;
  readonly id?: number;
  readonly isWordStart?: boolean;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly frameIndex?: number;
  readonly logProb?: number;
  readonly tdtStep?: number;
}

export interface TranscriptWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly tokenIndices?: readonly number[];
}

export interface TranscriptSegment {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly wordIndices?: readonly number[];
}

export interface TranscriptMetrics {
  readonly preprocessMs?: number;
  readonly encodeMs?: number;
  readonly decodeMs?: number;
  readonly postprocessMs?: number;
  readonly totalMs?: number;
  readonly rtf?: number;
}

export interface TranscriptMeta {
  readonly detailLevel: TranscriptDetailLevel;
  readonly isFinal: boolean;
  readonly modelFamily?: string;
  readonly modelId?: string;
  readonly backendId?: string;
  readonly language?: string;
  readonly sampleRate?: number;
  readonly durationSeconds?: number;
  readonly tokenCount?: number;
  readonly wordCount?: number;
  readonly segmentCount?: number;
  readonly averageConfidence?: number;
  readonly averageSegmentConfidence?: number;
  readonly averageWordConfidence?: number;
  readonly averageTokenConfidence?: number;
  readonly nativeAvailable?: boolean;
  readonly producedAt?: string;
  readonly backendNotes?: readonly string[];
  readonly metrics?: TranscriptMetrics;
}

export interface TranscriptResult {
  readonly text: string;
  readonly warnings: readonly TranscriptWarning[];
  readonly meta: TranscriptMeta;
  readonly segments?: readonly TranscriptSegment[];
  readonly words?: readonly TranscriptWord[];
  readonly tokens?: readonly TranscriptToken[];
}

export interface PartialTranscript {
  readonly kind: 'partial' | 'final';
  readonly revision: number;
  readonly text: string;
  readonly committedText: string;
  readonly previewText: string;
  readonly warnings: readonly TranscriptWarning[];
  readonly meta: TranscriptMeta;
  readonly segments?: readonly TranscriptSegment[];
  readonly words?: readonly TranscriptWord[];
  readonly tokens?: readonly TranscriptToken[];
}

export interface TranscriptionEnvelope<TNative = unknown> {
  readonly canonical: TranscriptResult;
  readonly native?: TNative;
}

export type TranscriptResponse<
  TNative = unknown,
  TFlavor extends TranscriptResponseFlavor = 'canonical'
> = TFlavor extends 'native'
  ? TNative
  : TFlavor extends 'canonical+native'
    ? TranscriptionEnvelope<TNative>
    : TranscriptResult;

