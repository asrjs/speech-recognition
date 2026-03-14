import type { TokenizerSpec, TextTokenizer } from '../../tokenizers/index.js';
import type { BaseTranscriptionOptions } from '../../types/index.js';

export interface WhisperSeq2SeqModelConfig {
  readonly ecosystem: 'openai';
  readonly architecture: 'whisper-seq2seq';
  readonly processorArchitecture: 'whisper-mel';
  readonly encoderArchitecture: 'whisper-transformer';
  readonly decoderArchitecture: 'transformer-decoder';
  readonly sampleRate: number;
  readonly melBins: number;
  readonly maxSourcePositions: number;
  readonly maxTargetPositions: number;
  readonly vocabularySize?: number;
  readonly languages: readonly string[];
  readonly tokenizer: TokenizerSpec;
}

export interface WhisperSeq2SeqModelOptions {
  readonly modelBaseUrl?: string;
  readonly revision?: string;
  readonly config?: Partial<WhisperSeq2SeqModelConfig>;
}

export interface WhisperNativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly special?: boolean;
}

export interface WhisperNativeSegment {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface WhisperNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language?: string;
  readonly segments?: readonly WhisperNativeSegment[];
  readonly tokens?: readonly WhisperNativeToken[];
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

export interface WhisperSeq2SeqTranscriptionOptions extends BaseTranscriptionOptions {
  readonly task?: 'transcribe' | 'translate';
  readonly returnSpecialTokens?: boolean;
  readonly returnPromptTokens?: boolean;
}

export interface WhisperSeq2SeqModelDependencies {
  readonly tokenizer?: TextTokenizer;
}
