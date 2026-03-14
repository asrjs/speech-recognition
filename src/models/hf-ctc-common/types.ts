import type { TokenizerSpec, TextTokenizer } from '../../tokenizers/index.js';
import type { BaseTranscriptionOptions } from '../../types/index.js';

export interface HfCtcModelConfig {
  readonly ecosystem: 'hf';
  readonly architecture: 'hf-ctc-common';
  readonly processorArchitecture: 'wav2vec2-conv';
  readonly encoderArchitecture: string;
  readonly decoderArchitecture: 'ctc';
  readonly sampleRate: number;
  readonly rawStride: number;
  readonly vocabularySize?: number;
  readonly languages: readonly string[];
  readonly tokenizer: TokenizerSpec;
}

export interface HfCtcModelOptions {
  readonly modelBaseUrl?: string;
  readonly revision?: string;
  readonly config?: Partial<HfCtcModelConfig>;
}

export interface HfCtcNativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
  readonly logitIndex?: number;
}

export interface HfCtcNativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface HfCtcNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly words?: readonly HfCtcNativeWord[];
  readonly tokens?: readonly HfCtcNativeToken[];
  readonly confidence?: {
    readonly utterance?: number;
    readonly tokenAverage?: number;
    readonly wordAverage?: number;
  };
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

export interface HfCtcTranscriptionOptions extends BaseTranscriptionOptions {
  readonly returnTokenIds?: boolean;
  readonly returnLogitIndices?: boolean;
}

export interface HfCtcModelDependencies {
  readonly tokenizer?: TextTokenizer;
}
