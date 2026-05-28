import type { AssetProvider, AudioBufferLike, BaseTranscriptionOptions, SpeechRuntimeHooks } from '../../types/index.js';
import type { TextTokenizer, TokenizerSpec } from '../../tokenizers/index.js';

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

export type WhisperExecutionBackend = 'webgpu' | 'wasm';
export type WhisperQuantization = 'fp32' | 'fp16' | 'int8' | 'q4' | 'uint8';

export interface WhisperDirectArtifacts {
  readonly encoderUrl: string;
  readonly decoderUrl: string;
  readonly tokenizerUrl: string;
}

export interface WhisperDirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: WhisperDirectArtifacts;
  readonly encoderBackend?: WhisperExecutionBackend;
  readonly decoderBackend?: WhisperExecutionBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface WhisperHuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly cacheKeyFallbackRevisions?: readonly string[];
  readonly encoderBackend?: WhisperExecutionBackend;
  readonly decoderBackend?: WhisperExecutionBackend;
  readonly encoderQuant?: WhisperQuantization;
  readonly decoderQuant?: WhisperQuantization;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type WhisperArtifactSource = WhisperDirectArtifactSource | WhisperHuggingFaceSource;

export interface WhisperSeq2SeqModelOptions {
  readonly modelBaseUrl?: string;
  readonly revision?: string;
  readonly config?: Partial<WhisperSeq2SeqModelConfig>;
  readonly source?: WhisperArtifactSource;
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
  readonly maxNewTokens?: number;
  readonly noTimestamps?: boolean;
  readonly numBeams?: number;
  readonly lengthPenalty?: number;
  readonly patience?: number;
}

export interface WhisperSeq2SeqModelDependencies {
  readonly tokenizer?: TextTokenizer;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
}

export interface WhisperExecutor {
  ready?(): Promise<void> | void;
  transcribe(
    audio: AudioBufferLike,
    options: WhisperSeq2SeqTranscriptionOptions,
    context: WhisperDecodeContext,
  ): Promise<WhisperNativeTranscript>;
  dispose(): Promise<void> | void;
}

export interface WhisperDecodeContext {
  readonly modelId: string;
  readonly classification: { readonly family?: string };
  readonly config: WhisperSeq2SeqModelConfig;
  readonly tokenizer: TextTokenizer;
}
