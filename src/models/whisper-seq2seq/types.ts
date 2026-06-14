import type {
  AssetProvider,
  AudioBufferLike,
  BaseTranscriptionOptions,
  SpeechRuntimeHooks,
  TranscriptMetrics,
} from '../../types/index.js';
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
  readonly experimentalGpuKvCache?: boolean;
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
  readonly experimentalGpuKvCache?: boolean;
}

export type WhisperArtifactSource = WhisperDirectArtifactSource | WhisperHuggingFaceSource | WhisperSplitGraphArtifactSource;

export interface ExternalDataEntry {
  /** Path relative to the graph file, used as ONNX internal location reference. */
  readonly path: string;
  /** Filename of the external data file. */
  readonly file: string;
  /** Size in bytes of the external data file. */
  readonly sizeBytes?: number;
  /** Optional SHA-256 hash of the external data file. */
  readonly sha256?: string;
}

export interface WhisperSplitGraphArtifacts {
  readonly encoderUrl: string;
  readonly decoderInitUrl: string;
  readonly decoderStepUrl: string;
  readonly decoderAlignUrl?: string;
  readonly tokenizerUrl: string;
  readonly manifestUrl: string;
  /** External data URLs for each graph file, keyed by graph name:
   *  encoder, decoder_init, decoder_step, decoder_align.
   *  Populated from manifest.json when the model uses external ONNX data. */
  readonly externalDataUrls?: Partial<Record<'encoder' | 'decoder_init' | 'decoder_step' | 'decoder_align', readonly ExternalDataEntry[]>>;
}

export interface WhisperSplitGraphArtifactSource {
  readonly kind: 'splitgraph';
  readonly artifacts: WhisperSplitGraphArtifacts;
  readonly encoderBackend?: WhisperExecutionBackend;
  readonly decoderBackend?: WhisperExecutionBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly experimentalGpuKvCache?: boolean;
}

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

export interface WhisperNativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly tokenIds?: readonly number[];
  readonly tokenIndices?: readonly number[];
}

export interface WhisperNativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language?: string;
  readonly segments?: readonly WhisperNativeSegment[];
  readonly words?: readonly WhisperNativeWord[];
  readonly tokens?: readonly WhisperNativeToken[];
  readonly metrics?: TranscriptMetrics;
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

export interface WhisperSeq2SeqTranscriptionOptions extends BaseTranscriptionOptions {
  readonly task?: 'transcribe' | 'translate';
  readonly returnSpecialTokens?: boolean;
  readonly returnPromptTokens?: boolean;
  readonly maxNewTokens?: number;
  readonly noTimestamps?: boolean;
  /** Number of beams for beam search (1 = greedy). WhisperX: beam_size */
  readonly numBeams?: number;
  /** Length penalty for beam search (0 = no penalty). WhisperX: length_penalty */
  readonly lengthPenalty?: number;
  /** Beam search patience: max consecutive EOS before stopping early. WhisperX: patience */
  readonly patience?: number;
  /** Greedy decoding temperature. 0 uses argmax; >0 samples from scaled logits. */
  readonly temperature?: number;
  /** Number of independent decodings to run. WhisperX: best_of (default: null = numBeams) */
  readonly bestOf?: number;
  /**
   * Optional per-token logit callback — fired after logit processing, before argmax.
   * Enables quality gates (logprob, entropy, no-speech) to collect per-token data.
   * Signature: (chosenTokenId, processedLogits, { tokens, beginIndex }) => void
   */
  readonly onTokenLogits?: (
    chosenTokenId: number,
    processedLogits: Float32Array,
    ctx: { readonly tokens: readonly number[]; readonly beginIndex: number },
  ) => void;
  /**
   * Extra tokens to append after the standard prompt.
   * Used by EnhancedWhisperExecutor for condition_on_previous_text.
   * Format: [<|0.00|>, ...previous_tokens]
   */
  readonly extraPromptTokens?: readonly number[];
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
