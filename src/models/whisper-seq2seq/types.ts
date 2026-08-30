import type { TokenQualityTrace } from '../../quality/types.js';
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
  readonly experimentalWebGpuEncoderGraphCapture?: boolean;
  readonly experimentalGpuKvCache?: boolean;
  readonly experimentalGpuKvBeam?: boolean;
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
  readonly experimentalWebGpuEncoderGraphCapture?: boolean;
  readonly experimentalGpuKvCache?: boolean;
  readonly experimentalGpuKvBeam?: boolean;
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

/**
 * Optional higher-precision artifact pair used only for Whisper word
 * alignment. The normal encoder/decoder path remains the fast inference
 * path; this pair is evaluated lazily when a caller selects reference word
 * timestamps (or when `wordTimestampSource` is `auto`).
 */
export interface WhisperSplitGraphAlignmentReference {
  /** Encoder with the precision/accumulation contract used by the reference. */
  readonly encoderUrl: string;
  /** Causal selected-head alignment graph paired with this encoder. */
  readonly decoderAlignUrl: string;
  /** Optional manifest; the primary splitgraph manifest is used otherwise. */
  readonly manifestUrl?: string;
  /** External-data shards for the reference encoder/alignment graphs. */
  readonly externalDataUrls?: Partial<
    Record<'encoder' | 'decoder_align', readonly ExternalDataEntry[]>
  >;
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
  /** Optional precision/reference pair used only for word alignment. */
  readonly alignmentReference?: WhisperSplitGraphAlignmentReference;
}

export interface WhisperSplitGraphArtifactSource {
  readonly kind: 'splitgraph';
  readonly artifacts: WhisperSplitGraphArtifacts;
  readonly encoderBackend?: WhisperExecutionBackend;
  readonly decoderBackend?: WhisperExecutionBackend;
  /** Backend for the optional alignment-reference encoder and decoder_align. */
  readonly alignmentReferenceBackend?: WhisperExecutionBackend;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly experimentalWebGpuEncoderGraphCapture?: boolean;
  readonly experimentalGpuKvCache?: boolean;
  readonly experimentalGpuKvBeam?: boolean;
  /** DIAGNOSTIC: Force encoder output to CPU (Track A2). When true, encoder
   *  output is downloaded to CPU even with gpuKv enabled, to measure
   *  cross-session GPU tensor handoff penalty. */
  readonly encoderOutputCpu?: boolean;
  /** DIAGNOSTIC (B2-C): Enable graph capture for decoder_step session. */
  readonly decoderGraphCapture?: boolean;
  /** DIAGNOSTIC (B2-B): freeDimensionOverrides for decoder_step session. */
  readonly decoderFreeDimensionOverrides?: Record<string, number>;
  /** DIAGNOSTIC (Edge A): Re-wrap encoder GPU output as fresh Tensor.fromGpuBuffer. */
  readonly encoderBufferRewrap?: boolean;
  /** DIAGNOSTIC (Edge B2): Force GPU flush before decoder_init. */
  readonly encoderGpuFlush?: boolean;
  /** PROFILING (encoderGpuDrain): Force GPU drain + re-wrap after encoder.
   *  Calls getData(false) to drain the GPU queue, then re-wraps the same
   *  GPUBuffer as a fresh tensor.  Adds ~18ms staging buffer overhead.
   *  Use for honest per-phase profiling; leave off for production latency. */
  readonly encoderGpuDrain?: boolean;
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
  /**
   * Selected-sequence scalar quality traces. Native-only; not mapped into the
   * canonical transcript contract.
   */
  readonly tokenTraces?: readonly TokenQualityTrace[];
}

export interface WhisperSeq2SeqTranscriptionOptions extends BaseTranscriptionOptions {
  readonly task?: 'transcribe' | 'translate';
  readonly returnSpecialTokens?: boolean;
  readonly returnPromptTokens?: boolean;
  readonly maxNewTokens?: number;
  readonly noTimestamps?: boolean;
  /** Number of beams for beam search (1 = greedy). WhisperX: beam_size */
  readonly numBeams?: number;
  /** Final beam rank penalty. Undefined = length normalization; 0 = raw score. */
  readonly lengthPenalty?: number;
  /** Beam search patience: multiplier for the finished-candidate budget. */
  readonly patience?: number;
  /** Decode temperature. 0 uses greedy/beam argmax; >0 samples from scaled logits and disables beam search. */
  readonly temperature?: number;
  /** Number of independent sampling decodings when temperature > 0. Whisper: best_of. */
  readonly bestOf?: number;
  /**
   * Experimental beam-search optimization. When true, active beam decoder-step
   * calls may be grouped into one ORT batch if the splitgraph model supports
   * batch-shaped decoder_step inputs. If the backend rejects a batch call, the
   * decode retries the active hypotheses through scalar CPU-KV steps. Default
   * false; stable beam remains the correctness oracle.
   */
  readonly experimentalBatchedBeam?: boolean;
  /**
   * Collect scalar logprob/entropy traces for the selected decode sequence.
   * Used by quality gates so beam search does not retain full-vocabulary logits.
   */
  readonly trackQuality?: boolean;
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
   * Optional raw logits callback for Whisper's first decoder position.
   * Fired before timestamp/suppression processing, once per decode attempt.
   */
  readonly onDecoderInitLogits?: (
    rawLogits: Float32Array,
    ctx: {
      readonly tokens: readonly number[];
      readonly beginIndex: number;
      readonly vocabSize: number;
      readonly noSpeechTokenId?: number;
    },
  ) => void;
  /**
   * Extra tokens to append after the standard prompt.
   * Used by EnhancedWhisperExecutor for condition_on_previous_text.
   * Format: [<|0.00|>, ...previous_tokens]
   */
  readonly extraPromptTokens?: readonly number[];
  /**
   * Optional WhisperX-style forced-alignment pass. When set, DTW/interpolated
   * word timestamps are refined after decode. GPU-KV greedy is unchanged unless
   * this aligner is provided.
   */
  readonly wordAligner?: WhisperWordAligner;
  /**
   * Word-timestamp encoder source. `fast` uses the primary inference encoder;
   * `reference` requires the optional alignment-reference artifact; `auto`
   * selects the reference when configured and otherwise uses `fast`.
   */
  readonly wordTimestampSource?: 'auto' | 'fast' | 'reference';
}

export interface WhisperForcedAlignmentWord {
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface WhisperWordAlignerInput {
  readonly transcript: string;
  readonly audio: AudioBufferLike;
  readonly durationSeconds: number;
  readonly language?: string | null;
}

export interface WhisperWordAligner {
  align(
    input: WhisperWordAlignerInput,
  ): Promise<readonly WhisperForcedAlignmentWord[]> | readonly WhisperForcedAlignmentWord[];
}

export interface WhisperSeq2SeqModelDependencies {
  readonly tokenizer?: TextTokenizer;
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly signal?: import('../../types/index.js').AbortSignalLike | null;
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
