import type { TokenizerSpec } from '../../tokenizers/index.js';
import type {
  AssetProvider,
  AudioBufferLike,
  BaseTranscriptionOptions,
  SpeechRuntimeHooks,
  TranscriptWarning,
} from '../../types/index.js';

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

export interface Wav2Vec2ModelConfig {
  readonly ecosystem: 'meta';
  readonly architecture: 'wav2vec2';
  readonly processorArchitecture: 'wav2vec2-conv';
  readonly encoderArchitecture: 'wav2vec2-conformer';
  readonly decoderArchitecture: 'ctc';
  readonly sampleRate: number;
  /** Total convolution stride (product of all conv layer strides). */
  readonly outputStride: number;
  /** Number of convolution feature extraction layers. */
  readonly numFeatExtractLayers: number;
  /** Conv layer output channels (same for all layers in base model). */
  readonly convDim: number;
  /** Conv kernel sizes per layer. */
  readonly convKernel: readonly number[];
  /** Conv stride per layer. */
  readonly convStride: readonly number[];
  /** Hidden size of the transformer encoder. */
  readonly hiddenSize: number;
  /** Number of transformer layers. */
  readonly numHiddenLayers: number;
  /** Number of attention heads per layer. */
  readonly numAttentionHeads: number;
  /** CTC vocabulary size (including blank). */
  readonly vocabularySize: number;
  /** CTC blank token ID (usually 0 = pad). */
  readonly ctcBlankId: number;
  /** Languages supported by this model. */
  readonly languages: readonly string[];
  /** Tokenizer specification. */
  readonly tokenizer: TokenizerSpec;
  /** Whether layer norm is applied before (true) or after (false) attention. */
  readonly doStableLayerNorm: boolean;
  /** Feature projection layer norm epsilon. */
  readonly layerNormEps: number;
  /** Feature extraction activation. */
  readonly featExtractActivation: string;
  /** Whether conv layers use bias. */
  readonly convBias: boolean;
  /** Feature extraction normalization: 'group' or 'layer'. */
  readonly featExtractNorm: 'group' | 'layer';
}

// ---------------------------------------------------------------------------
// Artifact sources
// ---------------------------------------------------------------------------

export interface Wav2Vec2DirectArtifacts {
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly modelDataUrl?: string;
  readonly modelDataFilename?: string;
}

export interface Wav2Vec2DirectArtifactSource {
  readonly kind: 'direct';
  readonly artifacts: Wav2Vec2DirectArtifacts;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export interface Wav2Vec2HuggingFaceSource {
  readonly kind: 'huggingface';
  readonly repoId: string;
  readonly revision?: string;
  readonly subfolder?: string;
  readonly modelFilename?: string;
  readonly modelDataFilename?: string;
  readonly tokenizerFilename?: string;
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
}

export type Wav2Vec2ArtifactSource = Wav2Vec2DirectArtifactSource | Wav2Vec2HuggingFaceSource;

// ---------------------------------------------------------------------------
// Model options
// ---------------------------------------------------------------------------

export interface Wav2Vec2ModelOptions {
  readonly modelBaseUrl?: string;
  readonly revision?: string;
  readonly config?: Partial<Wav2Vec2ModelConfig>;
  readonly source?: Wav2Vec2ArtifactSource;
}

// ---------------------------------------------------------------------------
// Transcription options
// ---------------------------------------------------------------------------

export interface Wav2Vec2TranscriptionOptions extends BaseTranscriptionOptions {
  readonly returnTokenIds?: boolean;
  readonly returnConfidence?: boolean;
}

// ---------------------------------------------------------------------------
// Native output types (CTC-specific)
// ---------------------------------------------------------------------------

export interface Wav2Vec2NativeToken {
  readonly index: number;
  readonly id?: number;
  readonly text: string;
  readonly startTime?: number;
  readonly endTime?: number;
  readonly confidence?: number;
}

export interface Wav2Vec2NativeSegment {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
}

export interface Wav2Vec2NativeWord {
  readonly index: number;
  readonly text: string;
  readonly startTime: number;
  readonly endTime: number;
  readonly confidence?: number;
  readonly tokenIds?: readonly number[];
  readonly tokenIndices?: readonly number[];
}

export interface Wav2Vec2NativeTranscript {
  readonly utteranceText: string;
  readonly isFinal: boolean;
  readonly language?: string;
  readonly segments?: readonly Wav2Vec2NativeSegment[];
  readonly words?: readonly Wav2Vec2NativeWord[];
  readonly tokens?: readonly Wav2Vec2NativeToken[];
  readonly warnings?: readonly { readonly code: string; readonly message: string }[];
}

// ---------------------------------------------------------------------------
// Executor interface
// ---------------------------------------------------------------------------

export interface Wav2Vec2TokenizerLike {
  encode(text: string): number[];
  decode(ids: readonly number[]): string;
  decodeTokenPiece?(tokenId: number): string;
}

export interface Wav2Vec2LogitsResult {
  readonly logits: Float32Array;
  readonly frameCount: number;
  readonly vocabSize: number;
  readonly sampleRate: number;
  readonly audioDurationSeconds: number;
  readonly blankId: number;
  readonly tokenizer: Wav2Vec2TokenizerLike;
  readonly warnings?: readonly TranscriptWarning[];
  readonly encodeMs?: number;
}

export interface Wav2Vec2Executor {
  ready?(): Promise<void> | void;
  extractLogits(
    audio: AudioBufferLike,
    options?: Wav2Vec2TranscriptionOptions,
  ): Promise<Wav2Vec2LogitsResult>;
  transcribe(
    audio: AudioBufferLike,
    options: Wav2Vec2TranscriptionOptions,
  ): Promise<Wav2Vec2NativeTranscript>;
  dispose(): Promise<void> | void;
}

// ---------------------------------------------------------------------------
// Model dependencies
// ---------------------------------------------------------------------------

export interface Wav2Vec2ModelDependencies {
  readonly assetProvider?: AssetProvider;
  readonly runtimeHooks?: SpeechRuntimeHooks;
  readonly executor?: Wav2Vec2Executor;
}

// ---------------------------------------------------------------------------
// Token span with timing (CTC output)
// ---------------------------------------------------------------------------

export interface Wav2Vec2TokenSpan {
  readonly tokenId: number;
  readonly text: string;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly frameCount: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
  readonly averageLogProb: number;
}

export interface Wav2Vec2SentenceTiming {
  readonly text: string;
  readonly startTokenIndex: number;
  readonly endTokenIndex: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
}

export interface Wav2Vec2UtteranceTiming {
  readonly hasSpeech: boolean;
  readonly startFrame: number | null;
  readonly endFrame: number | null;
  readonly startTime: number;
  readonly endTime: number;
  readonly duration: number;
  readonly confidence: number;
}
