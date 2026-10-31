import type { TranscriptWarning } from '../../types/index.js';
import type { WhisperKvCache } from './core.js';
import type { WhisperGenerationConfig, WhisperModelConfig } from './generation-config.js';
import type {
  WhisperAlignmentAttentionLayout,
  WhisperAlignmentAttentionValues,
} from './manifest.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from './ort.js';
import type { WhisperTokenizer } from './tokenizer.js';
import type { WhisperNativeToken } from './types.js';

export interface LoadedExecutorState {
  readonly ort: OrtModuleLike;
  readonly tokenizer: WhisperTokenizer;
  readonly encoderSession: OrtSessionLike;
  readonly decoderSession?: OrtSessionLike;
  readonly generationConfig: WhisperGenerationConfig;
  readonly modelConfig: WhisperModelConfig;
  readonly warnings: readonly TranscriptWarning[];
  readonly isSplitGraph: boolean;
  readonly decoderInitSession?: OrtSessionLike;
  readonly decoderStepSession?: OrtSessionLike;
  readonly decoderAlignSession?: OrtSessionLike;
  readonly decoderAlignUrl?: string;
  /** True only when manifest metadata proves decoder_align is causal. */
  readonly decoderAlignCausalSelfAttention?: boolean;
  /** Alignment value semantics declared by the decoder_align manifest. */
  readonly decoderAlignAttentionValues?: WhisperAlignmentAttentionValues;
  /** Alignment tensor layout declared by the decoder_align manifest. */
  readonly decoderAlignAttentionLayout?: WhisperAlignmentAttentionLayout;
  readonly enableProfiling?: boolean;
  readonly decoderAlignExternalData?: readonly { readonly dataUrl: string; readonly path: string }[];
  readonly alignmentReference?: {
    readonly encoderUrl: string;
    readonly decoderAlignUrl: string;
    readonly manifestUrl?: string;
    readonly backendForOrt: string;
    readonly encoderExternalData?: readonly { readonly dataUrl: string; readonly path: string }[];
    readonly decoderAlignExternalData?: readonly { readonly dataUrl: string; readonly path: string }[];
    readonly decoderAlignCausalSelfAttention?: boolean;
    readonly decoderAlignAttentionValues?: WhisperAlignmentAttentionValues;
    readonly decoderAlignAttentionLayout?: WhisperAlignmentAttentionLayout;
  };
  readonly decoderBackendForOrt?: string;
  readonly experimentalGpuKvCache?: boolean;
  readonly experimentalGpuKvBeam?: boolean;
  readonly sessionCreateMs?: number;
  /** DIAGNOSTIC (Edge A): Re-wrap encoder GPU buffer as fresh tensor. */
  readonly encoderBufferRewrap?: boolean;
  /** DIAGNOSTIC (Edge B2): Force GPU flush before decoder_init. */
  readonly encoderGpuFlush?: boolean;
  /** PROFILING (encoderGpuDrain): Force GPU drain + re-wrap after encoder. */
  readonly encoderGpuDrain?: boolean;
}

export interface WhisperAlignmentRuntime {
  readonly decoderAlignSession: OrtSessionLike;
  readonly decoderAlignCausalSelfAttention?: boolean;
  readonly decoderAlignAttentionValues?: WhisperAlignmentAttentionValues;
  readonly decoderAlignAttentionLayout?: WhisperAlignmentAttentionLayout;
  readonly decoderAlignExternalData?: readonly { readonly dataUrl: string; readonly path: string }[];
}

export interface DecoderStepResult {
  readonly lastLogits: Float32Array;
  readonly vocabSize: number;
  readonly pastKeyValues: Record<string, OrtTensorLike<Float32Array>>;
  readonly crossAttentions: readonly OrtTensorLike<Float32Array>[];
}

export interface BeamPayload {
  readonly tokenDetails: readonly WhisperNativeToken[];
  readonly pastKeyValues: Record<string, OrtTensorLike<Float32Array>>;
  readonly tokenTraces?: readonly {
    readonly tokenId: number;
    readonly logProb: number;
    readonly entropy: number;
  }[];
}

export interface SplitGraphDecodeCallbacks {
  runInit(
    promptTokens: readonly number[],
    encoderHiddenStates: Float32Array,
    encoderDims: readonly number[],
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: WhisperKvCache }>;
  runStep(
    tokenId: number,
    pastKv: WhisperKvCache,
  ): Promise<{ logits: Float32Array; vocabSize: number; presentKv: WhisperKvCache }>;
  runStepBatch?(
    tokenIds: readonly number[],
    pastKvs: readonly WhisperKvCache[],
  ): Promise<
    readonly { logits: Float32Array; vocabSize: number; presentKv: WhisperKvCache }[] | undefined
  >;
}

export interface SplitGraphDecodeResult {
  readonly tokens: readonly number[];
  readonly tokenTraces?: readonly {
    readonly tokenId: number;
    readonly logProb: number;
    readonly entropy: number;
  }[];
}

export interface DecoderSessionTiming {
  readonly inputMs: number;
  readonly runMs: number;
  readonly outputMs: number;
  readonly gpuInputCount: number;
  readonly cpuInputCount: number;
  readonly gpuOutputCount: number;
  readonly cpuOutputCount: number;
  readonly gpuDownloadCount: number;
  // Profiling sub-buckets (added by runDecoderInit / runDecoderStepSplit)
  readonly tensorCreateMs?: number;
  readonly logitReadMs?: number;
  readonly kvExtractMs?: number;
}

export interface TensorLocationCounts {
  readonly gpu: number;
  readonly cpu: number;
}

export interface WhisperEncoderKvTensorCacheEntry {
  readonly tensor: OrtTensorLike<Float32Array>;
  /** First raw cache view observed before the ORT input copy. */
  readonly sourceData: ArrayBufferView;
  /** Stable copied view handed back to the core for subsequent steps. */
  readonly stableData: ArrayBufferView;
  readonly dims: readonly number[];
  readonly type: 'float32' | 'float16';
}

export type WhisperPresentKvSnapshot = {
  readonly data: ArrayBufferView;
  readonly dims: readonly number[];
  readonly type?: string;
};

export interface SplitGraphAlignmentOptions {
  readonly alignmentData: Float32Array;
  readonly totalTokens: number;
  readonly promptLen: number;
  readonly textTokenCount: number;
  readonly frameCount: number;
  readonly medianFilterWidth?: number;
  readonly timePrecisionSeconds?: number;
  /** Explicit alignment-matrix rows for text tokens. Use this when timestamp
   *  or other special tokens sit between prompt and text tokens. */
  readonly textTokenRowIndices?: readonly number[];
  /** Crop DTW to this many encoder frames (e.g. actual audio duration / 0.02).
   *  The alignment matrix may still be padded to the 30s Whisper window. */
  readonly cropFrameCount?: number;
  /** Number of selected-head matrices in a [B, N, T, S] alignment output. */
  readonly alignmentHeadCount?: number;
  /** True when alignmentData contains raw cross-attention logits. */
  readonly alignmentValuesAreLogits?: boolean;
}

export interface SplitGraphTimestampSpanAlignmentOptions {
  readonly alignmentData: Float32Array;
  readonly tokenIds: readonly number[];
  readonly promptLen: number;
  readonly frameCount: number;
  readonly medianFilterWidth?: number;
  readonly timePrecisionSeconds?: number;
  readonly cropFrameCount?: number;
  readonly isTextToken: (tokenId: number) => boolean;
  readonly isTimestampToken: (tokenId: number) => boolean;
  readonly timestampTokenToSeconds: (tokenId: number) => number;
  /** Number of selected-head matrices in a [B, N, T, S] alignment output. */
  readonly alignmentHeadCount?: number;
  /** True when alignmentData contains raw cross-attention logits. */
  readonly alignmentValuesAreLogits?: boolean;
}
