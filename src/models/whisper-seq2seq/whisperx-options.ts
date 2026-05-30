/**
 * WhisperX-compatible transcription options.
 *
 * Maps the full WhisperX CLI parameter set into TypeScript.
 * Every CLI flag from WhisperX has a corresponding option here.
 *
 * Reference: https://github.com/m-bain/whisperX
 * Parameters extracted from whisperx/__main__.py (2026-05-30)
 */

// ---------------------------------------------------------------------------
// Model / Device
// ---------------------------------------------------------------------------

export interface WhisperModelOptions {
  /** Name of the Whisper model to use. Default: "large-v3-turbo" */
  readonly model?: string;
  /** Path to save/load model files. Default: ~/.cache/whisper */
  readonly modelDir?: string;
  /** If true, skip download and use cached model only. Default: false */
  readonly modelCacheOnly?: boolean;
  /** Device type: "cuda" | "cpu" | "auto". Default: "auto" */
  readonly device?: string;
  /** Device index for GPU. Default: 0 */
  readonly deviceIndex?: number;
  /** Compute type: "default" | "float16" | "float32" | "int8". Default: "default" */
  readonly computeType?: 'default' | 'float16' | 'float32' | 'int8';
  /** Number of CPU threads. Default: 0 (auto) */
  readonly threads?: number;
  /** Use fp16 inference. Default: true (for GPU) */
  readonly fp16?: boolean;
  /** Batch size for batched encoder. Default: 8 (large task, not yet implemented) */
  readonly batchSize?: number;
}

// ---------------------------------------------------------------------------
// VAD (Voice Activity Detection)
// ---------------------------------------------------------------------------

export interface WhisperVadOptions {
  /** VAD method. "silero" | "pyannote". We use "ten-vad" | "firered-vad". Default: "ten-vad" */
  readonly vadMethod?: string;
  /** Onset threshold for VAD (0-1). Lower = more sensitive. Default: 0.500 */
  readonly vadOnset?: number;
  /** Offset threshold for VAD. Default: 0.363 */
  readonly vadOffset?: number;
  /** Chunk size for merging VAD segments (seconds). Default: 30 */
  readonly chunkSize?: number;
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

export interface WhisperDecodingOptions {
  /** Initial temperature. Default: 0 */
  readonly temperature?: number;
  /** Temperature increment on fallback. Default: 0.2 */
  readonly temperatureIncrementOnFallback?: number;
  /** Number of independent decodings (best-of). Default: 5 */
  readonly bestOf?: number;
  /** Beam size for beam search. Default: 5 */
  readonly beamSize?: number;
  /** Beam search patience. Default: 1.0 */
  readonly patience?: number;
  /** Length penalty. Default: 1.0 */
  readonly lengthPenalty?: number;
  /** Comma-separated token IDs to suppress. Default: "-1" (suppress special chars) */
  readonly suppressTokens?: string;
  /** Suppress numeric symbols and currency symbols. Default: false */
  readonly suppressNumerals?: boolean;
}

// ---------------------------------------------------------------------------
// Quality gates
// ---------------------------------------------------------------------------

export interface WhisperQualityOptions {
  /** Compression ratio threshold. Default: 2.4 */
  readonly compressionRatioThreshold?: number;
  /** Average log-probability threshold. Default: -1.0 */
  readonly logprobThreshold?: number;
  /** No-speech token probability threshold. Default: 0.6 */
  readonly noSpeechThreshold?: number;
  /** Entropy threshold (nats). Default: 2.4 */
  readonly entropyThreshold?: number;
}

// ---------------------------------------------------------------------------
// Context / Prompt
// ---------------------------------------------------------------------------

export interface WhisperContextOptions {
  /** Initial prompt text for the first window. Default: null */
  readonly initialPrompt?: string | null;
  /** Hotwords / hint phrases (e.g. "WhisperX, PyAnnoted, GPU"). Default: null */
  readonly hotwords?: string | null;
  /** Condition on previous text across windows. Default: false */
  readonly conditionOnPreviousText?: boolean;
}

// ---------------------------------------------------------------------------
// Task / Language
// ---------------------------------------------------------------------------

export interface WhisperTaskOptions {
  /** Task: "transcribe" | "translate". Default: "transcribe" */
  readonly task?: 'transcribe' | 'translate';
  /** Language code (e.g. "en", "tr"). Default: null (auto-detect) */
  readonly language?: string | null;
}

// ---------------------------------------------------------------------------
// Alignment
// ---------------------------------------------------------------------------

export interface WhisperAlignmentOptions {
  /** Phoneme-level alignment model name. Default: null (use default) */
  readonly alignModel?: string | null;
  /** Interpolation method for unaligned words. "nearest" | "linear" | "ignore". Default: "nearest" */
  readonly interpolateMethod?: 'nearest' | 'linear' | 'ignore';
  /** Skip alignment. Default: false */
  readonly noAlign?: boolean;
  /** Return character-level alignments in output. Default: false */
  readonly returnCharAlignments?: boolean;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

export interface WhisperOutputOptions {
  /** Output directory. Default: "." */
  readonly outputDir?: string;
  /** Output format(s). "all" | "srt" | "vtt" | "txt" | "tsv" | "json" | "aud". Default: "all" */
  readonly outputFormat?: string;
  /** Max characters per line. Default: null */
  readonly maxLineWidth?: number | null;
  /** Max lines per segment. Default: null */
  readonly maxLineCount?: number | null;
  /** Highlight words in SRT/VTT. Default: false */
  readonly highlightWords?: boolean;
  /** Segment resolution: "sentence" | "chunk". Default: "sentence" */
  readonly segmentResolution?: 'sentence' | 'chunk';
  /** Verbose output. Default: true */
  readonly verbose?: boolean;
  /** Log level override (debug/info/warning/error/critical). Default: null */
  readonly logLevel?: string | null;
}

// ---------------------------------------------------------------------------
// Diarization
// ---------------------------------------------------------------------------

export interface WhisperDiarizationOptions {
  /** Apply speaker diarization. Default: false */
  readonly diarize?: boolean;
  /** Minimum speakers. Default: null */
  readonly minSpeakers?: number | null;
  /** Maximum speakers. Default: null */
  readonly maxSpeakers?: number | null;
  /** Speaker diarization model name. Default: "pyannote/speaker-diarization-community-1" */
  readonly diarizeModel?: string;
  /** Include speaker embeddings in JSON output. Default: false */
  readonly speakerEmbeddings?: boolean;
}

// ---------------------------------------------------------------------------
// Other
// ---------------------------------------------------------------------------

export interface WhisperOtherOptions {
  /** Hugging Face access token for gated models. Default: null */
  readonly hfToken?: string | null;
  /** Print progress during transcribe/align. Default: false */
  readonly printProgress?: boolean;
}

// ---------------------------------------------------------------------------
// Full unified options
// ---------------------------------------------------------------------------

export interface WhisperXOptions {
  readonly model?: WhisperModelOptions;
  readonly vad?: WhisperVadOptions;
  readonly decoding?: WhisperDecodingOptions;
  readonly quality?: WhisperQualityOptions;
  readonly context?: WhisperContextOptions;
  readonly task?: WhisperTaskOptions;
  readonly alignment?: WhisperAlignmentOptions;
  readonly output?: WhisperOutputOptions;
  readonly diarization?: WhisperDiarizationOptions;
  readonly other?: WhisperOtherOptions;
}

// ---------------------------------------------------------------------------
// Defaults matching WhisperX CLI
// ---------------------------------------------------------------------------

export const WHISPERX_DEFAULTS: Required<WhisperModelOptions & WhisperVadOptions &
  WhisperDecodingOptions & WhisperQualityOptions & WhisperContextOptions &
  WhisperTaskOptions & WhisperAlignmentOptions & WhisperOutputOptions &
  WhisperDiarizationOptions & WhisperOtherOptions> = {
  // Model
  model: 'large-v3-turbo',
  modelDir: '',
  modelCacheOnly: false,
  device: 'auto',
  deviceIndex: 0,
  computeType: 'default',
  threads: 0,
  fp16: true,
  batchSize: 8,
  // VAD
  vadMethod: 'ten-vad',
  vadOnset: 0.5,
  vadOffset: 0.363,
  chunkSize: 30,
  // Decoding
  temperature: 0,
  temperatureIncrementOnFallback: 0.2,
  bestOf: 5,
  beamSize: 5,
  patience: 1.0,
  lengthPenalty: 1.0,
  suppressTokens: '-1',
  suppressNumerals: false,
  // Quality
  compressionRatioThreshold: 2.4,
  logprobThreshold: -1.0,
  noSpeechThreshold: 0.6,
  entropyThreshold: 2.4,
  // Context
  initialPrompt: null,
  hotwords: null,
  conditionOnPreviousText: false,
  // Task
  task: 'transcribe',
  language: null,
  // Alignment
  alignModel: null,
  interpolateMethod: 'nearest',
  noAlign: false,
  returnCharAlignments: false,
  // Output
  outputDir: '.',
  outputFormat: 'all',
  maxLineWidth: null,
  maxLineCount: null,
  highlightWords: false,
  segmentResolution: 'sentence',
  verbose: true,
  logLevel: null,
  // Diarization
  diarize: false,
  minSpeakers: null,
  maxSpeakers: null,
  diarizeModel: 'pyannote/speaker-diarization-community-1',
  speakerEmbeddings: false,
  // Other
  hfToken: null,
  printProgress: false,
};
