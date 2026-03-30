export interface FireRedModelUrls {
  readonly vadUrl?: string;
  readonly streamVadWithCacheUrl?: string;
  readonly aedUrl?: string;
  readonly cmvnJsonUrl?: string;
}

export interface CmvnStats {
  readonly means: Float32Array;
  readonly istd: Float32Array;
}

export interface FireRedRuntimeOptions {
  readonly modelUrls?: FireRedModelUrls;
  readonly wasmPaths?: string | Record<string, string>;
  readonly wasmNumThreads?: number;
  readonly cacheAssets?: boolean;
  readonly backend?: 'wasm';
  readonly cmvn?: {
    readonly means: ArrayLike<number>;
    readonly istd: ArrayLike<number>;
  };
}

export interface FireRedFrameResult {
  readonly confidence: number;
  readonly is_speech: boolean;
  readonly isSpeech: boolean;
  readonly frame_offset: number;
  readonly frameOffset: number;
}

export interface StreamVadFrameResult {
  readonly frame_idx: number;
  readonly frameIdx: number;
  readonly is_speech: boolean;
  readonly isSpeech: boolean;
  readonly raw_prob: number;
  readonly rawProb: number;
  readonly smoothed_prob: number;
  readonly smoothedProb: number;
  readonly is_speech_start: boolean;
  readonly isSpeechStart: boolean;
  readonly is_speech_end: boolean;
  readonly isSpeechEnd: boolean;
  readonly speech_start_frame: number;
  readonly speechStartFrame: number;
  readonly speech_end_frame: number;
  readonly speechEndFrame: number;
}

export interface FireRedVadDetectResult {
  readonly dur: number;
  readonly timestamps: Array<[number, number]>;
  readonly wav_path?: string;
  readonly wavPath?: string;
}

export interface FireRedAedDetectResult {
  readonly dur: number;
  readonly event2timestamps: Record<string, Array<[number, number]>>;
  readonly event2ratio: Record<string, number>;
  readonly wav_path?: string;
  readonly wavPath?: string;
}

export interface FireRedVadConfig extends FireRedRuntimeOptions {
  readonly use_gpu?: boolean;
  readonly useGpu?: boolean;
  readonly smooth_window_size?: number;
  readonly smoothWindowSize?: number;
  readonly speech_threshold?: number;
  readonly speechThreshold?: number;
  readonly min_speech_frame?: number;
  readonly minSpeechFrame?: number;
  readonly max_speech_frame?: number;
  readonly maxSpeechFrame?: number;
  readonly min_silence_frame?: number;
  readonly minSilenceFrame?: number;
  readonly merge_silence_frame?: number;
  readonly mergeSilenceFrame?: number;
  readonly extend_speech_frame?: number;
  readonly extendSpeechFrame?: number;
  readonly chunk_max_frame?: number;
  readonly chunkMaxFrame?: number;
}

export interface FireRedStreamVadConfig extends FireRedRuntimeOptions {
  readonly use_gpu?: boolean;
  readonly useGpu?: boolean;
  readonly smooth_window_size?: number;
  readonly smoothWindowSize?: number;
  readonly speech_threshold?: number;
  readonly speechThreshold?: number;
  readonly pad_start_frame?: number;
  readonly padStartFrame?: number;
  readonly min_speech_frame?: number;
  readonly minSpeechFrame?: number;
  readonly max_speech_frame?: number;
  readonly maxSpeechFrame?: number;
  readonly min_silence_frame?: number;
  readonly minSilenceFrame?: number;
  readonly chunk_max_frame?: number;
  readonly chunkMaxFrame?: number;
}

export interface FireRedAedConfig extends FireRedRuntimeOptions {
  readonly use_gpu?: boolean;
  readonly useGpu?: boolean;
  readonly smooth_window_size?: number;
  readonly smoothWindowSize?: number;
  readonly speech_threshold?: number;
  readonly speechThreshold?: number;
  readonly singing_threshold?: number;
  readonly singingThreshold?: number;
  readonly music_threshold?: number;
  readonly musicThreshold?: number;
  readonly min_event_frame?: number;
  readonly minEventFrame?: number;
  readonly max_event_frame?: number;
  readonly maxEventFrame?: number;
  readonly min_silence_frame?: number;
  readonly minSilenceFrame?: number;
  readonly merge_silence_frame?: number;
  readonly mergeSilenceFrame?: number;
  readonly extend_speech_frame?: number;
  readonly extendSpeechFrame?: number;
  readonly chunk_max_frame?: number;
  readonly chunkMaxFrame?: number;
}

export interface NormalizedFireRedVadConfig extends FireRedRuntimeOptions {
  use_gpu: boolean;
  smooth_window_size: number;
  speech_threshold: number;
  min_speech_frame: number;
  max_speech_frame: number;
  min_silence_frame: number;
  merge_silence_frame: number;
  extend_speech_frame: number;
  chunk_max_frame: number;
}

export interface NormalizedFireRedStreamVadConfig extends FireRedRuntimeOptions {
  use_gpu: boolean;
  smooth_window_size: number;
  speech_threshold: number;
  pad_start_frame: number;
  min_speech_frame: number;
  max_speech_frame: number;
  min_silence_frame: number;
  chunk_max_frame: number;
}

export interface NormalizedFireRedAedConfig extends FireRedRuntimeOptions {
  use_gpu: boolean;
  smooth_window_size: number;
  speech_threshold: number;
  singing_threshold: number;
  music_threshold: number;
  min_event_frame: number;
  max_event_frame: number;
  min_silence_frame: number;
  merge_silence_frame: number;
  extend_speech_frame: number;
  chunk_max_frame: number;
}

export interface FireredVadStreamPackedCreateOptions extends FireRedRuntimeOptions {
  readonly modelUrl?: string;
  readonly cmvnJsonUrl?: string;
  readonly threshold?: number;
}

export interface FireRedAssetCacheValue {
  readonly bytes: Uint8Array;
  readonly contentType?: string;
}

export interface FireRedAssetCache {
  get(key: string): Promise<FireRedAssetCacheValue | null>;
  set(key: string, value: FireRedAssetCacheValue): Promise<void>;
}
