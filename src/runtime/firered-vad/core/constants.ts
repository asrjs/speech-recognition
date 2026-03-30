export const SAMPLE_RATE = 16000;
export const FRAME_LENGTH_MS = 25;
export const FRAME_SHIFT_MS = 10;
export const FRAME_LENGTH_S = 0.025;
export const FRAME_SHIFT_S = 0.01;
export const FRAME_LENGTH_SAMPLE = Math.floor((SAMPLE_RATE * FRAME_LENGTH_MS) / 1000);
export const FRAME_SHIFT_SAMPLE = Math.floor((SAMPLE_RATE * FRAME_SHIFT_MS) / 1000);
export const FRAME_PER_SECONDS = Math.floor(1000 / FRAME_SHIFT_MS);
export const FEATURE_DIM = 80;

export const STREAM_CACHE_LAYERS = 8;
export const STREAM_CACHE_BATCH = 1;
export const STREAM_CACHE_PROJ = 128;
export const STREAM_CACHE_LEN = 19;

export const FIRERED_GITHUB_RAW_ONNX_BASE =
  'https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/pretrained_models/onnx_models/';

export const DEFAULT_MODEL_URLS = {
  vadUrl: `${FIRERED_GITHUB_RAW_ONNX_BASE}fireredvad_vad.onnx`,
  streamVadWithCacheUrl:
    `${FIRERED_GITHUB_RAW_ONNX_BASE}fireredvad_stream_vad_with_cache.onnx`,
  aedUrl: `${FIRERED_GITHUB_RAW_ONNX_BASE}fireredvad_aed.onnx`,
} as const;
