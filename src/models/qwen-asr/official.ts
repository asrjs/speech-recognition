import { OFFICIAL_QWEN3_ASR_GRAPH_DEFAULTS } from './config.js';
import type {
  Qwen3AsrArtifactSource,
  Qwen3AsrDirectArtifacts,
  Qwen3AsrModelConfig,
} from './types.js';

export const OFFICIAL_QWEN3_ASR_ENCODER_DYNAMIC = 'audio-encoder-dynamic.onnx';
export const OFFICIAL_QWEN3_ASR_ENCODER_STATIC_T1100 = 'audio-encoder-static-t1100.onnx';

export type OfficialQwen3AsrEncoderVariant = 'dynamic' | 'static-t1100';
export type OfficialQwen3AsrDecoderDtype = 'float16' | 'float32';

export function parseOfficialQwen3AsrEncoderVariant(
  value: string | null | undefined,
): OfficialQwen3AsrEncoderVariant {
  return value === 'static-t1100' ? 'static-t1100' : 'dynamic';
}

export function officialQwen3AsrEncoderFilename(
  variant: OfficialQwen3AsrEncoderVariant = 'dynamic',
): string {
  return variant === 'static-t1100'
    ? OFFICIAL_QWEN3_ASR_ENCODER_STATIC_T1100
    : OFFICIAL_QWEN3_ASR_ENCODER_DYNAMIC;
}

export function isOfficialQwen3AsrStackedSource(
  source: Qwen3AsrArtifactSource | undefined,
): boolean {
  return source?.kind === 'direct' && Boolean(source.artifacts.decoderStepUrl);
}

export function applyOfficialQwen3AsrGraphDefaults(
  config: Qwen3AsrModelConfig,
  source?: Qwen3AsrArtifactSource,
): Qwen3AsrModelConfig {
  if (!isOfficialQwen3AsrStackedSource(source)) return config;
  return {
    ...config,
    graph: {
      ...config.graph,
      ...OFFICIAL_QWEN3_ASR_GRAPH_DEFAULTS,
    },
  };
}

function joinOfficialUrl(baseUrl: string, filename: string): string {
  const base = baseUrl.replace(/\/+$/, '');
  const name = filename.replace(/^\/+/, '');
  return `${base}/${name}`;
}

/**
 * Artifact URLs for the official stacked Qwen3-ASR graph.
 *
 * The default encoder is `audio-encoder-dynamic.onnx` (T % 100 == 0 after JS
 * pad-to-chunk). Pass `encoder: 'static-t1100'` only when loading that explicit
 * optional graph. This is not a hosted-weight preset.
 */
export function resolveOfficialQwen3AsrDirectArtifacts(options: {
  readonly baseUrl: string;
  readonly encoder?: OfficialQwen3AsrEncoderVariant;
  readonly dtype?: OfficialQwen3AsrDecoderDtype;
}): Qwen3AsrDirectArtifacts {
  const file = (filename: string): string => joinOfficialUrl(options.baseUrl, filename);
  const encoderUrl = file(officialQwen3AsrEncoderFilename(options.encoder ?? 'dynamic'));
  const dtype = options.dtype ?? 'float32';
  if (dtype === 'float16') {
    return {
      encoderUrl,
      decoderUrl: file('decoder-prefill-fp16.onnx'),
      decoderStepUrl: file('decoder-step-fp16.onnx'),
      tokenizerUrl: file('tokenizer/tokenizer.json'),
      decoderPrefillDataUrl: file('decoder-fp16.onnx.data'),
      decoderPrefillDataPath: 'decoder-fp16.onnx.data',
      decoderStepDataUrl: file('decoder-fp16.onnx.data'),
      decoderStepDataPath: 'decoder-fp16.onnx.data',
    };
  }
  return {
    encoderUrl,
    decoderUrl: file('decoder-prefill.onnx'),
    decoderStepUrl: file('decoder-step.onnx'),
    tokenizerUrl: file('tokenizer/tokenizer.json'),
    decoderPrefillDataUrl: file('decoder-prefill.onnx.data'),
    decoderPrefillDataPath: 'decoder-prefill.onnx.data',
    decoderStepDataUrl: file('decoder-step.onnx.data'),
    decoderStepDataPath: 'decoder-step.onnx.data',
  };
}
