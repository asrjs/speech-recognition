import type { WhisperTokenizer } from './tokenizer.js';
import type { OrtModuleLike, OrtSessionLike, OrtTensorLike } from './ort.js';

function isGpuBufferTensor(tensor: OrtTensorLike): boolean {
  return tensor.location === 'gpu-buffer';
}

export function resolveWhisperLanguageCode(
  requestedLanguage: string | undefined,
  configuredLanguages: readonly string[] | undefined,
): string {
  if (requestedLanguage && requestedLanguage !== 'auto') return requestedLanguage;
  return configuredLanguages?.find((language) => language !== 'auto') ?? 'en';
}

export function resolveWhisperLanguageTokenId(
  tokenizer: Pick<WhisperTokenizer, 'getTokenId'>,
  language: string,
): number {
  return tokenizer.getTokenId(`<|${language}|>`)
    ?? tokenizer.getTokenId('<|en|>')
    ?? 50259;
}

export function buildWhisperForcedAlignmentTokenIds(
  tokenizer: Pick<WhisperTokenizer, 'getTokenId'>,
  language: string,
  textTokenIds: readonly number[],
  task: 'transcribe' | 'translate' = 'transcribe',
  noTimestampsTokenId?: number,
): number[] {
  const taskToken = task === 'translate' ? '<|translate|>' : '<|transcribe|>';
  const fallbackTaskId = task === 'translate' ? 50359 : 50360;
  return [
    tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
    resolveWhisperLanguageTokenId(tokenizer, language),
    tokenizer.getTokenId(taskToken) ?? fallbackTaskId,
    tokenizer.getTokenId('<|notimestamps|>') ?? noTimestampsTokenId ?? 50363,
    ...textTokenIds,
    tokenizer.getTokenId('<|endoftext|>') ?? 50257,
  ];
}

export function getWhisperForcedAlignmentTextRowStart(promptLength: number): number {
  return Math.max(0, Math.floor(promptLength) - 1);
}

export function float32ToFloat16Bits(src: Float32Array): Uint16Array {
  const dst = new Uint16Array(src.length);
  const sourceBits = new Uint32Array(src.buffer, src.byteOffset, src.length);
  for (let i = 0; i < src.length; i++) {
    const uint32 = sourceBits[i]!;
    const sign = (uint32 >>> 16) & 0x8000;
    const exponent = (uint32 >>> 23) & 0xff;
    const mantissa = uint32 & 0x7fffff;

    if (exponent === 0xff) {
      dst[i] = sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
      continue;
    }

    let halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) {
      dst[i] = sign | 0x7c00;
      continue;
    }

    if (halfExponent <= 0) {
      if (halfExponent < -10) {
        dst[i] = sign;
        continue;
      }
      const significand = mantissa | 0x800000;
      const shift = 14 - halfExponent;
      let halfMantissa = significand >>> shift;
      const remainder = significand & ((1 << shift) - 1);
      const halfway = 1 << (shift - 1);
      if (remainder > halfway || (remainder === halfway && (halfMantissa & 1) !== 0)) {
        halfMantissa += 1;
      }
      dst[i] = sign | halfMantissa;
      continue;
    }

    let halfMantissa = mantissa >>> 13;
    const remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
      halfMantissa += 1;
      if (halfMantissa === 0x400) {
        halfMantissa = 0;
        halfExponent += 1;
      }
    }
    dst[i] = halfExponent >= 0x1f
      ? sign | 0x7c00
      : sign | (halfExponent << 10) | halfMantissa;
  }
  return dst;
}

export function float16BitsToFloat32(src: Uint16Array): Float32Array {
  const dst = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const bits = src[i]!;
    const sign = (bits >>> 15) & 0x1;
    const exp = (bits >>> 10) & 0x1f;
    const mant = bits & 0x3ff;
    let value: number;
    if (exp === 0) {
      value = mant === 0 ? 0 : (mant / 1024) * 2 ** -14;
    } else if (exp === 0x1f) {
      value = mant === 0 ? Infinity : Number.NaN;
    } else {
      value = (1 + mant / 1024) * 2 ** (exp - 15);
    }
    dst[i] = sign ? -value : value;
  }
  return dst;
}

export function tensorDataAsFloat32(data: ArrayBufferView): Float32Array {
  if (data instanceof Float32Array) return data;
  if (data instanceof Uint16Array) return float16BitsToFloat32(data);
  return Float32Array.from(data as unknown as ArrayLike<number>);
}

export async function maybeCastEncoderHiddenStates(
  encoderHiddenStates: OrtTensorLike<Float32Array>,
  decoderSession: OrtSessionLike,
  ort: OrtModuleLike,
): Promise<OrtTensorLike<Float32Array>> {
  const metadata = (decoderSession as unknown as { inputMetadata?: Array<{ name?: string; type?: string; shape?: number[] }> }).inputMetadata;
  const encMeta = metadata?.find((m) => m.name === 'encoder_hidden_states');
  const expectedType = encMeta?.type;
  if (expectedType !== 'float16' && expectedType !== 'float32') {
    return encoderHiddenStates;
  }
  if (encoderHiddenStates.type === expectedType) {
    return encoderHiddenStates;
  }
  const rawData = isGpuBufferTensor(encoderHiddenStates) && encoderHiddenStates.getData
    ? await encoderHiddenStates.getData(true)
    : encoderHiddenStates.data;
  const dims = encoderHiddenStates.dims as number[];
  const size = dims.reduce((a, b) => a * b, 1);
  const f32Data = tensorDataAsFloat32(rawData as ArrayBufferView);
  const boundedF32Data = f32Data.length === size ? f32Data : f32Data.subarray(0, size);
  if (expectedType === 'float16') {
    return new ort.Tensor('float16', float32ToFloat16Bits(boundedF32Data), dims) as unknown as OrtTensorLike<Float32Array>;
  }
  return new ort.Tensor('float32', boundedF32Data, dims) as unknown as OrtTensorLike<Float32Array>;
}

export async function maybeCastWhisperFeatureTensor(
  featureTensor: OrtTensorLike<Float32Array>,
  encoderSession: OrtSessionLike,
  ort: OrtModuleLike,
): Promise<OrtTensorLike<Float32Array>> {
  const metadata = (encoderSession as unknown as {
    inputMetadata?: Array<{ name?: string; type?: string; shape?: number[] }>;
  }).inputMetadata;
  const featureMeta = metadata?.find((entry) => entry.name === 'input_features');
  const expectedType = featureMeta?.type;
  if (expectedType !== 'float16' && expectedType !== 'float32') {
    return featureTensor;
  }
  if (featureTensor.type === expectedType) {
    return featureTensor;
  }

  const rawData = isGpuBufferTensor(featureTensor) && featureTensor.getData
    ? await featureTensor.getData(true)
    : featureTensor.data;
  const dims = featureTensor.dims as number[];
  const size = dims.reduce((a, b) => a * b, 1);
  const f32Data = tensorDataAsFloat32(rawData as ArrayBufferView);
  const boundedF32Data = f32Data.length === size ? f32Data : f32Data.subarray(0, size);
  const recast = expectedType === 'float16'
    ? new ort.Tensor('float16', float32ToFloat16Bits(boundedF32Data), dims) as unknown as OrtTensorLike<Float32Array>
    : new ort.Tensor('float32', boundedF32Data, dims) as unknown as OrtTensorLike<Float32Array>;
  featureTensor.dispose?.();
  return recast;
}

export async function runWhisperEncoderOnFeatures(
  encoderSession: OrtSessionLike,
  featureTensor: OrtTensorLike<Float32Array>,
): Promise<Record<string, OrtTensorLike>> {
  try {
    return await encoderSession.run({ input_features: featureTensor });
  } finally {
    featureTensor.dispose?.();
  }
}
