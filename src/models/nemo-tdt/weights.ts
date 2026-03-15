import type { NemoTdtQuantization } from './types.js';

export interface NemoTdtWeightSetup {
  readonly backend: 'webgpu' | 'wasm';
  readonly encoderDefault: NemoTdtQuantization;
  readonly decoderDefault: NemoTdtQuantization;
  readonly encoderFallback: NemoTdtQuantization;
  readonly encoderPreferred: readonly NemoTdtQuantization[];
  readonly decoderPreferred: readonly NemoTdtQuantization[];
}

export function normalizeNemoTdtWeightBackend(backendId: string): 'webgpu' | 'wasm' {
  return backendId.startsWith('webgpu') ? 'webgpu' : 'wasm';
}

export function getDefaultNemoTdtWeightSetup(backendId: string): NemoTdtWeightSetup {
  const backend = normalizeNemoTdtWeightBackend(backendId);
  if (backend === 'webgpu') {
    return {
      backend,
      encoderDefault: 'fp16',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['fp16', 'fp32', 'int8'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    };
  }

  return {
    backend,
    encoderDefault: 'int8',
    decoderDefault: 'int8',
    encoderFallback: 'fp32',
    encoderPreferred: ['int8', 'fp32', 'fp16'],
    decoderPreferred: ['int8', 'fp32', 'fp16'],
  };
}

export function pickPreferredNemoTdtQuantization(
  available: readonly NemoTdtQuantization[],
  backendId: string,
  role: 'encoder' | 'decoder',
): NemoTdtQuantization {
  const setup = getDefaultNemoTdtWeightSetup(backendId);
  const preference = role === 'encoder' ? setup.encoderPreferred : setup.decoderPreferred;
  const fallback = role === 'encoder' ? setup.encoderDefault : setup.decoderDefault;

  return (
    preference.find((quantization) => available.includes(quantization)) ?? available[0] ?? fallback
  );
}
