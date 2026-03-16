import type { NemoAedQuantization } from './types.js';

export interface NemoAedWeightSetup {
  readonly backend: 'webgpu' | 'wasm';
  readonly encoderDefault: NemoAedQuantization;
  readonly decoderDefault: NemoAedQuantization;
  readonly encoderFallback: NemoAedQuantization;
  readonly decoderFallback: NemoAedQuantization;
  readonly encoderPreferred: readonly NemoAedQuantization[];
  readonly decoderPreferred: readonly NemoAedQuantization[];
}

export function normalizeNemoAedWeightBackend(backendId: string): 'webgpu' | 'wasm' {
  return backendId.startsWith('webgpu') ? 'webgpu' : 'wasm';
}

export function getDefaultNemoAedWeightSetup(backendId: string): NemoAedWeightSetup {
  const backend = normalizeNemoAedWeightBackend(backendId);
  if (backend === 'webgpu') {
    return {
      backend,
      encoderDefault: 'fp16',
      decoderDefault: 'fp16',
      encoderFallback: 'fp32',
      decoderFallback: 'fp32',
      encoderPreferred: ['fp16', 'fp32', 'int8'],
      decoderPreferred: ['fp16', 'fp32', 'int8'],
    };
  }

  return {
    backend,
    encoderDefault: 'fp32',
    decoderDefault: 'fp32',
    encoderFallback: 'fp32',
    decoderFallback: 'fp32',
    encoderPreferred: ['fp32', 'int8', 'fp16'],
    decoderPreferred: ['fp32', 'int8', 'fp16'],
  };
}
