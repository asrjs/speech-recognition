import { describe, expect, it } from 'vitest';

import {
  getDefaultNemoRnntWeightSetup,
  pickPreferredNemoRnntQuantization,
} from '@asrjs/speech-recognition/models/nemo-rnnt';
import { resolveNemoRnntArtifacts } from '../src/models/nemo-rnnt/ort.js';

describe('nemo-rnnt weight defaults', () => {
  it('prefers fp16 encoder weights with fp32 fallback on webgpu', () => {
    expect(getDefaultNemoRnntWeightSetup('webgpu')).toEqual({
      backend: 'webgpu',
      encoderDefault: 'fp16',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['fp16', 'fp32', 'int8'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    });
  });

  it('prefers int8 encoder weights on wasm', () => {
    expect(getDefaultNemoRnntWeightSetup('wasm')).toEqual({
      backend: 'wasm',
      encoderDefault: 'int8',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['int8', 'fp32', 'fp16'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    });
  });

  it('picks preferred encoder and decoder quantization from available artifacts', () => {
    expect(pickPreferredNemoRnntQuantization(['fp16', 'fp32'], 'webgpu', 'encoder')).toBe('fp16');
    expect(pickPreferredNemoRnntQuantization(['fp32', 'int8'], 'wasm', 'encoder')).toBe('int8');
    expect(pickPreferredNemoRnntQuantization(['fp16', 'fp32', 'int8'], 'webgpu', 'decoder')).toBe(
      'int8',
    );
  });

  it('uses the optimized defaults when resolving huggingface artifacts', () => {
    const webgpuResolved = resolveNemoRnntArtifacts(
      {
        kind: 'huggingface',
        repoId: 'ysdede/parakeet-realtime-eou-120m-v1-onnx',
        preprocessorName: 'nemo128',
      },
      'webgpu',
    );
    const wasmResolved = resolveNemoRnntArtifacts(
      {
        kind: 'huggingface',
        repoId: 'ysdede/parakeet-realtime-eou-120m-v1-onnx',
        preprocessorName: 'nemo128',
      },
      'wasm',
    );

    expect(webgpuResolved.artifacts.encoderFilename).toBe('encoder-model.fp16.onnx');
    expect(webgpuResolved.artifacts.decoderFilename).toBe('decoder_joint-model.int8.onnx');
    expect(wasmResolved.artifacts.encoderFilename).toBe('encoder-model.int8.onnx');
    expect(wasmResolved.artifacts.decoderFilename).toBe('decoder_joint-model.int8.onnx');
  });
});
