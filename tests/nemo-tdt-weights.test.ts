import { describe, expect, it } from 'vitest';

import {
  getDefaultNemoTdtWeightSetup,
  pickPreferredNemoTdtQuantization,
} from '@asrjs/speech-recognition/models/nemo-tdt';
import { resolveNemoTdtArtifacts } from '../src/models/nemo-tdt/ort.js';
import { buildSpeechModelLoadOptions } from '../src/runtime/catalog.js';
import type { NemoTdtArtifactSource } from '../src/models/nemo-tdt/types.js';

describe('nemo-tdt weight defaults', () => {
  it.each([
    ['webgpu-hybrid', {}, 'fp16', 'int8'],
    ['wasm', {}, 'int8', 'int8'],
    ['webgpu-hybrid', { encoderQuant: 'fp32', decoderQuant: 'fp16' }, 'fp32', 'fp16'],
    ['webgpu-hybrid', { encoderBackend: 'wasm' }, 'int8', 'int8'],
  ] as const)(
    'keeps catalog and executor precision decisions aligned for %s with %j',
    (backend, overrides, encoder, decoder) => {
      const request = buildSpeechModelLoadOptions({
        modelId: 'parakeet-tdt-0.6b-v2',
        backend,
        ...overrides,
      });
      const { source } = request.options as { source: NemoTdtArtifactSource };
      // A runtime primary of WASM must not erase an explicit WebGPU component split.
      const resolved = resolveNemoTdtArtifacts(source, 'wasm');
      expect(resolved.artifacts.encoderFilename).toBe(
        encoder === 'fp32' ? 'encoder-model.onnx' : `encoder-model.${encoder}.onnx`,
      );
      expect(resolved.artifacts.decoderFilename).toBe(`decoder_joint-model.${decoder}.onnx`);
    },
  );

  it('prefers fp16 encoder weights with fp32 fallback on webgpu', () => {
    expect(getDefaultNemoTdtWeightSetup('webgpu')).toEqual({
      backend: 'webgpu',
      encoderDefault: 'fp16',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['fp16', 'fp32', 'int8'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    });
  });

  it('prefers int8 encoder weights on wasm', () => {
    expect(getDefaultNemoTdtWeightSetup('wasm')).toEqual({
      backend: 'wasm',
      encoderDefault: 'int8',
      decoderDefault: 'int8',
      encoderFallback: 'fp32',
      encoderPreferred: ['int8', 'fp32', 'fp16'],
      decoderPreferred: ['int8', 'fp32', 'fp16'],
    });
  });

  it('picks preferred encoder and decoder quantization from available artifacts', () => {
    expect(pickPreferredNemoTdtQuantization(['fp16', 'fp32'], 'webgpu', 'encoder')).toBe('fp16');
    expect(pickPreferredNemoTdtQuantization(['fp32', 'int8'], 'wasm', 'encoder')).toBe('int8');
    expect(pickPreferredNemoTdtQuantization(['fp16', 'fp32', 'int8'], 'webgpu', 'decoder')).toBe(
      'int8',
    );
  });

  it('uses the optimized defaults when resolving huggingface artifacts', () => {
    const webgpuResolved = resolveNemoTdtArtifacts(
      {
        kind: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        preprocessorName: 'nemo128',
      },
      'webgpu',
    );
    const wasmResolved = resolveNemoTdtArtifacts(
      {
        kind: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v3-onnx',
        preprocessorName: 'nemo128',
      },
      'wasm',
    );

    expect(webgpuResolved.artifacts.encoderFilename).toBe('encoder-model.fp16.onnx');
    expect(webgpuResolved.artifacts.decoderFilename).toBe('decoder_joint-model.int8.onnx');
    expect(wasmResolved.artifacts.encoderFilename).toBe('encoder-model.int8.onnx');
    expect(wasmResolved.artifacts.decoderFilename).toBe('decoder_joint-model.int8.onnx');
  });

  it('preserves the opt-in decoder state output location in resolved artifacts', () => {
    const resolved = resolveNemoTdtArtifacts(
      {
        kind: 'direct',
        artifacts: {
          encoderUrl: 'https://example.test/encoder.onnx',
          decoderUrl: 'https://example.test/decoder.onnx',
          tokenizerUrl: 'https://example.test/vocab.txt',
        },
        decoderBackend: 'webgpu',
        decoderStateOutputLocation: 'gpu-buffer',
        webgpuOptions: { storageBufferCacheMode: 'simple' },
      },
      'webgpu',
    );

    expect(resolved.decoderStateOutputLocation).toBe('gpu-buffer');
    expect(resolved.webgpuOptions).toEqual({ storageBufferCacheMode: 'simple' });
  });
});
