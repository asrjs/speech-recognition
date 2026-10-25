import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_NEMO_TDT_CLASSIFICATION,
  parseNemoTdtConfig,
} from '../src/models/nemo-tdt/config.js';
import { OrtNemoTdtExecutor } from '../src/models/nemo-tdt/executor.js';
import * as ort from '../src/models/nemo-tdt/ort.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';
import * as huggingface from '../src/runtime/huggingface.js';
import type { AssetProvider, ResolvedAssetHandle } from '../src/types/index.js';

afterEach(() => vi.restoreAllMocks());

function createResolvedHandle(filename: string): ResolvedAssetHandle {
  return {
    request: { id: filename, filename },
    async *openStream() {
      yield new Uint8Array();
    },
    async readBytes() {
      return new Uint8Array();
    },
    async readText() {
      return '';
    },
    async readJson<T>() {
      return {} as T;
    },
    async getLocator(target) {
      return target === 'url' ? `blob:test/${filename}` : null;
    },
    dispose() {},
  };
}

function createExecutor(
  encoderQuant: 'fp16' | undefined,
  timeline: string[],
  createSession: typeof ort.createOrtSession,
  decoderQuant?: 'fp16' | 'fp32' | 'int8',
) {
  const requests: string[] = [];
  const assetProvider: AssetProvider = {
    canResolve: () => true,
    async resolve(request) {
      const filename = request.filename ?? '';
      requests.push(filename);
      timeline.push(`asset:${filename}`);
      return createResolvedHandle(filename);
    },
  };
  vi.spyOn(huggingface, 'fetchModelFiles').mockResolvedValue([
    'encoder-model.fp16.onnx',
    'encoder-model.onnx',
    'decoder_joint-model.onnx',
    'decoder_joint-model.int8.onnx',
    'vocab.txt',
  ]);
  vi.spyOn(ort, 'initOrt').mockResolvedValue({} as ort.OrtModuleLike);
  vi.spyOn(ort, 'createOrtSession').mockImplementation(createSession);
  vi.spyOn(ParakeetTokenizer, 'fromUrl').mockResolvedValue({} as ParakeetTokenizer);

  const progress: Array<{
    phase: string;
    file?: string;
    backendId?: string;
    warning?: { code: string; message: string; recoverable?: boolean };
  }> = [];
  const executor = new OrtNemoTdtExecutor(
    'parakeet-tdt-0.6b-v2',
    DEFAULT_NEMO_TDT_CLASSIFICATION,
    parseNemoTdtConfig('parakeet-tdt-0.6b-v2'),
    'wasm',
    {
      source: {
        kind: 'huggingface',
        repoId: 'ysdede/parakeet-tdt-0.6b-v2-onnx',
        encoderBackend: 'webgpu',
        decoderBackend: 'wasm',
        preprocessorBackend: 'js',
        ...(encoderQuant ? { encoderQuant } : {}),
        ...(decoderQuant ? { decoderQuant } : {}),
      },
    },
    {
      assetProvider,
      runtimeHooks: {
        onProgress(event) {
          progress.push(event);
          if (event.phase === 'model-load:fallback') timeline.push(`fallback:${event.file}`);
        },
      },
    },
  );
  return { executor, progress, requests, timeline };
}

describe('NeMo TDT encoder fallback progress', () => {
  it('reports the failed default and next precision before requesting fallback weights', async () => {
    const timeline: string[] = [];
    const encoderSession = { release() {} };
    const createSession = vi.fn(async (_ort, url) => {
      timeline.push(`session:${url}`);
      if (url.includes('encoder-model.fp16.onnx')) throw new Error('WebGPU rejected FP16');
      return encoderSession;
    });
    const { executor, progress, requests } = createExecutor(undefined, timeline, createSession);

    await executor.ready();

    const fallback = progress.find((event) => event.phase === 'model-load:fallback');
    expect(fallback).toMatchObject({
      backendId: 'webgpu',
      file: 'encoder-model.onnx',
      warning: {
        code: 'nemo-tdt.encoder-fp16-fallback-fp32',
        recoverable: true,
      },
    });
    expect(fallback?.warning?.message).toContain('WebGPU rejected FP16');
    expect(timeline.indexOf('fallback:encoder-model.onnx')).toBeLessThan(
      timeline.indexOf('asset:encoder-model.onnx'),
    );
    expect(requests).toContain('encoder-model.onnx');

    await executor.dispose();
  });

  it('keeps an explicit FP16 request strict and does not start fallback', async () => {
    const timeline: string[] = [];
    const createSession = vi.fn(async () => {
      throw new Error('WebGPU rejected explicit FP16');
    });
    const { executor, progress, requests } = createExecutor('fp16', timeline, createSession);

    await expect(executor.ready()).rejects.toThrow('WebGPU rejected explicit FP16');

    expect(createSession).toHaveBeenCalledOnce();
    expect(progress).toEqual([]);
    expect(requests).not.toContain('encoder-model.onnx');

    await executor.dispose();
  });

  it('reports decoder external-data fallback before requesting INT8 weights', async () => {
    const timeline: string[] = [];
    const session = { release() {} };
    const createSession = vi.fn(async (_ort, url) => {
      timeline.push(`session:${url}`);
      if (url.includes('decoder_joint-model.onnx')) {
        throw new Error('Failed to load external data file');
      }
      return session;
    });
    const { executor, progress, requests } = createExecutor(
      undefined,
      timeline,
      createSession,
      'fp32',
    );

    await executor.ready();

    const fallback = progress.find(
      (event) =>
        event.phase === 'model-load:fallback' &&
        event.warning?.code === 'nemo-tdt.decoder-fp32-fallback',
    );
    expect(fallback).toMatchObject({
      backendId: 'wasm',
      file: 'decoder_joint-model.int8.onnx',
      warning: { recoverable: true },
    });
    expect(fallback?.warning?.message).toContain('Failed to load external data file');
    expect(timeline.indexOf('fallback:decoder_joint-model.int8.onnx')).toBeLessThan(
      timeline.indexOf('asset:decoder_joint-model.int8.onnx'),
    );
    expect(requests).toContain('decoder_joint-model.int8.onnx');

    await executor.dispose();
  });
});
