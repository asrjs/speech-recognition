import path from 'node:path';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { afterEach, describe, expect, it, vi } from 'vitest';

const ortMock = vi.hoisted(() => ({ create: vi.fn() }));
vi.mock('onnxruntime-web', () => {
  const runtime = {
    env: { wasm: {} },
    Tensor: class RuntimeTensor {},
    InferenceSession: { create: ortMock.create },
  };
  return { ...runtime, default: runtime };
});

import { DEFAULT_NEMO_AED_CLASSIFICATION, parseNemoAedConfig } from '../src/models/nemo-aed/config.js';
import { OrtNemoAedExecutor } from '../src/models/nemo-aed/executor.js';
import type { NemoAedArtifactSource } from '../src/models/nemo-aed/types.js';

const tempDirs: string[] = [];

async function createFixture() {
  const directory = await mkdtemp(path.join(tmpdir(), 'asrjs-nemo-aed-external-data-'));
  tempDirs.push(directory);
  const files = {
    encoder: path.join(directory, 'encoder-model.onnx'),
    encoderData: path.join(directory, 'encoder-model.onnx.data'),
    decoder: path.join(directory, 'decoder-model.onnx'),
    decoderData: path.join(directory, 'decoder-model.onnx.data'),
    tokenizer: path.join(directory, 'tokenizer.json'),
  };
  await Promise.all([
    writeFile(files.encoder, new Uint8Array([1])),
    writeFile(files.encoderData, new Uint8Array([11, 12, 13, 14])),
    writeFile(files.decoder, new Uint8Array([2])),
    writeFile(files.decoderData, new Uint8Array([21, 22, 23])),
    writeFile(files.tokenizer, JSON.stringify({
      kind: 'canary-aggregate-tokenizer', version: 1, prompt_format: 'canary2', vocab_size: 1,
      langs: [], language_codes: [], bos_id: 0, eos_id: 0, pad_id: 0,
      special_tokens: {}, subtokenizers: { all: { offset: 0, size: 1, pieces: ['x'] } },
    })),
  ]);
  return files;
}

function createExecutor(
  files: Awaited<ReturnType<typeof createFixture>>,
  withExternalData: boolean | { readonly encoderDataUrl: string; readonly decoderDataUrl: string } = true,
) {
  const source = {
    kind: 'direct',
    preprocessorBackend: 'js',
    encoderBackend: 'wasm',
    decoderBackend: 'wasm',
    artifacts: {
      encoderUrl: pathToFileURL(files.encoder).href,
      decoderUrl: pathToFileURL(files.decoder).href,
      tokenizerUrl: pathToFileURL(files.tokenizer).href,
      ...(withExternalData ? {
        encoderDataUrl: typeof withExternalData === 'object'
          ? withExternalData.encoderDataUrl
          : pathToFileURL(files.encoderData).href,
        decoderDataUrl: typeof withExternalData === 'object'
          ? withExternalData.decoderDataUrl
          : pathToFileURL(files.decoderData).href,
      } : {}),
      encoderFilename: 'encoder-model.onnx',
      decoderFilename: 'decoder-model.onnx',
    },
  } satisfies NemoAedArtifactSource;
  return new OrtNemoAedExecutor(
    'nvidia/canary-1b-v2', DEFAULT_NEMO_AED_CLASSIFICATION,
    parseNemoAedConfig('nvidia/canary-1b-v2'), 'wasm', { source },
  );
}

afterEach(async () => {
  vi.clearAllMocks();
  await Promise.all(tempDirs.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe('NeMo AED ONNX external data', () => {
  it('mounts encoder and decoder sidecar bytes under each graph-relative filename', async () => {
    const files = await createFixture();
    ortMock.create.mockImplementation(async () => ({ run: async () => ({}), release: vi.fn() }));
    const executor = createExecutor(files);
    try {
      await executor.ready();
      const calls = ortMock.create.mock.calls as unknown as Array<[string, Record<string, unknown>]>;
      expect(calls).toHaveLength(2);
      const encoderMounts = calls[0]![1].externalData as Array<{ path: string; data: Uint8Array }>;
      const decoderMounts = calls[1]![1].externalData as Array<{ path: string; data: Uint8Array }>;
      expect(encoderMounts).toEqual([{ path: 'encoder-model.onnx.data', data: await readFile(files.encoderData) }]);
      expect(decoderMounts).toEqual([{ path: 'decoder-model.onnx.data', data: await readFile(files.decoderData) }]);
    } finally {
      await executor.dispose();
    }
  });

  it('leaves inline graph sessions unchanged and releases the encoder if decoder creation fails', async () => {
    const inlineFiles = await createFixture();
    ortMock.create.mockImplementation(async () => ({ run: async () => ({}), release: vi.fn() }));
    const inlineExecutor = createExecutor(inlineFiles, false);
    try {
      await inlineExecutor.ready();
      const inlineCalls = ortMock.create.mock.calls as unknown as Array<[string, Record<string, unknown>]>;
      expect(inlineCalls).toHaveLength(2);
      expect(inlineCalls.every(([, options]) => !('externalData' in options))).toBe(true);
    } finally {
      await inlineExecutor.dispose();
    }

    vi.clearAllMocks();
    const failingFiles = await createFixture();
    const released: string[] = [];
    ortMock.create
      .mockImplementationOnce(async () => ({ run: async () => ({}), release: () => released.push('encoder') }))
      .mockRejectedValueOnce(new Error('decoder session failed'));
    const failingExecutor = createExecutor(failingFiles);
    await expect(failingExecutor.ready()).rejects.toThrow('decoder session failed');
    expect(released).toEqual(['encoder']);
    await failingExecutor.dispose();
  });

  it('keeps an actual tiny Blob URL intact in Node ORT mounts', async () => {
    const files = await createFixture();
    const sidecarUrl = URL.createObjectURL(new Blob([new Uint8Array([41, 42, 43])]));
    ortMock.create.mockImplementation(async () => ({ run: async () => ({}), release: vi.fn() }));
    const executor = createExecutor(files, { encoderDataUrl: sidecarUrl, decoderDataUrl: sidecarUrl });
    try {
      await executor.ready();
      const calls = ortMock.create.mock.calls as unknown as Array<[string, Record<string, unknown>]>;
      expect(calls[0]![1].externalData).toEqual([{ data: sidecarUrl, path: 'encoder-model.onnx.data' }]);
      expect(calls[1]![1].externalData).toEqual([{ data: sidecarUrl, path: 'decoder-model.onnx.data' }]);
    } finally {
      await executor.dispose();
      URL.revokeObjectURL(sidecarUrl);
    }
  });
});
