import { afterEach, describe, expect, it, vi } from 'vitest';

import { AssetLoadAbortedError } from '../src/io/abort.js';
import { MedAsrTextTokenizer } from '../src/models/lasr-ctc/tokenizer.js';
import { ParakeetTokenizer } from '../src/models/nemo-tdt/tokenizer.js';
import { WhisperTokenizer, fetchText } from '../src/models/whisper-seq2seq/tokenizer.js';

const originalFetch = globalThis.fetch;
const originalCreateObjectURL = URL.createObjectURL;

afterEach(() => {
  vi.restoreAllMocks();
  globalThis.fetch = originalFetch;
  URL.createObjectURL = originalCreateObjectURL;
});

function abortingFetch(): typeof fetch {
  return vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.signal?.aborted) {
      const error = new Error('Aborted');
      error.name = 'AbortError';
      throw error;
    }
    const controller = new AbortController();
    init?.signal?.addEventListener('abort', () => controller.abort());
    await new Promise<void>((_resolve, reject) => {
      const fail = () => {
        const error = new Error('Aborted');
        error.name = 'AbortError';
        reject(error);
      };
      if (init?.signal?.aborted) {
        fail();
        return;
      }
      init?.signal?.addEventListener('abort', fail);
    });
    return new Response('unused', { status: 200 });
  }) as typeof fetch;
}

describe('tokenizer fromUrl abort', () => {
  it('stops MedAsr vocabulary fetch and does not create object URLs', async () => {
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockImplementation(() => 'blob:unused');
    const controller = new AbortController();
    globalThis.fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      queueMicrotask(() => controller.abort());
      if (init?.signal) {
        await new Promise<void>((_resolve, reject) => {
          const fail = () => {
            const error = new Error('Aborted');
            error.name = 'AbortError';
            reject(error);
          };
          if (init.signal.aborted) {
            fail();
            return;
          }
          init.signal.addEventListener('abort', fail);
        });
      }
      return new Response('<blk> 0\nhello 1\n', { status: 200 });
    }) as typeof fetch;

    await expect(
      MedAsrTextTokenizer.fromUrl('https://example.com/vocab.txt', controller.signal),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it('stops Parakeet vocabulary fetch when aborted after the body starts', async () => {
    const controller = new AbortController();
    globalThis.fetch = abortingFetch();
    queueMicrotask(() => controller.abort());

    await expect(
      ParakeetTokenizer.fromUrl('https://example.com/tokenizer.txt', {
        signal: controller.signal,
      }),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
  });

  it('does not swallow abort during Whisper optional config fetchText', async () => {
    const controller = new AbortController();
    globalThis.fetch = abortingFetch();
    queueMicrotask(() => controller.abort());

    await expect(
      fetchText('https://example.com/generation_config.json', controller.signal),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
    await expect(
      WhisperTokenizer.fromUrl('https://example.com/tokenizer.json', controller.signal),
    ).rejects.toBeInstanceOf(AssetLoadAbortedError);
  });
});
