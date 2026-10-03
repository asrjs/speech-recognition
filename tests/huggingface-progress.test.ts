import { afterEach, describe, expect, it, vi } from 'vitest';
import { getModelFile, getModelText } from '../src/runtime/huggingface.js';
import * as io from '../src/io/index.js';

function createChunkedResponse(totalBytes: number, chunkBytes: number): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      let emitted = 0;
      while (emitted < totalBytes) {
        const nextSize = Math.min(chunkBytes, totalBytes - emitted);
        controller.enqueue(new Uint8Array(nextSize).fill(97));
        emitted += nextSize;
      }
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'content-length': String(totalBytes),
      'content-type': 'text/plain',
    },
  });
}

describe('Hugging Face progress reporting', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('isolates built-in pinned reads from legacy fallback bytes without deleting them', async () => {
    const legacyKey = 'hf-test/model-pinned--vocab.txt';
    const entries = new Map([[legacyKey, { bytes: new TextEncoder().encode('wrong main bytes') }]]);
    const cache = new io.IndexedDbAssetCache();
    vi.spyOn(io, 'getDefaultIndexedDbAssetCache').mockReturnValue(cache);
    vi.spyOn(cache, 'get').mockImplementation(async (key) => entries.get(key) ?? null);
    vi.spyOn(cache, 'set').mockImplementation(async (key, value) => {
      entries.set(key, value);
    });
    const deleting = vi.spyOn(cache, 'delete');
    const fetching = vi.fn(async () => new Response('correct pinned bytes'));
    vi.stubGlobal('fetch', fetching);

    expect(await getModelText('test/model', 'vocab.txt', { revision: 'pinned' })).toBe(
      'correct pinned bytes',
    );
    expect(await getModelText('test/model', 'vocab.txt', { revision: 'pinned' })).toBe(
      'correct pinned bytes',
    );
    expect(fetching).toHaveBeenCalledTimes(1);
    expect(entries.get(legacyKey)?.bytes).toEqual(new TextEncoder().encode('wrong main bytes'));
    expect(deleting).not.toHaveBeenCalled();
  });

  it.each([undefined, 'main'])('preserves existing main cache entries (%s)', async (revision) => {
    const cache = new io.IndexedDbAssetCache();
    vi.spyOn(io, 'getDefaultIndexedDbAssetCache').mockReturnValue(cache);
    const reading = vi
      .spyOn(cache, 'get')
      .mockResolvedValue({ bytes: new TextEncoder().encode('cached main') });
    const fetching = vi.fn();
    vi.stubGlobal('fetch', fetching);
    expect(await getModelText('test/model', 'vocab.txt', { revision })).toBe('cached main');
    expect(reading).toHaveBeenCalledWith('hf-test/model-main--vocab.txt');
    expect(fetching).not.toHaveBeenCalled();
  });

  it('coalesces chunk-level downloads into integer percent milestones', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => createChunkedResponse(1000, 1)),
    );

    const events: Array<{
      readonly loaded: number;
      readonly total: number;
      readonly file: string;
      readonly percent?: number;
      readonly loadedMiB: number;
      readonly totalMiB?: number;
      readonly isComplete?: boolean;
      readonly source?: 'cache' | 'network';
    }> = [];

    const text = await getModelText('ysdede/example', 'vocab.txt', {
      progress: (event) => events.push(event),
    });

    expect(text.length).toBe(1000);
    expect(events.length).toBeLessThanOrEqual(101);
    expect(events[0]?.file).toBe('vocab.txt');
    expect(events[0]?.percent).toBe(0);
    expect(events.at(-1)).toMatchObject({
      loaded: 1000,
      total: 1000,
      file: 'vocab.txt',
      percent: 100,
      isComplete: true,
      source: 'network',
    });
    expect(
      events.every(
        (event, index) => index === 0 || (event.percent ?? 0) > (events[index - 1]?.percent ?? -1),
      ),
    ).toBe(true);
    expect(events.every((event) => event.loadedMiB >= 0)).toBe(true);
    expect(events.every((event) => event.totalMiB !== undefined)).toBe(true);
  });

  it('still emits a final completion event when total size is unknown', async () => {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(512 * 1024));
        controller.enqueue(new Uint8Array(600 * 1024));
        controller.close();
      },
    });

    vi.stubGlobal(
      'fetch',
      vi.fn(
        async () =>
          new Response(stream, {
            status: 200,
            headers: { 'content-type': 'application/octet-stream' },
          }),
      ),
    );

    const events: Array<{
      readonly loaded: number;
      readonly isComplete?: boolean;
      readonly percent?: number;
    }> = [];

    await getModelText('ysdede/example', 'tokenizer.txt', {
      progress: (event) => events.push(event),
    });

    expect(events.length).toBeGreaterThanOrEqual(2);
    expect(events.at(-1)).toMatchObject({
      loaded: 1138688,
      percent: 100,
      isComplete: true,
    });
  });

  it('can transfer blob locator ownership for deterministic disposal', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: { 'content-length': '3', 'content-type': 'application/octet-stream' },
      })),
    );
    const handles: Array<{ dispose(): Promise<void> | void }> = [];

    const locator = await getModelFile('ysdede/example', 'encoder.onnx', {
      preferBlobUrl: true,
      onResolvedHandle: (handle) => handles.push(handle),
    });

    expect(locator).toMatch(/^blob:/);
    expect(handles).toHaveLength(1);
    await handles[0]!.dispose();
  });

  it('refuses preferBlobUrl without onResolvedHandle so blob URLs cannot leak', async () => {
    const fetchSpy = vi.fn();
    vi.stubGlobal('fetch', fetchSpy);
    const createObjectURL = vi.spyOn(URL, 'createObjectURL');

    await expect(
      getModelFile('ysdede/example', 'encoder.onnx', { preferBlobUrl: true }),
    ).rejects.toThrow(/requires onResolvedHandle so the blob URL can be revoked/);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(createObjectURL).not.toHaveBeenCalled();
  });

  it('disposes the asset handle when returning a remote HTTP locator', async () => {
    const createObjectURL = vi.spyOn(URL, 'createObjectURL');
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL');
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), {
        status: 200,
        headers: { 'content-length': '3', 'content-type': 'application/octet-stream' },
      })),
    );

    const locator = await getModelFile('ysdede/example', 'encoder.onnx');

    expect(locator).toMatch(/^https?:\/\//);
    expect(createObjectURL).not.toHaveBeenCalled();
    expect(revokeObjectURL).not.toHaveBeenCalled();
  });
});
