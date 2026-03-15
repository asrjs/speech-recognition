import { afterEach, describe, expect, it, vi } from 'vitest';
import { getModelText } from '../src/runtime/huggingface.js';

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
});
