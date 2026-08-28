import { describe, expect, it, vi } from 'vitest';
import { createBrowserTranscriptionWorkerClient } from '@asrjs/speech-recognition/browser';

class FakeWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  public readonly messages: any[] = [];
  public terminated = false;
  public holdTranscribe = false;

  postMessage(message: any, transfer: Transferable[] = []): void {
    const request = structuredClone(message, { transfer });
    this.messages.push(request);

    if (request.type === 'TRANSCRIBE_MONO_PCM' && this.holdTranscribe) {
      return;
    }

    queueMicrotask(() => {
      if (request.type === 'LOAD_BUILT_IN_MODEL') {
        this.onmessage?.({
          data: {
            id: request.id,
            type: 'SUCCESS',
            payload: {
              source: 'built-in',
              modelId: request.payload.modelId,
            },
            meta: {
              state: 'ready',
              error: null,
              model: {
                source: 'built-in',
                modelId: request.payload.modelId,
              },
            },
          },
        } as MessageEvent);
        return;
      }

      if (request.type === 'TRANSCRIBE_MONO_PCM') {
        this.onmessage?.({
          data: {
            id: request.id,
            type: 'SUCCESS',
            payload: {
              text: 'ok',
            },
            meta: {
              state: 'ready',
              error: null,
              model: {
                source: 'built-in',
                modelId: 'parakeet',
              },
            },
          },
        } as MessageEvent);
        return;
      }

      this.onmessage?.({
        data: {
          id: request.id,
          type: 'SUCCESS',
          payload: null,
          meta: {
            state: 'idle',
            error: null,
            model: null,
          },
        },
      } as MessageEvent);
    });
  }

  terminate(): void {
    this.terminated = true;
  }
}

class ThrowingWorker extends FakeWorker {
  override postMessage(message: any, transfer: Transferable[] = []): void {
    if (message.type === 'LOAD_BUILT_IN_MODEL') {
      throw new Error('sync postMessage failure');
    }
    super.postMessage(message, transfer);
  }
}

class ErroringWorker extends FakeWorker {
  override postMessage(message: any, transfer: Transferable[] = []): void {
    if (message.type === 'LOAD_BUILT_IN_MODEL') {
      queueMicrotask(() => {
        this.onerror?.({
          message: 'worker boot failed',
        } as ErrorEvent);
      });
      return;
    }
    super.postMessage(message, transfer);
  }
}

describe('BrowserTranscriptionWorkerClient', () => {
  it('loads a model and transcribes PCM through the worker transport', async () => {
    const worker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker,
    });

    await client.loadBuiltInModel({
      modelId: 'parakeet',
      backend: 'webgpu',
    });
    const result = await client.transcribeMonoPcm(new Float32Array([0, 0.1]), 16000);

    expect(result).toMatchObject({ text: 'ok' });
    expect(worker.messages.map((message) => message.type)).toEqual([
      'LOAD_BUILT_IN_MODEL',
      'TRANSCRIBE_MONO_PCM',
    ]);
    expect(client.getStatus()).toMatchObject({
      state: 'ready',
      model: { modelId: 'parakeet' },
    });

    await client.dispose();
    expect(worker.terminated).toBe(true);
  });

  it('does not detach the caller PCM buffer during worker transcription', async () => {
    const worker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker,
    });

    await client.loadBuiltInModel({
      modelId: 'parakeet',
      backend: 'webgpu',
    });

    const pcm = new Float32Array([0.25, -0.5, 0.125]);
    await client.transcribeMonoPcm(pcm, 16000);

    expect(pcm.buffer.byteLength).toBe(12);
    expect(Array.from(pcm)).toEqual([0.25, -0.5, 0.125]);

    await client.dispose();
  });

  it('recreates the worker after an asynchronous worker error', async () => {
    const workers = [new ErroringWorker(), new FakeWorker()];
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => workers.shift() ?? new FakeWorker(),
    });

    await expect(
      client.loadBuiltInModel({
        modelId: 'parakeet',
        backend: 'webgpu',
      }),
    ).rejects.toThrow('worker boot failed');

    await expect(
      client.loadBuiltInModel({
        modelId: 'parakeet',
        backend: 'webgpu',
      }),
    ).resolves.toMatchObject({
      modelId: 'parakeet',
      source: 'built-in',
    });
  });

  it('removes pending bookkeeping when postMessage throws synchronously', async () => {
    const failingWorker = new ThrowingWorker();
    const fallbackWorker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: (() => {
        let callCount = 0;
        return () => {
          callCount += 1;
          return callCount === 1 ? failingWorker : fallbackWorker;
        };
      })(),
    });

    await expect(
      client.loadBuiltInModel({
        modelId: 'parakeet',
        backend: 'webgpu',
      }),
    ).rejects.toThrow('sync postMessage failure');

    await expect(
      client.loadBuiltInModel({
        modelId: 'parakeet',
        backend: 'webgpu',
      }),
    ).resolves.toMatchObject({
      modelId: 'parakeet',
      source: 'built-in',
    });
  });

  it('does not create a worker when load is already aborted', async () => {
    let created = 0;
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => {
        created += 1;
        return new FakeWorker();
      },
    });

    await expect(
      client.loadBuiltInModel({
        modelId: 'parakeet',
        backend: 'webgpu',
        signal: { aborted: true },
      }),
    ).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(created).toBe(0);
    expect(client.getStatus().state).toBe('idle');
  });

  it('terminates the worker when load is aborted before LOAD returns', async () => {
    const worker = new FakeWorker();
    worker.postMessage = (message) => {
      worker.messages.push(message);
    };
    const controller = new AbortController();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker,
    });

    const loading = client.loadBuiltInModel({
      modelId: 'parakeet',
      backend: 'webgpu',
      signal: controller.signal,
    });
    await Promise.resolve();
    controller.abort();

    await expect(loading).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(worker.terminated).toBe(true);
    expect(client.getStatus().state).toBe('idle');
  });

  it('does not transcribe when the signal is already aborted', async () => {
    const worker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker,
    });

    await client.loadBuiltInModel({
      modelId: 'parakeet',
      backend: 'webgpu',
    });
    worker.messages.length = 0;

    await expect(
      client.transcribeMonoPcm(new Float32Array([0, 0.1]), 16000, { signal: { aborted: true } }),
    ).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(worker.messages).toEqual([]);
    expect(worker.terminated).toBe(false);

    await client.dispose();
  });

  it('cancels transcribe cooperatively and reuses the loaded worker', async () => {
    const worker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker,
    });

    await client.loadBuiltInModel({
      modelId: 'parakeet',
      backend: 'webgpu',
    });

    worker.messages.length = 0;
    worker.holdTranscribe = true;
    const controller = new AbortController();
    const transcribing = client.transcribeMonoPcm(new Float32Array([0, 0.1]), 16000, {
      signal: controller.signal,
    });
    await Promise.resolve();
    controller.abort();

    await expect(transcribing).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(worker.messages.map((message) => message.type)).toEqual([
      'TRANSCRIBE_MONO_PCM',
      'CANCEL_TRANSCRIBE',
    ]);
    expect(worker.messages[1]?.payload).toEqual({ requestId: worker.messages[0]?.id });
    expect(worker.terminated).toBe(false);
    expect(client.getStatus()).toMatchObject({
      state: 'ready',
      model: { modelId: 'parakeet' },
    });

    worker.holdTranscribe = false;
    await expect(
      client.transcribeMonoPcm(new Float32Array([0, 0.1]), 16000),
    ).resolves.toMatchObject({ text: 'ok' });
    expect(worker.terminated).toBe(false);

    await client.dispose();
  });

  it('observes asynchronous changes on an abort-like signal', async () => {
    const worker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker,
    });

    await client.loadBuiltInModel({
      modelId: 'parakeet',
      backend: 'webgpu',
    });

    worker.messages.length = 0;
    worker.holdTranscribe = true;
    const signal = { aborted: false };
    const transcribing = client.transcribeMonoPcm(new Float32Array([0, 0.1]), 16000, {
      signal,
    });
    const transcribingSettled = transcribing.then(
      (value) => ({ status: 'fulfilled' as const, value }),
      (reason: unknown) => ({ status: 'rejected' as const, reason }),
    );

    try {
      await vi.waitFor(() => expect(worker.messages[0]?.type).toBe('TRANSCRIBE_MONO_PCM'));
      signal.aborted = true;

      await vi.waitFor(
        () =>
          expect(worker.messages.some((message) => message.type === 'CANCEL_TRANSCRIBE')).toBe(
            true,
          ),
        { timeout: 250 },
      );
      await expect(transcribingSettled).resolves.toMatchObject({
        status: 'rejected',
        reason: {
          name: 'AssetLoadAbortedError',
          code: 'asset-load-aborted',
        },
      });
      expect(worker.terminated).toBe(false);

      worker.holdTranscribe = false;
      await expect(
        client.transcribeMonoPcm(new Float32Array([0, 0.1]), 16000),
      ).resolves.toMatchObject({ text: 'ok' });
    } finally {
      worker.holdTranscribe = false;
      await client.dispose();
      await transcribingSettled;
    }
  });
});
