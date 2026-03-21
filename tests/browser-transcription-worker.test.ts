import { describe, expect, it } from 'vitest';
import { createBrowserTranscriptionWorkerClient } from '@asrjs/speech-recognition/browser';

class FakeWorker {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  public readonly messages: any[] = [];
  public terminated = false;

  postMessage(message: any): void {
    this.messages.push(message);
    queueMicrotask(() => {
      if (message.type === 'LOAD_BUILT_IN_MODEL') {
        this.onmessage?.({
          data: {
            id: message.id,
            type: 'SUCCESS',
            payload: {
              source: 'built-in',
              modelId: message.payload.modelId,
            },
            meta: {
              state: 'ready',
              error: null,
              model: {
                source: 'built-in',
                modelId: message.payload.modelId,
              },
            },
          },
        } as MessageEvent);
        return;
      }

      if (message.type === 'TRANSCRIBE_MONO_PCM') {
        this.onmessage?.({
          data: {
            id: message.id,
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
          id: message.id,
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

describe('BrowserTranscriptionWorkerClient', () => {
  it('loads a model and transcribes PCM through the worker transport', async () => {
    const worker = new FakeWorker();
    const client = createBrowserTranscriptionWorkerClient({
      workerFactory: () => worker as never,
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
});
