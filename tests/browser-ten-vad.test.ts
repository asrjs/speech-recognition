import { describe, expect, it } from 'vitest';
import { TenVadAdapter } from '@asrjs/speech-recognition/browser';

class FakeWorker {
  onmessage = null;
  onerror = null;
  messages = [];

  constructor(mode = 'ok') {
    this.mode = mode;
  }

  postMessage(message) {
    this.messages.push(message);
    if (message.type === 'INIT') {
      if (this.mode === 'fail-init') {
        this.onmessage?.({ data: { type: 'ERROR', id: message.id, payload: 'init failed' } });
      } else {
        this.onmessage?.({
          data: { type: 'INIT', id: message.id, payload: { success: true, version: 'test' } },
        });
      }
      return;
    }

    if (message.type === 'PROCESS') {
      this.onmessage?.({
        data: {
          type: 'RESULT',
          payload: {
            probabilities: new Float32Array([0.8, 0.82, 0.84, 0.81]),
            flags: new Uint8Array([1, 1, 1, 1]),
            globalSampleOffset: message.payload.globalSampleOffset,
            hopCount: 4,
          },
        },
      });
      return;
    }

    this.onmessage?.({
      data: {
        type: message.type,
        id: message.id,
        payload: { success: true },
      },
    });
  }

  terminate() {}
}

describe('TenVadAdapter', () => {
  it('handles init, process, reset, and dispose with aligned frame offsets', async () => {
    const adapter = new TenVadAdapter(
      {
        hopSize: 256,
      },
      {
        workerFactory: () => new FakeWorker(),
      },
    );

    await adapter.init();
    const seen = [];
    const unsubscribe = adapter.subscribe((event) => seen.push(event));

    const processed = adapter.process(new Float32Array(512), 1024);
    expect(processed).toBe(true);
    expect(seen.at(-1)?.payload.globalSampleOffset).toBe(1024);

    expect(adapter.findFirstSpeechFrame(1024, 2048)).toBe(1024);
    expect(adapter.hasRecentSpeech(2048, 128, 16000)).toBe(true);
    expect(adapter.hasRecentSilence(2048, 128, 16000)).toBe(false);

    await adapter.reset();
    await adapter.dispose();
    unsubscribe();
  });

  it('degrades gracefully when init fails', async () => {
    const adapter = new TenVadAdapter({}, { workerFactory: () => new FakeWorker('fail-init') });
    await expect(adapter.init()).rejects.toThrow('init failed');
    expect(adapter.getStatus().state).toBe('degraded');
  });
});
