import { describe, expect, it } from 'vitest';
import {
  FireRedVadAdapter,
  TenVadAdapter,
  resolveDefaultFireRedVadModelUrls,
  resolveDefaultTenVadAssetUrls,
  resolveFireRedVadModelUrls,
  resolveSupportedFireRedVadHopSize,
  resolveSupportedTenVadHopSize,
  resolveTenVadAssetUrls,
} from '@asrjs/speech-recognition/browser';
import { isTenVadWorkerRequest, isTenVadWorkerResponse } from '../src/runtime/ten-vad-protocol.js';
import type { TenVadRequestMessage } from '../src/runtime/ten-vad-protocol.js';

type FakeWorkerMode = 'ok' | 'fail-init' | 'hold-init';

class FakeWorker {
  onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  messages: TenVadRequestMessage[] = [];
  private readonly mode: FakeWorkerMode;

  constructor(mode: FakeWorkerMode = 'ok') {
    this.mode = mode;
  }

  postMessage(message: TenVadRequestMessage) {
    this.messages.push(message);
    if (message.type === 'INIT') {
      if (this.mode === 'fail-init') {
        this.onmessage?.({ data: { type: 'ERROR', id: message.id, payload: 'init failed' } });
      } else if (this.mode === 'hold-init') {
        return;
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

  emit(message: unknown): void {
    this.onmessage?.({ data: message } as MessageEvent<unknown>);
  }
}

describe('TEN-VAD worker protocol', () => {
  it('accepts the canonical control request and rejects malformed payloads', () => {
    const request = {
      type: 'INIT' as const,
      id: 1,
      payload: {
        hopSize: 256,
        threshold: 0.5,
        scriptUrl: 'https://example.com/ten_vad.js',
        wasmUrl: 'https://example.com/ten_vad.wasm',
        fallbackScriptUrl: null,
        fallbackWasmUrl: null,
      },
    };

    expect(isTenVadWorkerRequest(request)).toBe(true);
    expect(
      isTenVadWorkerRequest({
        ...request,
        payload: { ...request.payload, fallbackWasmUrl: 42 },
      }),
    ).toBe(false);
  });

  it('accepts complete result messages and rejects incomplete typed-array data', () => {
    const response = {
      type: 'RESULT' as const,
      payload: {
        probabilities: new Float32Array([0.8]),
        flags: new Uint8Array([1]),
        globalSampleOffset: 0,
        hopCount: 1,
      },
    };

    expect(isTenVadWorkerResponse(response)).toBe(true);
    expect(
      isTenVadWorkerResponse({
        ...response,
        payload: { ...response.payload, hopCount: 2 },
      }),
    ).toBe(false);
  });
});

describe('TenVadAdapter', () => {
  it('resolves default TEN-VAD asset URLs without adding a fallback loop', () => {
    const defaults = resolveDefaultTenVadAssetUrls();
    const resolved = resolveTenVadAssetUrls();

    expect(resolved.scriptUrl).toBe(defaults.scriptUrl);
    expect(resolved.wasmUrl).toBe(defaults.wasmUrl);
    expect(resolved.fallbackScriptUrl).toBeNull();
    expect(resolved.fallbackWasmUrl).toBeNull();
  });

  it('resolves TEN-VAD asset URLs from assetBaseUrl overrides', () => {
    const defaults = resolveDefaultTenVadAssetUrls();
    const resolved = resolveTenVadAssetUrls({
      assetBaseUrl: 'https://example.com/vendor/ten-vad/',
    });

    expect(resolved.scriptUrl).toBe('https://example.com/vendor/ten-vad/ten_vad.js');
    expect(resolved.wasmUrl).toBe('https://example.com/vendor/ten-vad/ten_vad.wasm');
    expect(resolved.fallbackScriptUrl).toBe(defaults.scriptUrl);
    expect(resolved.fallbackWasmUrl).toBe(defaults.wasmUrl);
  });

  it('normalizes preferred hop sizes to TEN-VAD supported hops', () => {
    expect(resolveSupportedTenVadHopSize(16000, 512)).toBe(256);
    expect(resolveSupportedTenVadHopSize(16000, 200)).toBe(160);
    expect(resolveSupportedTenVadHopSize(48000)).toBe(768);
  });

  it('handles init, process, reset, and dispose with aligned frame offsets', async () => {
    const adapter = new TenVadAdapter(
      {
        hopSize: 256,
        minSpeechDurationMs: 40,
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

  it('uses duration-derived smoothing defaults for safe speech acceptance', async () => {
    const adapter = new TenVadAdapter(
      {
        hopSize: 256,
      },
      {
        workerFactory: () => new FakeWorker(),
      },
    );

    await adapter.init();
    const processed = adapter.process(new Float32Array(512), 1024);
    expect(processed).toBe(true);

    expect(adapter.findFirstSpeechFrame(1024, 2048)).toBeNull();
    expect(adapter.hasRecentSpeech(2048, 128, 16000)).toBe(false);

    await adapter.dispose();
  });

  it('degrades gracefully when init fails', async () => {
    const adapter = new TenVadAdapter({}, { workerFactory: () => new FakeWorker('fail-init') });
    await expect(adapter.init()).rejects.toThrow('init failed');
    expect(adapter.getStatus().state).toBe('degraded');
  });

  it('ignores unknown or malformed worker responses without degrading', async () => {
    const worker = new FakeWorker();
    const adapter = new TenVadAdapter(
      { hopSize: 256, minSpeechDurationMs: 16 },
      { workerFactory: () => worker },
    );

    await adapter.init();
    worker.emit({ type: 'UNRECOGNIZED', id: 999, payload: null });
    worker.emit({ type: 'RESULT', payload: { hopCount: 1 } });

    expect(adapter.getStatus().state).toBe('ready');
    expect(adapter.getStatus().probability).toBe(0);
    expect(adapter.process(new Float32Array(256), 0)).toBe(true);
    expect(adapter.getStatus().probability).toBeCloseTo(0.81);

    await adapter.dispose();
  });

  it('rejects a pending init when disposed before the worker responds', async () => {
    const worker = new FakeWorker();
    worker.postMessage = (message) => {
      worker.messages.push(message);
    };
    const adapter = new TenVadAdapter({}, { workerFactory: () => worker });

    const initializing = adapter.init();
    await Promise.resolve();
    await adapter.dispose();

    await expect(initializing).rejects.toThrow('disposed');
    expect(worker.onmessage).toBeNull();
  });

  it('does not create a worker when init is already aborted', async () => {
    let created = 0;
    const adapter = new TenVadAdapter(
      {},
      {
        workerFactory: () => {
          created += 1;
          return new FakeWorker();
        },
      },
    );

    await expect(adapter.init({ aborted: true })).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(created).toBe(0);
    expect(adapter.getStatus().state).toBe('idle');
  });

  it('terminates the worker when init is aborted before INIT returns', async () => {
    const worker = new FakeWorker();
    let terminated = false;
    worker.terminate = () => {
      terminated = true;
    };
    worker.postMessage = (message) => {
      worker.messages.push(message);
    };
    const controller = new AbortController();
    const adapter = new TenVadAdapter({}, { workerFactory: () => worker });

    const initializing = adapter.init(controller.signal);
    await Promise.resolve();
    controller.abort();

    await expect(initializing).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(terminated).toBe(true);
    expect(worker.onmessage).toBeNull();
    expect(adapter.getStatus().state).toBe('idle');
  });

  it('observes asynchronous changes on a minimal abort-like signal during init', async () => {
    const worker = new FakeWorker('hold-init');
    const signal = { aborted: false };
    const adapter = new TenVadAdapter({}, { workerFactory: () => worker });

    const initializing = adapter.init(signal);
    await Promise.resolve();
    signal.aborted = true;

    await expect(initializing).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(worker.onmessage).toBeNull();
    expect(adapter.getStatus().state).toBe('idle');
  });

  it('does not reset the worker when only non-worker tuning changes', async () => {
    const worker = new FakeWorker();
    const adapter = new TenVadAdapter(
      {
        hopSize: 256,
        threshold: 0.5,
      },
      {
        workerFactory: () => worker,
      },
    );

    await adapter.init();
    const beforeMessageCount = worker.messages.length;

    adapter.updateConfig({
      minSpeechDurationMs: 320,
      minSilenceDurationMs: 96,
      speechPaddingMs: 64,
    });

    expect(worker.messages).toHaveLength(beforeMessageCount);

    await adapter.dispose();
  });

  it('resets cached temporal state when worker-facing config changes', async () => {
    const adapter = new TenVadAdapter(
      {
        hopSize: 256,
        threshold: 0.5,
        minSpeechDurationMs: 16,
      },
      {
        workerFactory: () => new FakeWorker(),
      },
    );

    await adapter.init();
    adapter.process(new Float32Array(512), 0);
    expect(adapter.getStatus().speaking).toBe(true);

    adapter.updateConfig({
      threshold: 0.6,
    });

    expect(adapter.getStatus().probability).toBe(0);
    expect(adapter.getStatus().speaking).toBe(false);

    await adapter.dispose();
  });
});

describe('FireRedVadAdapter', () => {
  it('resolves default FireRed model URLs without adding a fallback loop', () => {
    const defaults = resolveDefaultFireRedVadModelUrls();
    const resolved = resolveFireRedVadModelUrls();

    expect(resolved.scriptUrl).toBe(defaults.scriptUrl);
    expect(resolved.wasmUrl).toBe(defaults.wasmUrl);
    expect(resolved.fallbackScriptUrl).toBeNull();
    expect(resolved.fallbackWasmUrl).toBeNull();
  });

  it('resolves FireRed model URLs from assetBaseUrl overrides', () => {
    const resolved = resolveFireRedVadModelUrls({
      assetBaseUrl: 'https://example.com/vendor/firered-vad/',
    });

    expect(resolved.scriptUrl).toBe(
      'https://example.com/vendor/firered-vad/fireredvad_stream_vad_with_cache.onnx',
    );
    expect(resolved.wasmUrl).toBe('https://example.com/vendor/firered-vad/cmvn.json');
    expect(resolved.fallbackScriptUrl).toBeNull();
    expect(resolved.fallbackWasmUrl).toBeNull();
  });

  it('normalizes unsupported preferred hop sizes to FireRed 10ms hops', () => {
    expect(resolveSupportedFireRedVadHopSize(16000, 512)).toBe(160);
    expect(resolveSupportedFireRedVadHopSize(16000, 200)).toBe(160);
    expect(resolveSupportedFireRedVadHopSize(48000)).toBe(480);
  });

  it('uses the FireRed worker surface', async () => {
    const adapter = new FireRedVadAdapter(
      {
        hopSize: 256,
        minSpeechDurationMs: 40,
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

    await adapter.dispose();
    unsubscribe();
  });

  it('forwards explicit CMVN JSON URLs to the FireRed worker init payload', async () => {
    const worker = new FakeWorker();
    const adapter = new FireRedVadAdapter(
      {
        cmvnJsonUrl: 'https://example.com/cmvn.json',
      },
      {
        workerFactory: () => worker,
      },
    );

    await adapter.init();
    const initMessage = worker.messages.find((message) => message.type === 'INIT');
    expect(initMessage?.payload.cmvnJsonUrl).toBe('https://example.com/cmvn.json');

    await adapter.dispose();
  });

  it('rejects a pending init when disposed before the worker responds', async () => {
    const worker = new FakeWorker();
    worker.postMessage = (message) => {
      worker.messages.push(message);
    };
    const adapter = new FireRedVadAdapter({}, { workerFactory: () => worker });

    const initializing = adapter.init();
    await Promise.resolve();
    await adapter.dispose();

    await expect(initializing).rejects.toThrow('disposed');
    expect(worker.onmessage).toBeNull();
  });

  it('does not create a worker when init is already aborted', async () => {
    let created = 0;
    const adapter = new FireRedVadAdapter(
      {},
      {
        workerFactory: () => {
          created += 1;
          return new FakeWorker();
        },
      },
    );

    await expect(adapter.init({ aborted: true })).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(created).toBe(0);
    expect(adapter.getStatus().state).toBe('idle');
  });

  it('terminates the worker when init is aborted before INIT returns', async () => {
    const worker = new FakeWorker();
    let terminated = false;
    worker.terminate = () => {
      terminated = true;
    };
    worker.postMessage = (message) => {
      worker.messages.push(message);
    };
    const controller = new AbortController();
    const adapter = new FireRedVadAdapter({}, { workerFactory: () => worker });

    const initializing = adapter.init(controller.signal);
    await Promise.resolve();
    controller.abort();

    await expect(initializing).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(terminated).toBe(true);
    expect(worker.onmessage).toBeNull();
    expect(adapter.getStatus().state).toBe('idle');
  });

  it('observes asynchronous changes on a minimal abort-like signal during FireRed init', async () => {
    const worker = new FakeWorker('hold-init');
    const signal = { aborted: false };
    const adapter = new FireRedVadAdapter({}, { workerFactory: () => worker });

    const initializing = adapter.init(signal);
    await Promise.resolve();
    signal.aborted = true;

    await expect(initializing).rejects.toMatchObject({
      name: 'AssetLoadAbortedError',
      code: 'asset-load-aborted',
    });
    expect(worker.onmessage).toBeNull();
    expect(adapter.getStatus().state).toBe('idle');
  });
});
