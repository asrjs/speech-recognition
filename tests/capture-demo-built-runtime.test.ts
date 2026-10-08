import { afterEach, expect, it, vi } from 'vitest';
import { observeCaptureWorkers, runCapturedController } from '../examples/demo/src/capture-controller.js';

afterEach(() => vi.unstubAllGlobals());

it('observes only the requested native worker and preserves constructor arguments', () => {
  class NativeWorker {
    terminateCalls = 0;
    terminate = vi.fn(() => { this.terminateCalls += 1; });
    listener: (event: any) => void = () => {};
    constructor(public url: unknown, public options: unknown) {}
    addEventListener(_type: string, listener: (event: any) => void) { this.listener = listener; }
  }
  vi.stubGlobal('Worker', NativeWorker);
  vi.stubGlobal('location', { href: 'https://localhost/test.html' });
  const observer = observeCaptureWorkers('/dist/runtime/ten-vad-worker.js');
  try {
    const options: WorkerOptions = { type: 'module' };
    const worker = new Worker('/dist/runtime/ten-vad-worker.js?worker_file&type=module', options) as any;
    new Worker('/unrelated.js');
    new Worker('https://other.invalid/dist/runtime/ten-vad-worker.js');
    expect(worker).toBeInstanceOf(NativeWorker);
    expect(worker.options).toBe(options);
    expect(observer.workers).toHaveLength(1);
    worker.listener({ data: { type: 'RESULT' } });
    worker.listener({ data: { type: 'DISPOSE', payload: { success: true } } });
    expect(observer.acknowledgements).toEqual([{ type: 'DISPOSE', payload: { success: true } }]);
    worker.terminate();
    expect(observer.workers[0].terminated).toBe(true);
    expect(worker.terminateCalls).toBe(1);
  } finally { observer.restore(); }
  expect(globalThis.Worker).toBe(NativeWorker);
  observer.restore();
});

it('does not overwrite another observer owner on restoration', () => {
  vi.stubGlobal('Worker', class {});
  vi.stubGlobal('location', { href: 'https://localhost/' });
  const observer = observeCaptureWorkers('/worker.js');
  const replacement = class {};
  globalThis.Worker = replacement as any;
  expect(() => observer.restore()).toThrow('ownership changed');
  expect(globalThis.Worker).toBe(replacement);
});

it('uses the supplied public runtime without injecting a default-worker factory', async () => {
  const create = vi.fn((_options: unknown) => { throw new Error('creation sentinel'); });
  await expect(runCapturedController({ receipt: {}, transcribe: vi.fn(), micMode: 'speech-detect',
    library: { createBrowserRealtimeMicrophoneController: create, startMicrophoneCapture: vi.fn() },
    workerObserver: { workers: [], acknowledgements: [] },
  })).rejects.toThrow('creation sentinel');
  expect(create).toHaveBeenCalledOnce();
  expect(create.mock.calls[0][0]).not.toHaveProperty('tenVadOptions');
});

it('rejects a mixed default observer and injected worker before controller creation', async () => {
  const create = vi.fn();
  await expect(runCapturedController({ receipt: {}, transcribe: vi.fn(), micMode: 'speech-detect',
    workerUrl: '/source-worker.ts', workerObserver: { workers: [], acknowledgements: [] },
    library: { createBrowserRealtimeMicrophoneController: create, startMicrophoneCapture: vi.fn() },
  })).rejects.toThrow('Invalid default-worker observer');
  expect(create).not.toHaveBeenCalled();
});
