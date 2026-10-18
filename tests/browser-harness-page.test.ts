import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
import { describe, expect, it, vi } from 'vitest';
import { PcmAudioBuffer } from '../src/audio/audio.js';

function pageCore() {
  const html = readFileSync('tools/browser-harness/harness.html', 'utf8');
  const source = html.split('<script type="module">')[1]!.split('// UI Wire-up')[0]!
    .replace(/\s*import[\s\S]*?from\s+['"][^'"]+['"];?/g, '');
  const transcriber = { pushAudio: vi.fn(async () => null), finalize: vi.fn(async () => ({ text: 'final words', meta: { backendId: 'wasm' } })), dispose: vi.fn() };
  const handle = { supportsStreaming: true, model: { backend: { id: 'wasm' } }, dispose: vi.fn(),
    createStreamingTranscriber: vi.fn(() => transcriber),
    transcribeMonoPcm: vi.fn(async () => ({ text: 'offline words', meta: { backendId: 'wasm' } })),
  };
  const load = vi.fn(async () => handle);
  const loadLocal = vi.fn(async () => handle);
  const pcm = new Float32Array(2800);
  const run = new Function('window', 'document', 'navigator', 'crypto', 'ort', 'loadSpeechModel',
    'PcmAudioBuffer', 'fetch', 'decodeWavToMonoPcm', 'console', 'loadSpeechModelFromLocalEntries', `${source}\nfunction updateTelemetryPills() {}\nreturn ASR_HARNESS;`);
  const api = run({}, { getElementById: () => null }, {}, webcrypto, {}, load,
    PcmAudioBuffer, async () => ({ ok: true, arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer }),
    () => ({ pcm, sampleRate: 8000 }), { log() {} }, loadLocal);
  return { api, transcriber, handle, load, loadLocal };
}

describe('actual browser page core contracts', () => {
  it('accepts File entries without downloading blobs and invalidates reuse on file changes', async () => {
    const { api, loadLocal } = pageCore();
    const entries = [{ path: 'tokenizer/tokenizer.json', basename: 'tokenizer.json', size: 2, lastModified: 1 }];
    const spec = { localFolder: 'model', modelId: 'test', backend: 'wasm', localEntries: entries };
    const first = await api.loadModel(spec);
    const warm = await api.loadModel(spec);
    expect(loadLocal.mock.calls[0]![0].entries).toBe(entries);
    expect(warm.instanceId).toBe(first.instanceId);
    await api.loadModel({ ...spec, localEntries: [{ ...entries[0], lastModified: 2 }] });
    expect(loadLocal).toHaveBeenCalledTimes(2);
  });

  it('uses actual sample rate for offline and streaming, finalizes and disposes', async () => {
    const { api, handle, transcriber } = pageCore();
    const offline = await api.runOffline({ modelId: 'test', backend: 'wasm' });
    expect(handle.transcribeMonoPcm.mock.calls[0]![1]).toBe(8000);
    expect(offline.status).toBe('EXECUTED');
    expect(offline.audio.sha256).toHaveLength(64);
    const stream = await api.runStreaming({ modelId: 'test', backend: 'wasm', chunkMs: 100 });
    expect(transcriber.pushAudio.mock.calls.map(([audio]: any) => [audio.numberOfFrames, audio.sampleRate])).toEqual([[800, 8000], [800, 8000], [800, 8000], [400, 8000]]);
    expect(stream.transcript).toBe('final words');
    expect(transcriber.finalize).toHaveBeenCalledOnce();
    expect(transcriber.dispose).toHaveBeenCalledOnce();
    expect(stream.gpuObservation.instrumented).toBe(false);
    expect(stream.cached).toBe(true);
    expect(stream.instanceId).toBe(offline.instanceId);
  });
  it('rejects unsupported streaming and invalid chunk size instead of simulating', async () => {
    const { api, handle } = pageCore();
    handle.supportsStreaming = false;
    await expect(api.runStreaming({ modelId: 'test' })).rejects.toThrow('does not support native streaming');
    await expect(api.runStreaming({ modelId: 'test', chunkMs: -1 })).rejects.toThrow('positive');
    expect(handle.createStreamingTranscriber).not.toHaveBeenCalled();
  });
  it('does not reuse handles across changed backend or model options', async () => {
    const { api, load } = pageCore();
    const first = await api.loadModel({ modelId: 'test', backend: 'wasm', options: { cpuThreads: 1 } });
    const second = await api.loadModel({ modelId: 'test', backend: 'wasm', options: { cpuThreads: 2 } });
    const third = await api.loadModel({ modelId: 'test', backend: 'webgpu', options: { cpuThreads: 2 } });
    expect(load).toHaveBeenCalledTimes(3);
    expect(new Set([first.instanceId, second.instanceId, third.instanceId]).size).toBe(3);
    expect(first.loadFingerprint).not.toBe(second.loadFingerprint);
    expect(load.mock.calls[1]![0].options.cpuThreads).toBe(2);
  });
  it('keeps failed cleanup visible and does not discard its owned handle', async () => {
    const { api, handle } = pageCore();
    await api.loadModel({ modelId: 'test' });
    handle.dispose.mockRejectedValue(new Error('release failed'));
    await expect(api.loadModel({ modelId: 'test', forceReload: true })).rejects.toThrow('release failed');
    await expect(api.disposeAll()).rejects.toThrow('failed to dispose');
    expect(api.activeModels.size).toBe(1);
  });
});
