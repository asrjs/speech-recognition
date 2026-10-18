import { describe, expect, it } from 'vitest';
import { JFK_ORACLE, verifyRun, verifyRuns, verificationExitCode } from '../tools/browser-harness/verification.mjs';
import { runMatrix } from '../tools/browser-harness/runner.mjs';

function receipt(overrides = {}) {
  return {
    status: 'EXECUTED', mode: 'offline', transcript: JFK_ORACLE.text,
    audio: { fixture: JFK_ORACLE.fixture, sha256: JFK_ORACLE.sha256, samples: 16000, sampleRate: 16000 },
    metrics: { inferenceMs: 20, gpuDispatches: 4 },
    gpuObservation: { instrumented: true, available: true },
    reportedBackendId: 'webgpu', cached: false, instanceId: 'model-1', ...overrides,
  };
}
const spec = { mode: 'offline', backend: 'webgpu', oracle: JFK_ORACLE };

describe('browser harness verification gates', () => {
  it('accepts complete normalized text with bound fixture and observed GPU work', () => {
    expect(verifyRun(receipt({ transcript: JFK_ORACLE.text.toUpperCase().replaceAll(',', ';') }), spec).status).toBe('PASS');
  });
  it.each(['fellow americans ask country', 'And so my fellow Americans askk not what your country can do for you ask what you can do for your country', JFK_ORACLE.text + ' extra words', ''])('rejects altered or incomplete text: %s', transcript => {
    expect(verifyRun(receipt({ transcript }), spec).status).toBe('FAIL');
  });
  it('does not manufacture a match when no oracle was requested', () => {
    const result = verifyRun(receipt(), { ...spec, oracle: null });
    expect(result.status).toBe('UNVERIFIED');
    expect(result.transcript.matchExpected).toBeNull();
  });
  it('rejects the correct text from the wrong audio', () => {
    expect(verifyRun(receipt({ audio: { ...receipt().audio, sha256: '0'.repeat(64) } }), spec).status).toBe('FAIL');
  });
  it.each([0, undefined])('requires observed GPU work instead of just navigator.gpu (%s)', gpuDispatches => {
    expect(verifyRun(receipt({ metrics: { inferenceMs: 20, gpuDispatches } }), spec).status).toBe('UNVERIFIED');
  });
  it('does not accept an unavailable dispatch instrument', () => {
    expect(verifyRun(receipt({ gpuObservation: { available: true, instrumented: false } }), spec).status).toBe('UNVERIFIED');
  });
  it('rejects reported WASM fallback under a WebGPU request', () => {
    expect(verifyRun(receipt({ reportedBackendId: 'wasm' }), spec).status).toBe('FAIL');
  });
  it('checks cold and warm output independently', () => {
    expect(verifyRuns(receipt({ transcript: 'wrong' }), receipt({ cached: true }), spec).status).toBe('FAIL');
  });
  it('zero load time does not prove reuse', () => {
    expect(verifyRuns(receipt(), receipt({ metrics: { inferenceMs: 20, gpuDispatches: 4, loadMs: 0 } }), spec).status).toBe('FAIL');
    expect(verifyRuns(receipt(), receipt({ cached: true, instanceId: 'different-model' }), spec).warmSessionReused).toBe(false);
  });
  it('checks both offline and streaming transcripts', () => {
    const both = { offline: receipt(), streaming: receipt({ mode: 'streaming', transcript: 'wrong' }) };
    expect(verifyRuns(both, null, { ...spec, mode: 'both' }).status).toBe('FAIL');
  });
  it('fails missing or wrong execution modes', () => {
    expect(verifyRun(receipt(), { ...spec, mode: 'streaming' }).status).toBe('FAIL');
    expect(verifyRuns({}, null, { ...spec, mode: 'both' }).status).toBe('FAIL');
  });
  it('distinguishes acceptance, missing evidence, and failure exit codes', () => {
    expect(['PASS', 'UNVERIFIED', 'FAIL', 'PARTIAL_PASS'].map(verificationExitCode)).toEqual([0, 2, 1, 1]);
  });
});

describe('matrix uses the requested execution path', () => {
  function harness() {
    const calls: string[] = [];
    return { calls,
      async runOffline(options: { forceReload: boolean }) { calls.push('offline'); return receipt({ cached: !options.forceReload }); },
      async runStreaming(options: { forceReload: boolean }) { calls.push('streaming'); return receipt({ mode: 'streaming', cached: !options.forceReload }); },
      async disposeAll() { calls.push('dispose'); },
      async probeWebGpu() { return { adapter: {}, limits: {} }; },
    };
  }
  it.each(['offline', 'streaming', 'both'])('runs %s in cold and warm phases', async mode => {
    const api = harness();
    const result = await runMatrix(api, { ...spec, mode, dirs: ['one'] });
    expect(result.status).toBe('PASS');
    expect(api.calls).toEqual(mode === 'both' ? ['offline', 'streaming', 'offline', 'streaming', 'dispose'] : [mode, mode, 'dispose']);
  });
  it('fails a mixed matrix and preserves the failed model', async () => {
    const api = harness();
    const run = api.runOffline;
    api.runOffline = async (options: any) => { if (options.localFolder === 'bad') throw new Error('load failed'); return run(options); };
    const result = await runMatrix(api, { ...spec, dirs: ['one', 'bad'] });
    expect(result.status).toBe('FAIL');
    expect(result.results.map((item: any) => item.status)).toEqual(['PASS', 'FAIL']);
    expect(verificationExitCode(result.status)).toBe(1);
  });
  it.each(['offline', 'streaming', 'both'])('does not label cold-only %s metrics as warm', async mode => {
    const api = harness();
    const result = await runMatrix(api, { ...spec, mode, dirs: ['one'], warmRerun: false });
    const item = result.results[0];
    expect(item.warm).toBeNull();
    expect(item.summaryPhase).toBe('cold');
    expect(item.metrics.coldInferenceMs).toBe(20);
    expect(item.metrics.warmLoadMs).toBeNull();
    expect(item.metrics.warmInferenceMs).toBeNull();
    expect(api.calls).toEqual(mode === 'both' ? ['offline', 'streaming', 'dispose'] : [mode, 'dispose']);
  });
  it('rejects an empty matrix and invalid mode', async () => {
    await expect(runMatrix(harness(), { ...spec, dirs: [] })).rejects.toThrow();
    expect((await runMatrix(harness(), { ...spec, mode: 'typo', dirs: ['one'] })).status).toBe('FAIL');
  });
  it('does not suppress model disposal errors', async () => {
    const api = harness(); api.disposeAll = async () => { throw new Error('release failed'); };
    const result = await runMatrix(api, { ...spec, dirs: ['one'] });
    expect(result.status).toBe('FAIL');
    expect(result.results[0].disposalError).toBe('release failed');
  });
});
