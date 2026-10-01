// Recorded graph calls only; no token search or alternative ASR decoder.
import { initOrt, createOrtSession, releaseOrtSession, disposeOrtOutputs } from '/dist/models/nemo-tdt/ort.js';

globalThis.replayRecordedHead = async (fixture, backend) => {
  if (!['wasm', 'webgpu'].includes(backend)) throw new Error('Explicit browser backend required');
  const receipt = { backendRequested: backend, runs: [], cleanupErrors: [], gpuDispatches: 0 };
  let session;
  const feeds = [];
  const originals = [];
  try {
    if (backend === 'webgpu') {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('WEBGPU_NO_ADAPTER');
      receipt.adapter = { ...Object.fromEntries(['vendor', 'architecture', 'device', 'description'].map(k => [k, adapter.info?.[k] ?? null])) };
      for (const name of ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect']) {
        const prototype = GPUComputePassEncoder.prototype;
        const original = prototype[name];
        originals.push([prototype, name, original]);
        prototype[name] = function (...args) { receipt.gpuDispatches++; return original.apply(this, args); };
      }
    }
    const ort = await initOrt(backend, { wasmPaths: `${location.origin}/ort-dist/`, cpuThreads: 1 });
    receipt.versions = ort.env.versions;
    receipt.crossOriginIsolated = globalThis.crossOriginIsolated;
    const started = performance.now();
    session = await createOrtSession(ort, `${location.origin}/graph.onnx`, { backendId: backend, cpuThreads: 1 });
    receipt.sessionCreateMs = performance.now() - started;
    for (const step of fixture.steps) {
      feeds.push(Object.fromEntries(Object.entries(step.feeds).map(([name, value]) => [name,
        new ort.Tensor(value.dtype, value.dtype === 'int32' ? Int32Array.from(value.data) : Float32Array.from(value.data), value.shape)])));
    }
    for (let repeat = 0; repeat < 2; repeat++) {
      for (let index = 0; index < feeds.length; index++) {
        let output;
        try {
          const before = receipt.gpuDispatches;
          const start = performance.now();
          output = await session.run(feeds[index]);
          const values = {};
          for (const [name, tensor] of Object.entries(output)) {
            values[name] = { dtype: tensor.type, shape: [...tensor.dims], data: Array.from(tensor.data) };
          }
          receipt.runs.push({ repeat, step: index, runAndReadbackMs: performance.now() - start,
            gpuDispatches: receipt.gpuDispatches - before, outputs: values });
        } finally { disposeOrtOutputs(output); }
      }
    }
  } catch (error) {
    receipt.error = String(error?.stack ?? error);
  } finally {
    for (const tensors of feeds) {
      for (const tensor of Object.values(tensors)) {
        try { tensor.dispose?.(); } catch (error) { receipt.cleanupErrors.push(String(error)); }
      }
    }
    try { await releaseOrtSession(session); } catch (error) { receipt.cleanupErrors.push(String(error)); }
    for (const [prototype, name, original] of originals) prototype[name] = original;
  }
  return receipt;
};
