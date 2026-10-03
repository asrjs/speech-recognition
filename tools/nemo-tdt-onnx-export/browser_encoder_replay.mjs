// Real encoder component replay through the library bridge, no ASR decoder.
import { initOrt, createOrtSession, releaseOrtSession, disposeOrtOutputs } from '/dist/models/nemo-tdt/ort.js';

globalThis.replayEncoder = async (fixture, backend) => {
  const receipt = { backendRequested: backend, runs: [], cleanupErrors: [], gpuDispatches: 0 };
  let session;
  const originals = [];
  try {
    if (!['wasm', 'webgpu'].includes(backend)) throw new Error('Explicit backend required');
    if (backend === 'webgpu') {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('WEBGPU_NO_ADAPTER');
      receipt.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
        fallback: adapter.info.isFallbackAdapter, maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
      for (const name of ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect']) {
        const original = GPUComputePassEncoder.prototype[name];
        originals.push([name, original]);
        GPUComputePassEncoder.prototype[name] = function (...args) {
          receipt.gpuDispatches++;
          return original.apply(this, args);
        };
      }
    }
    const ort = await initOrt(backend, { wasmPaths: `${location.origin}/ort-dist/`, cpuThreads: 1 });
    receipt.versions = ort.env.versions;
    receipt.crossOriginIsolated = crossOriginIsolated;
    const start = performance.now();
    session = await createOrtSession(ort, `${location.origin}/graph.onnx`, {
      backendId: backend, cpuThreads: 1, externalDataUrl: `${location.origin}/encoder-model.onnx.data`,
      externalDataPath: 'encoder-model.onnx.data',
    });
    receipt.sessionCreateMs = performance.now() - start;
    const bytes = Uint8Array.from(atob(fixture.featuresF32le), c => c.charCodeAt(0));
    if (bytes.byteLength !== fixture.shape.reduce((a, b) => a * b, 1) * 4)
      throw new Error('Feature byte count differs');
    for (let repeat = 0; repeat < 2; repeat++) {
      const feeds = { audio_signal: new ort.Tensor('float32', new Float32Array(bytes.buffer), fixture.shape),
        length: new ort.Tensor('int64', BigInt64Array.from([BigInt(fixture.length)]), [1]) };
      let outputs;
      try {
        const before = receipt.gpuDispatches, began = performance.now();
        outputs = await session.run(feeds);
        const values = {};
        for (const [name, tensor] of Object.entries(outputs))
          values[name] = { dtype: tensor.type, shape: [...tensor.dims], data: Array.from(tensor.data, Number) };
        receipt.runs.push({ repeat, outputs: values, runAndReadbackMs: performance.now() - began,
          gpuDispatches: receipt.gpuDispatches - before });
      } finally {
        disposeOrtOutputs(outputs);
        for (const tensor of Object.values(feeds)) tensor.dispose?.();
      }
    }
  } catch (error) { receipt.error = String(error?.stack ?? error); }
  finally {
    try { await releaseOrtSession(session); } catch (error) { receipt.cleanupErrors.push(String(error)); }
    for (const [name, original] of originals) GPUComputePassEncoder.prototype[name] = original;
  }
  return receipt;
};
