// Real encoder component replay through the library bridge, no ASR decoder.
import { initOrt, createOrtSession, releaseOrtSession, disposeOrtOutputs } from '/dist/models/nemo-tdt/ort.js';

globalThis.replayEncoder = async (fixture, backend) => {
  const receipt = { backendRequested: backend, runs: [], cleanupErrors: [], gpuDispatches: 0,
    crossSessionRuns: [], crossSessionRequested: Boolean(fixture.crossSession),
    coResidentRuns: [], coResidentRequested: Boolean(fixture.coResidentDecoder) };
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
    const inputs = [{ featureSource: 'official', values: new Float32Array(bytes.buffer) }];
    if (fixture.pcmF32le !== undefined) {
      const pcmBytes = Uint8Array.from(atob(fixture.pcmF32le), c => c.charCodeAt(0));
      const hash = await crypto.subtle.digest('SHA-256', pcmBytes);
      receipt.pcmSha256 = [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
      if (receipt.pcmSha256 !== fixture.pcmSha256) throw new Error('Captured PCM byte identity differs');
      const { JSMelProcessor } = await import('/dist/audio/js-mel.js');
      const production = new JSMelProcessor({ nMels: fixture.shape[1],
        normalization: 'per_feature', validLengthMode: 'onnx' }).process(new Float32Array(pcmBytes.buffer));
      if (production.frameCount !== fixture.shape[2] || production.length !== fixture.length ||
          production.features.length !== inputs[0].values.length)
        throw new Error('Production feature shape/length differs');
      receipt.productionFeatures = { dtype: 'float32', shape: fixture.shape,
        data: Array.from(production.features) };
      receipt.productionLength = production.length;
      inputs.push({ featureSource: 'production-js', values: production.features });
    }
    for (const input of inputs) for (let repeat = 0; repeat < 2; repeat++) {
      const feeds = { audio_signal: new ort.Tensor('float32', input.values, fixture.shape),
        length: new ort.Tensor('int64', BigInt64Array.from([BigInt(fixture.length)]), [1]) };
      let outputs;
      try {
        const before = receipt.gpuDispatches, began = performance.now();
        outputs = await session.run(feeds);
        const values = {};
        for (const [name, tensor] of Object.entries(outputs))
          values[name] = { dtype: tensor.type, shape: [...tensor.dims], data: Array.from(tensor.data, Number) };
        receipt.runs.push({ repeat, featureSource: input.featureSource, outputs: values, runAndReadbackMs: performance.now() - began,
          gpuDispatches: receipt.gpuDispatches - before });
      } finally {
        disposeOrtOutputs(outputs);
        for (const tensor of Object.values(feeds)) tensor.dispose?.();
      }
    }

    // Opt-in cross-session control. Within-session repeats are bitwise exact,
    // which says nothing about variation between two separately created
    // sessions. The hybrid run holds an encoder session and a decoder session
    // together; this creates a second identical encoder session and compares.
    if (fixture.crossSession) {
      try {
        const second = await createOrtSession(ort, `${location.origin}/graph.onnx`, {
          backendId: backend, cpuThreads: 1,
          externalDataUrl: `${location.origin}/encoder-model.onnx.data`,
          externalDataPath: 'encoder-model.onnx.data',
        });
        try {
          receipt.resolvedEncoderSessionOptions = {
            backendId: backend, cpuThreads: 1,
            externalDataPath: 'encoder-model.onnx.data',
            graph: 'graph.onnx',
          };
          for (const input of inputs) {
            const feeds = { audio_signal: new ort.Tensor('float32', input.values, fixture.shape),
              length: new ort.Tensor('int64', BigInt64Array.from([BigInt(fixture.length)]), [1]) };
            let outputs;
            try {
              outputs = await second.run(feeds);
              const values = {};
              for (const [name, tensor] of Object.entries(outputs))
                values[name] = { dtype: tensor.type, shape: [...tensor.dims], data: Array.from(tensor.data, Number) };
              receipt.crossSessionRuns.push({ featureSource: input.featureSource, outputs: values });
            } finally {
              disposeOrtOutputs(outputs);
              for (const tensor of Object.values(feeds)) tensor.dispose?.();
            }
          }
        } finally {
          try { await releaseOrtSession(second); } catch (error) { receipt.cleanupErrors.push(String(error)); }
        }
      } catch (error) { receipt.crossSessionError = String(error?.stack ?? error); }
    }

    // Opt-in co-residency control. The hybrid run holds an encoder session and
    // a WASM decoder session together; this holds a decoder session alongside
    // the encoder and re-runs it, testing whether co-residency alone changes
    // encoder output. The decoder session is created, never executed.
    if (fixture.coResidentDecoder) {
      try {
        const decoderSession = await createOrtSession(ort, `${location.origin}/decoder-graph.onnx`, {
          backendId: 'wasm', cpuThreads: 1,
        });
        try {
          for (const input of inputs) {
            const feeds = { audio_signal: new ort.Tensor('float32', input.values, fixture.shape),
              length: new ort.Tensor('int64', BigInt64Array.from([BigInt(fixture.length)]), [1]) };
            let outputs;
            try {
              outputs = await session.run(feeds);
              const values = {};
              for (const [name, tensor] of Object.entries(outputs))
                values[name] = { dtype: tensor.type, shape: [...tensor.dims], data: Array.from(tensor.data, Number) };
              receipt.coResidentRuns.push({ featureSource: input.featureSource, outputs: values });
            } finally {
              disposeOrtOutputs(outputs);
              for (const tensor of Object.values(feeds)) tensor.dispose?.();
            }
          }
        } finally {
          try { await releaseOrtSession(decoderSession); }
          catch (error) { receipt.cleanupErrors.push(String(error)); }
        }
      } catch (error) { receipt.coResidentError = String(error?.stack ?? error); }
    }
  } catch (error) { receipt.error = String(error?.stack ?? error); }
  finally {
    try { await releaseOrtSession(session); } catch (error) { receipt.cleanupErrors.push(String(error)); }
    for (const [name, original] of originals) GPUComputePassEncoder.prototype[name] = original;
  }
  return receipt;
};
