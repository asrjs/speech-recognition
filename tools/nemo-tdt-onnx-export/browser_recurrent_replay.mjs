// Conditional frozen-encoder control; production library owns all decoding.
import {
  PcmAudioBuffer,
  createSpeechRuntime,
  createWasmBackend,
  createWebGpuBackend,
  loadSpeechModel,
} from '/dist/index.js';
import { createNemoTdtModelFamily } from '/dist/models/nemo-tdt/index.js';
import { initOrt } from '/dist/models/nemo-tdt/ort.js';
import { observeSessionRuns } from '/observe-session.mjs';

globalThis.replayLibraryRecurrence = async (fixture, backend) => {
  if (!['wasm', 'webgpu'].includes(backend)) throw new Error('Explicit browser backend required');
  const realEncoder = fixture.realEncoder === true;
  const encoderBackend = realEncoder ? fixture.encoderBackend : 'wasm';
  if (!['wasm', 'webgpu'].includes(encoderBackend)) throw new Error('Explicit encoder backend required');
  const usesWebGpu = backend === 'webgpu' || encoderBackend === 'webgpu';
  const receipt = {
    backendRequested: backend,
    encoderBackend,
    decoderBackend: backend,
    features: 'EXECUTED_NOT_ACCEPTED',
    realEncoder: realEncoder ? 'EXECUTED_NOT_COMPONENT_ACCEPTED' : 'BYPASSED_FROZEN_ORACLE',
    runs: [],
    cleanupErrors: [],
    gpuDispatches: 0,
    crossOriginIsolated: globalThis.crossOriginIsolated,
  };
  const originals = [];
  let runtime, loaded, observer;
  try {
    if (usesWebGpu) {
      const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('WEBGPU_NO_ADAPTER');
      receipt.adapter = Object.fromEntries(
        ['vendor', 'architecture', 'device', 'description'].map((k) => [
          k,
          adapter.info?.[k] ?? null,
        ]),
      );
      receipt.adapterLimits = Object.fromEntries(
        ['maxBufferSize', 'maxStorageBufferBindingSize'].map((k) => [k, adapter.limits[k]]),
      );
      for (const name of ['dispatchWorkgroups', 'dispatchWorkgroupsIndirect']) {
        const prototype = GPUComputePassEncoder.prototype,
          original = prototype[name];
        originals.push([prototype, name, original]);
        prototype[name] = function (...args) {
          receipt.gpuDispatches++;
          return original.apply(this, args);
        };
      }
    }
    const ort = await initOrt(usesWebGpu ? 'webgpu' : 'wasm', {
      wasmPaths: `${location.origin}/ort-dist/`,
      cpuThreads: 1,
    });
    receipt.ortVersions = ort.env.versions;
    runtime = createSpeechRuntime();
    runtime.registerBackend(createWasmBackend());
    if (usesWebGpu) runtime.registerBackend(createWebGpuBackend());
    runtime.registerModelFamily(createNemoTdtModelFamily());
    const source = {
      kind: 'direct',
      encoderBackend,
      decoderBackend: backend,
      preprocessorBackend: 'js',
      cpuThreads: 1,
      wasmPaths: `${location.origin}/ort-dist/`,
      artifacts: {
        encoderUrl: `${location.origin}/${realEncoder ? 'encoder-model.onnx' : 'frozen-encoder.onnx'}`,
        ...(realEncoder ? { encoderFilename: 'encoder-model.onnx',
          encoderDataUrl: `${location.origin}/encoder-model.onnx.data` } : {}),
        decoderUrl: `${location.origin}/graph.onnx`,
        tokenizerUrl: `${location.origin}/vocab.txt`,
      },
    };
    receipt.source = source;
    const started = performance.now();
    loaded = await loadSpeechModel({
      runtime,
      family: 'nemo-tdt',
      modelId: fixture.modelId,
      backend,
      options: { config: fixture.config, source },
    });
    receipt.modelLoadMs = performance.now() - started;
    receipt.loaded = { backend: loaded.model.backend.id, family: loaded.model.info.family };
    if (receipt.loaded.backend !== backend || receipt.loaded.family !== 'nemo-tdt')
      throw new Error('Loaded backend/family differs');
    const bytes = Uint8Array.from(atob(fixture.pcmF32leBase64), (character) =>
      character.charCodeAt(0),
    );
    if (!bytes.length || bytes.length % 4) throw new Error('Invalid f32le PCM bytes');
    const view = new DataView(bytes.buffer),
      audio = new Float32Array(bytes.length / 4);
    for (let index = 0; index < audio.length; index++)
      audio[index] = view.getFloat32(index * 4, true);
    const pcm = PcmAudioBuffer.fromMono(audio, fixture.sampleRate);
    const digest = await crypto.subtle.digest('SHA-256', pcm.channels[0]);
    receipt.pcmSha256 = Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, '0'),
    ).join('');
    if (receipt.pcmSha256 !== fixture.pcmSha256) throw new Error('Public PCM bytes differ');
    for (let repeat = 0; repeat < 2; repeat++) {
      const before = receipt.gpuDispatches,
        start = performance.now();
      const traceEnabled = fixture.traceHead === true && repeat === 0;
      if (traceEnabled)
        observer = observeSessionRuns(
          ort.InferenceSession,
          ['encoder_outputs', 'targets', 'target_length', 'input_states_1', 'input_states_2'],
          ['outputs', 'output_states_1', 'output_states_2', 'prednet_lengths'],
          fixture.maxTraceCalls,
        );
      let result,
        headCalls = [];
      try {
        result = await loaded.transcribe(pcm, {
          detail: 'detailed',
          responseFlavor: 'canonical+native',
          returnTokenIds: true,
          returnFrameIndices: true,
          returnTdtSteps: true,
          returnDecoderState: true,
          gridBatching: false,
          disableWindowing: true,
        });
      } finally {
        if (observer) {
          observer.restore();
          headCalls = observer.calls;
          observer = undefined;
        }
      }
      const state = result.native?.decoderState;
      receipt.runs.push({
        repeat,
        traceEnabled,
        headCalls,
        observerRestored: !observer,
        transcribeMs: performance.now() - start,
        gpuDispatches: receipt.gpuDispatches - before,
        text: result.canonical.text,
        metrics: result.canonical.meta.metrics,
        tokenIds: result.native?.debug?.tokenIds,
        frameIndices: result.native?.debug?.frameIndices,
        tdtSteps: result.native?.debug?.tdtSteps,
        decoderState: state
          ? {
              h: { dtype: 'float32', shape: [...state.dims1], data: Array.from(state.s1) },
              c: { dtype: 'float32', shape: [...state.dims2], data: Array.from(state.s2) },
            }
          : null,
      });
    }
  } catch (error) {
    receipt.error = String(error?.stack ?? error);
  } finally {
    try {
      observer?.restore();
    } catch (error) {
      receipt.cleanupErrors.push(String(error));
    }
    try {
      await loaded?.dispose();
    } catch (error) {
      receipt.cleanupErrors.push(String(error));
    }
    try {
      await runtime?.dispose();
    } catch (error) {
      receipt.cleanupErrors.push(String(error));
    }
    for (const [prototype, name, original] of originals) prototype[name] = original;
  }
  return receipt;
};
