/** Family-owned browser smoke: all inference and decoding stay in the public library API. */
import {
  createSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '@asrjs/speech-recognition/browser';
import { matchesSmokeTranscript } from './smoke-transcript.mjs';
import { createWindowTokenFixture } from './window-token-fixture.mjs';

const field = (id) => document.getElementById(id);
field('run').addEventListener('click', async () => {
  const receipt = { status: 'FAIL_LOCAL_HYBRID_SMOKE', disposed: false, results: [] };
  let model;
  let audioContext;
  field('run').disabled = true;
  field('status').textContent = 'Running';
  try {
    const files = [...field('model').files];
    const audio = field('audio').files[0];
    const expected = field('expected').value.trim();
    if (!files.length || !audio || !expected)
      throw new Error('Model, audio and expected phrase required');
    if (!crossOriginIsolated) throw new Error('Cross-origin isolation required');
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter required');
    receipt.adapter = {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      fallback: adapter.info.isFallbackAdapter,
    };
    const configResponse = await fetch('/harness/local-engine.json');
    if (!configResponse.ok) throw new Error('Local engine configuration unavailable');
    const engine = await configResponse.json();
    receipt.engine = engine;
    const loadStart = performance.now();
    model = await loadSpeechModelFromLocalEntries({
      modelId: 'parakeet-tdt-0.6b-v3',
      entries: createSpeechModelLocalEntries(files),
      backend: 'webgpu-hybrid',
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
      encoderQuant: field('precision').value,
      decoderQuant: 'fp32',
      preprocessorBackend: 'js',
      wasmPaths: engine.wasmPaths,
    });
    receipt.loadMs = performance.now() - loadStart;
    receipt.selection = model.selection;
    audioContext = new AudioContext({ sampleRate: 16000 });
    const decoded = await audioContext.decodeAudioData(await audio.arrayBuffer());
    if (decoded.numberOfChannels !== 1 || decoded.sampleRate !== 16000) {
      throw new Error('Smoke fixture must decode to mono 16kHz');
    }
    const pcm = decoded.getChannelData(0);
    const digest = await crypto.subtle.digest('SHA-256', pcm);
    receipt.audio = {
      samples: pcm.length,
      sampleRate: decoded.sampleRate,
      pcmSha256: [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join(''),
    };
    let transcriptionPcm = pcm;
    let transcriptionOptions;
    let expectedTranscript = expected;
    if (field('window-tokens').checked) {
      const fixture = createWindowTokenFixture(pcm);
      transcriptionPcm = fixture.pcm;
      transcriptionOptions = fixture.options;
      expectedTranscript = `${expected} ${expected}`;
      receipt.windowing = fixture.windowing;
    }
    for (let iteration = 0; iteration < 2; iteration++) {
      if (iteration === 1 && field('cancel').checked) {
        const controller = new AbortController();
        const cancellation = { requested: false, rejected: false, completeEmitted: false };
        receipt.cancellation = cancellation;
        try {
          await model.transcribeMonoPcm(pcm, 16000, {
            signal: controller.signal,
            onProgress(event) {
              if (event.stage === 'complete') cancellation.completeEmitted = true;
              if (
                event.stage === 'decode' && event.completedUnits > 0 &&
                event.completedUnits < event.totalUnits && !cancellation.requested
              ) {
                cancellation.requested = true;
                cancellation.completedUnits = event.completedUnits;
                cancellation.totalUnits = event.totalUnits;
                controller.abort();
              }
            },
          });
        } catch (error) {
          cancellation.errorName = error.name;
          cancellation.stageId = error.stageId;
          cancellation.rejected = error.name === 'PipelineAbortedError' && error.stageId === 'decode';
          if (!cancellation.rejected) throw error;
        }
        if (!cancellation.requested || !cancellation.rejected || cancellation.completeEmitted)
          throw new Error('Requested in-flight decode cancellation did not reject without completion');
      }
      const start = performance.now();
      const result = await model.transcribeMonoPcm(transcriptionPcm, 16000, transcriptionOptions);
      receipt.results.push({
        text: result.text,
        meta: result.meta,
        ...(transcriptionOptions ? { tokens: result.tokens, words: result.words, warnings: result.warnings } : {}),
        wallMs: performance.now() - start,
      });
      if (
        result.meta.metrics?.encoderBackend !== 'webgpu' ||
        result.meta.metrics?.decoderBackend !== 'wasm'
      ) {
        throw new Error('Actual component providers do not match requested hybrid placement');
      }
      if (!matchesSmokeTranscript(result.text, expectedTranscript))
        throw new Error('Full normalized transcript differs from expected smoke phrase');
    }
    if (receipt.results[0].text !== receipt.results[1].text)
      throw new Error('Repeated transcript differs');
    receipt.inferencePassed = true;
  } catch (error) {
    receipt.error = String(error);
  } finally {
    try {
      if (model) {
        await model.dispose();
        receipt.disposed = true;
      }
    } catch (error) {
      receipt.cleanupError = String(error);
    }
    try {
      if (audioContext) await audioContext.close();
    } catch (error) {
      receipt.audioCleanupError = String(error);
    }
    if (
      receipt.inferencePassed &&
      receipt.disposed &&
      !receipt.error &&
      !receipt.cleanupError &&
      !receipt.audioCleanupError
    ) {
      receipt.status = 'PASS_LOCAL_HYBRID_SMOKE';
    }
    field('result').textContent = JSON.stringify(receipt, null, 2);
    field('status').textContent = receipt.status;
    field('run').disabled = false;
  }
});
