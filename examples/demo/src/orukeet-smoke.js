/** Orukeet demo: local-folder load, repeated transcription and awaited disposal. */
import {
  createSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '../../../src/runtime/local-browser.ts';
import { decodeWavToMonoPcm } from '../../../src/browser-media.ts';

const MODEL_ID = 'oruk/orukeet';
const field = (id) => document.getElementById(id);
const query = new URLSearchParams(location.search);
for (const id of ['backend', 'precision', 'repeat']) {
  if (query.has(id)) field(id).value = query.get(id);
}
let autoStarted = false;
function maybeAutoRun() {
  if (query.get('auto') === '1' && !autoStarted && field('model').files.length && field('audio').files.length) {
    autoStarted = true;
    field('run').click();
  }
}
field('model').addEventListener('change', maybeAutoRun);
field('audio').addEventListener('change', maybeAutoRun);
const status = (message) => {
  field('status').textContent = message;
};

async function decodeFixture(file) {
  if ((query.get('audioStrategy') ?? 'wav') === 'wav') {
    const decoded = decodeWavToMonoPcm(await file.arrayBuffer());
    if (decoded.numberOfChannels !== 1 || decoded.sampleRate !== 16000) {
      throw new Error('Deterministic fixture must be mono WAV at 16000 Hz; no implicit resampling');
    }
    return decoded.pcm;
  }
  if (query.get('audioStrategy') !== 'audiocontext') throw new Error('Unknown audio strategy');
  const audioContext = new AudioContext({ sampleRate: 16000 });
  try {
    const decoded = await audioContext.decodeAudioData(await file.arrayBuffer());
    if (decoded.numberOfChannels !== 1) throw new Error('Fixture must be mono');
    return decoded.getChannelData(0);
  } finally {
    await audioContext.close();
  }
}

field('run').addEventListener('click', async () => {
  const result = { status: 'FAIL_ORUKEET_SMOKE', disposed: false, transcripts: [] };
  field('result').textContent = '';
  field('run').disabled = true;
  let entries = [];
  let model = null;

  try {
    const files = [...field('model').files];
    const audioFile = field('audio').files[0];
    const repeat = Math.max(1, Number.parseInt(field('repeat').value, 10) || 1);
    const verifyCancellation = query.get('cancel') === '1';
    if (verifyCancellation && repeat < 2) throw new Error('Cancellation requires two successful repeats to prove reuse');
    if (!files.length || !audioFile) throw new Error('Model folder and audio fixture are required');

    if (field('backend').value === 'webgpu-hybrid' && !crossOriginIsolated) {
      // Cross-origin isolation is not required by WebGPU, but the page shares the
      // demo server with threaded WASM profiles; fail loudly instead of silently
      // falling back to a slower path.
      status('Note: not cross-origin isolated; WebGPU encoder path requested');
    }
    const adapter = await navigator.gpu?.requestAdapter();
    result.adapter = adapter
      ? { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture }
      : null;
    if (field('backend').value === 'webgpu-hybrid' && !result.adapter) {
      throw new Error('WebGPU adapter required for the hybrid profile');
    }

    status('Preparing local entries…');
    entries = createSpeechModelLocalEntries(files);

    status('Loading model…');
    const started = performance.now();
    model = await loadSpeechModelFromLocalEntries({
      modelId: MODEL_ID,
      entries,
      backend: field('backend').value,
      encoderQuant: field('precision').value,
      preprocessorBackend: 'js',
      // Serve the ORT runtime from the repository's installed package through
      // Vite's /@fs/ route, which serves files untransformed. Without this the
      // loader resolves its wasm sidecar from a CDN and fails offline.
      wasmPaths: import.meta.env.ASRJS_ORT_WASM_BASE,
      signal: null,
    });
    result.loadMs = performance.now() - started;
    result.selection = model.selection;

    const pcm = await decodeFixture(audioFile);
    result.audioSamples = pcm.length;
    // Canonical bytes describe the exact waveform passed to the public API,
    // independently of the input WAV container or the platform's endianness.
    const pcmBytes = new ArrayBuffer(pcm.length * 4);
    const pcmView = new DataView(pcmBytes);
    for (let index = 0; index < pcm.length; index += 1) pcmView.setFloat32(index * 4, pcm[index], true);
    const pcmDigest = new Uint8Array(await crypto.subtle.digest('SHA-256', pcmBytes));
    result.effectivePcm = {
      sampleRate: 16000, channels: 1, samples: pcm.length, validSamples: pcm.length,
      rightPaddingSamples: 0, encoding: 'f32le',
      preparation: (query.get('audioStrategy') ?? 'wav') === 'wav'
        ? 'public decodeWavToMonoPcm; no resampling' : 'AudioContext decode at 16000 Hz',
      sha256: [...pcmDigest].map((byte) => byte.toString(16).padStart(2, '0')).join(''),
    };

    for (let index = 0; index < repeat; index += 1) {
      if (index === 1 && verifyCancellation) {
        status('Cancelling partial decode…');
        const controller = new AbortController();
        const cancellation = { requested: false, rejected: false, completeEmitted: false };
        result.cancellation = cancellation;
        try {
          await model.transcribeMonoPcm(pcm, 16000, {
            signal: controller.signal,
            onProgress(event) {
              if (event.stage === 'complete') cancellation.completeEmitted = true;
              if (event.stage === 'decode' && event.completedUnits > 0 &&
                  event.completedUnits < event.totalUnits && !cancellation.requested) {
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
        if (!cancellation.requested || !cancellation.rejected || cancellation.completeEmitted) {
          throw new Error('Requested partial-decode cancellation did not reject without completion');
        }
      }
      status(`Transcribing ${index + 1}/${repeat}…`);
      const envelope = await model.transcribeMonoPcm(pcm, 16000, {
        returnTokenIds: true,
        responseFlavor: 'canonical+native',
      });
      const transcription = envelope.canonical;
      result.transcripts.push({
        text: transcription.text,
        tokens: envelope.native.debug?.tokenIds?.length ?? 0,
        tokenIds: envelope.native.debug?.tokenIds ?? [],
        metrics: transcription.meta.metrics,
        warnings: transcription.warnings,
      });
    }

    const [first, ...rest] = result.transcripts;
    if (rest.some((item) => item.text !== first.text)) {
      throw new Error('Repeated transcripts differ');
    }
    if (!first.text.trim()) throw new Error('Empty transcript');

    status('Disposing…');
    await model.dispose();
    model = null;
    result.disposed = true;
    result.status = 'PASS_ORUKEET_SMOKE';
    status('Done');
  } catch (error) {
    result.error = String(error);
    status(`Failed: ${error}`);
  } finally {
    if (model) {
      try {
        await model.dispose();
        result.disposed = true;
      } catch (error) {
        result.cleanupError = String(error);
        result.status = 'FAIL_ORUKEET_SMOKE';
      }
    }
    field('result').textContent = JSON.stringify(result, null, 2);
    field('run').disabled = false;
  }
});
