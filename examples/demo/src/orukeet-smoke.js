/** Orukeet demo: local-folder load, repeated transcription and awaited disposal. */
import {
  createSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
} from '../../../src/runtime/local-browser.ts';

const MODEL_ID = 'oruk/orukeet';
const field = (id) => document.getElementById(id);
const status = (message) => {
  field('status').textContent = message;
};

async function decodeFixture(file) {
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
      wasmPaths: '/@fs/N:/github/asrjs/speech-recognition/node_modules/onnxruntime-web/dist/',
      signal: null,
    });
    result.loadMs = performance.now() - started;
    result.selection = model.selection;

    const pcm = await decodeFixture(audioFile);
    result.audioSamples = pcm.length;

    for (let index = 0; index < repeat; index += 1) {
      status(`Transcribing ${index + 1}/${repeat}…`);
      const transcription = await model.transcribeMonoPcm(pcm, 16000, { returnTokenIds: true });
      result.transcripts.push({
        text: transcription.text,
        tokens: transcription.tokens?.length ?? 0,
        metrics: transcription.metrics,
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
    if (model) await model.dispose().catch(() => undefined);
    field('result').textContent = JSON.stringify(result, null, 2);
    field('run').disabled = false;
  }
});
