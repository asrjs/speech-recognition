import { createSpeechController } from '@asrjs/speech-recognition/controller';
import { createBrowserRealtimeStarter, startMicrophoneCapture } from '@asrjs/speech-recognition/browser';

const controller = createSpeechController();
const tracks = [];
let starter;
let current;
const receipt = { load: 'UNVERIFIED', file: 'UNVERIFIED', captures: [], progress: [], errors: [] };
const status = document.querySelector('#status');
const snapshot = () => {
  const state = controller.getSnapshot();
  return { status: state.status, modelId: state.modelId, text: state.microphone?.transcriptText ?? state.result?.text ?? '', mic: state.microphone, error: state.error ? String(state.error) : null };
};
window.micAcceptance = {
  receipt, snapshot, mode: 'manual',
  async load(origin, backend = 'webgpu-hybrid') {
    await controller.load({
      modelId: 'parakeet-tdt-0.6b-v3', backend,
      onProgress: event => { if (event.phase !== 'asset:download') receipt.progress.push(event); },
      options: { source: { kind: 'direct',
        artifacts: {
          encoderUrl: origin + '/model/encoder-model.fp16.onnx',
          decoderUrl: origin + '/model/decoder_joint-model.int8.onnx',
          tokenizerUrl: origin + '/model/vocab.txt',
          encoderFilename: 'encoder-model.fp16.onnx', decoderFilename: 'decoder_joint-model.int8.onnx',
        },
        encoderBackend: 'webgpu', decoderBackend: 'wasm', preprocessorBackend: 'js',
        wasmPaths: '/ort/', cpuThreads: 4,
      } },
    });
    receipt.load = 'PASS';
    const baseline = await controller.transcribeSource(origin + '/fixtures/jfk2.en.wav');
    receipt.baseline = {text: baseline.text, metrics: baseline.meta?.metrics};
    receipt.file = 'EXECUTED';
    status.textContent = baseline.text;
  },
  async start() {
    const mode = this.mode;
    current = { mode, started: false, status: 'UNVERIFIED', utterances: [], updates: [] };
    receipt.captures.push(current);
    await controller.startMicrophone({
      micMode: mode === 'manual' ? 'manual' : 'speech-detect',
      config: { gateMode: mode === 'manual' ? 'rough-only' : 'vad-only', vadBackend: 'ten-vad', tenVadEnabled: mode !== 'manual' },
      createStarter(options) { starter = createBrowserRealtimeStarter(options); return starter; },
      async startCapture(options) {
        const handle = await startMicrophoneCapture(options);
        tracks.push(...handle.stream.getTracks());
        return handle;
      },
      onUtterance(utterance) {
        current.utterances.push({frames: utterance.pcm.length, sampleRate: utterance.sampleRate, reason: utterance.reason, context: utterance.context});
      },
      onTranscript(update) {
        current.updates.push({text: update.canonical.text, context: update.context, kind: update.kind, metrics: update.canonical.meta?.metrics});
      },
    });
    current.started = true;
    current.startVadState = starter.getSnapshot().tenVad;
    status.textContent = 'Recording';
  },
  async stop() {
    await controller.stopMicrophone();
    current.final = snapshot();
    current.trackStates = tracks.map(track => track.readyState);
    current.status = 'EXECUTED';
    status.textContent = current.final.text;
  },
  async verifyReuse(origin) {
    const result = await controller.transcribeSource(origin + '/fixtures/jfk2.en.wav');
    receipt.afterCapture = {text: result.text, status: controller.getSnapshot().status};
  },
  async dispose() {
    await controller.dispose();
    receipt.disposal = {status: controller.getSnapshot().status, tracks: tracks.map(track => track.readyState)};
    return receipt;
  },
};
document.querySelector('#start').onclick = () => window.micAcceptance.start().catch(error => { receipt.errors.push(String(error)); status.textContent = String(error); });
document.querySelector('#stop').onclick = () => window.micAcceptance.stop().catch(error => { receipt.errors.push(String(error)); status.textContent = String(error); });
window.micAcceptanceReady = true;
