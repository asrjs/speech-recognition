import { resolveWhisperPresetManifest } from '../../../src/presets/whisper/manifest.ts';
import { createSpeechPipeline } from '../../../src/runtime/load.ts';
import { decodeAudioSourceToMonoPcm } from '../../../src/runtime/media.ts';
import './styles.css';

const HF_REPO = 'ysdede/whisper-large-v3-turbo-onnx-4graph';
const HF_REVISION = 'main';
const SAMPLE_AUDIO_URL = '/audio/jfk.ogg';
const IDB_DB_NAME = 'asrjs-cache-db';
const FP32_BASELINE_TEXT =
  'And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.';
const WHISPER_LARGE_TURBO_MODEL_ID = 'onnx-community/whisper-large-v3-turbo';
const WHISPER_LARGE_TURBO_PRESET = resolveWhisperPresetManifest(WHISPER_LARGE_TURBO_MODEL_ID);

if (!WHISPER_LARGE_TURBO_PRESET) {
  throw new Error(`Missing built-in preset for ${WHISPER_LARGE_TURBO_MODEL_ID}.`);
}

const FOLDERS = ['fp16_iofp32', 'fp16', 'fp32', 'q8'];
const BACKENDS = ['webgpu', 'wasm'];

const GRAPH_FILES = {
  encoder: 'encoder_model.onnx',
  decoder_init: 'decoder_init.onnx',
  decoder_step: 'decoder_step.onnx',
  decoder_align: 'decoder_align.onnx',
};

const FOLDER_EXTERNAL = {
  fp16_iofp32: { encoder: true, decoder_init: true, decoder_step: true, decoder_align: true },
  fp16:       { encoder: true, decoder_init: true, decoder_step: true, decoder_align: true },
  fp32:       { encoder: true, decoder_init: true, decoder_step: true, decoder_align: true },
  q8:         { encoder: false, decoder_init: false, decoder_step: false, decoder_align: false },
};

const FOLDER_LABEL = {
  fp16_iofp32: 'fp16io (fp16 internal, fp32 I/O)',
  fp16:        'fp16 (pure, float16 I/O)',
  fp32:        'fp32 (full precision)',
  q8:          'q8 (int8 quantized)',
};

const PRESETS = {
  'fp16io-fp32-webgpu': {
    label: 'fp16io enc + fp32 dec (WebGPU)',
    detail: 'Primary target — fastest on WebGPU, fp16io encoder + fp32 decoder',
    encoderFolder: 'fp16_iofp32', decoderFolder: 'fp32',
    encoderBackend: 'webgpu', decoderBackend: 'webgpu',
  },
  'q8-fp32-webgpu': {
    label: 'q8 enc + fp32 dec (WebGPU)',
    detail: 'Backup — q8 encoder (no fetch limit) + fp32 decoder',
    encoderFolder: 'q8', decoderFolder: 'fp32',
    encoderBackend: 'webgpu', decoderBackend: 'webgpu',
  },
  'fp32-webgpu': {
    label: 'fp32 full (WebGPU probe)',
    detail: 'Baseline — 2.4GB encoder exceeds browser fetch limit',
    encoderFolder: 'fp32', decoderFolder: 'fp32',
    encoderBackend: 'webgpu', decoderBackend: 'webgpu',
  },
  'fp16-fp32-webgpu': {
    label: 'fp16 enc + fp32 dec (WebGPU probe)',
    detail: 'Pure fp16 encoder (float16 input) — distribution mismatch',
    encoderFolder: 'fp16', decoderFolder: 'fp32',
    encoderBackend: 'webgpu', decoderBackend: 'webgpu',
  },
  'fp16-webgpu': {
    label: 'fp16 full (WebGPU NaN probe)',
    detail: 'Known broken — fp16 decoder NaN, regression only',
    encoderFolder: 'fp16', decoderFolder: 'fp16',
    encoderBackend: 'webgpu', decoderBackend: 'webgpu',
  },
  'q8-wasm': {
    label: 'q8 full WASM (BROKEN)',
    detail: 'ConvInteger unsupported on browser WASM',
    encoderFolder: 'q8', decoderFolder: 'q8',
    encoderBackend: 'wasm', decoderBackend: 'wasm',
  },
};

const DEFAULT_PRESET = 'fp16io-fp32-webgpu';
const p = PRESETS[DEFAULT_PRESET];

const state = {
  preset: DEFAULT_PRESET,
  repoId: HF_REPO,
  revision: HF_REVISION,
  encoderFolder: p.encoderFolder,
  decoderFolder: p.decoderFolder,
  encoderBackend: p.encoderBackend,
  decoderBackend: p.decoderBackend,
  language: 'en',
  maxNewTokens: 200,
  noTimestamps: true,
  audio: null,
  progress: new Map(),
  loadedModelKey: '',
  running: false,
  status: 'idle',
  transcript: '',
  tokens: [],
  metrics: { loadMs: 0, transcribeMs: 0, totalMs: 0, cacheEntries: 0 },
  log: ['Ready.'],
};

let activeProgressCallback = null;
let pipeline = createPipeline();

function createPipeline() {
  return createSpeechPipeline({
    cacheModels: true,
    hooks: {
      onProgress(event) {
        activeProgressCallback?.(event);
      },
    },
  });
}

// ── helpers ──

function hfResolveUrl(repoId, revision, filename) {
  return `https://huggingface.co/${repoId.split('/').map(encodeURIComponent).join('/')}/resolve/${encodeURIComponent(revision || 'main')}/${filename.split('/').map(encodeURIComponent).join('/')}`;
}

function graphPath(folder, graphName) {
  return `${folder}/${GRAPH_FILES[graphName]}`;
}

function externalFor(folder) {
  return { ...FOLDER_EXTERNAL[folder] };
}

function buildSplitGraphSource() {
  const { repoId, revision, encoderFolder, decoderFolder } = state;
  const encExt = externalFor(encoderFolder);
  const decExt = externalFor(decoderFolder);

  const encoderUrl = hfResolveUrl(repoId, revision, graphPath(encoderFolder, 'encoder'));
  const decoderInitUrl = hfResolveUrl(repoId, revision, graphPath(decoderFolder, 'decoder_init'));
  const decoderStepUrl = hfResolveUrl(repoId, revision, graphPath(decoderFolder, 'decoder_step'));
  const decoderAlignUrl = hfResolveUrl(repoId, revision, graphPath(decoderFolder, 'decoder_align'));
  const tokenizerUrl = hfResolveUrl(repoId, revision, `${decoderFolder}/tokenizer.json`);
  const manifestUrl = hfResolveUrl(repoId, revision, `${decoderFolder}/manifest.json`);

  const externalDataUrls = {};
  for (const g of ['encoder', 'decoder_init', 'decoder_step', 'decoder_align']) {
    const needsExt = g === 'encoder' ? encExt.encoder : decExt[g];
    if (needsExt) {
      externalDataUrls[g] = [{ path: `./${GRAPH_FILES[g]}.data`, file: `${GRAPH_FILES[g]}.data` }];
    }
  }

  return {
    kind: 'splitgraph',
    artifacts: { encoderUrl, decoderInitUrl, decoderStepUrl, decoderAlignUrl, tokenizerUrl, manifestUrl, externalDataUrls },
    encoderBackend: state.encoderBackend,
    decoderBackend: state.decoderBackend,
  };
}

function buildModelRequest() {
  const source = buildSplitGraphSource();
  const cacheKey = [
    'whisper4', state.repoId, state.revision || 'main',
    state.encoderFolder, state.decoderFolder,
    state.encoderBackend, state.decoderBackend,
  ].join(':');

  return {
    modelId: WHISPER_LARGE_TURBO_PRESET.modelId,
    backend: state.encoderBackend === 'webgpu' || state.decoderBackend === 'webgpu' ? 'webgpu' : 'wasm',
    cacheKey,
    classification: WHISPER_LARGE_TURBO_PRESET.classification,
    options: { config: WHISPER_LARGE_TURBO_PRESET.config, source },
  };
}

function buildTranscribeOptions() {
  return {
    language: state.language,
    task: 'transcribe',
    noTimestamps: state.noTimestamps,
    maxNewTokens: Number(state.maxNewTokens) || 200,
    detail: 'segments',
    returnSpecialTokens: true,
    responseFlavor: 'canonical+native',
  };
}

// ── progress & logging ──

function handleRuntimeProgress(event) {
  if (event.phase === 'asset:download' && event.file) {
    state.progress.set(event.file, { ...(state.progress.get(event.file) || {}), ...event, updatedAt: performance.now() });
    renderProgress();
    renderMetrics();
    return;
  }
  if (event.message) log(event.message);
}

function log(message) {
  const now = new Date().toLocaleTimeString();
  state.log.push(`[${now}] ${message}`);
  if (state.log.length > 240) state.log.shift();
  renderLog();
}

function setRunning(running) {
  state.running = running;
  document.querySelectorAll('[data-run-control]').forEach(n => { n.disabled = running; });
}

function setStatus(status) { state.status = status; renderStatus(); }

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '-';
  const u = ['B', 'KB', 'MB', 'GB'];
  let v = bytes, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i >= 2 ? 2 : 0)} ${u[i]}`;
}

function formatMs(ms) {
  if (!Number.isFinite(ms) || ms <= 0) return '-';
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

async function estimateStorage() {
  if (!navigator.storage?.estimate) return;
  const e = await navigator.storage.estimate();
  state.metrics.cacheEntries = e.usage || 0;
  renderMetrics();
}

// ── audio ──

async function prepareAudio(source, labelPrefix = '') {
  const decoded = await decodeAudioSourceToMonoPcm(source, { targetSampleRate: 16000, strategy: 'browser-target-rate' });
  return {
    pcm: decoded.pcm, sampleRate: decoded.sampleRate, durationSeconds: decoded.durationSec,
    label: labelPrefix ? `${labelPrefix} (${decoded.durationSec.toFixed(1)}s ${decoded.sampleRate}Hz)` : `${decoded.durationSec.toFixed(1)}s ${decoded.sampleRate}Hz`,
    metrics: decoded.metrics,
  };
}

async function loadSampleAudio() {
  state.audio = await prepareAudio(SAMPLE_AUDIO_URL);
  log(`Loaded sample audio (${state.audio.label}).`);
  renderAudio();
}

async function handleFileInput(file) {
  if (!file) return;
  state.audio = await prepareAudio(file, file.name);
  log(`Loaded ${file.name}.`);
  renderAudio();
}

// ── actions ──

async function loadModelOnly() {
  state.progress.clear();
  setRunning(true); setStatus('loading');
  activeProgressCallback = handleRuntimeProgress;
  const request = buildModelRequest();
  const t0 = performance.now();
  try {
    log(`Loading ${state.encoderFolder} enc + ${state.decoderFolder} dec on ${state.encoderBackend}/${state.decoderBackend}.`);
    const loaded = await pipeline.loadModel(request);
    state.loadedModelKey = request.cacheKey;
    state.metrics.loadMs = performance.now() - t0;
    log(`Model ready on ${loaded.model.backend.id}.`);
    setStatus('ready');
  } catch (error) {
    setStatus('error');
    log(error?.stack || error?.message || String(error));
  } finally {
    activeProgressCallback = null;
    setRunning(false);
    renderMetrics();
  }
}

async function transcribe() {
  if (!state.audio) await loadSampleAudio();
  state.progress.clear(); state.tokens = []; state.transcript = '';
  renderResults();
  setRunning(true); setStatus('running');
  activeProgressCallback = handleRuntimeProgress;
  const request = buildModelRequest();
  const t0 = performance.now();
  try {
    const loaded = await pipeline.loadModel(request);
    const afterLoad = performance.now();
    state.loadedModelKey = request.cacheKey;
    log('Transcribing audio.');
    const result = await loaded.transcribeMonoPcm(state.audio.pcm, state.audio.sampleRate, buildTranscribeOptions());
    const t1 = performance.now();
    state.metrics.loadMs = afterLoad - t0;
    state.metrics.transcribeMs = t1 - afterLoad;
    state.metrics.totalMs = t1 - t0;
    state.transcript = result.canonical?.text || result.native?.utteranceText || '';
    state.tokens = result.native?.tokens || [];
    log(`Transcript: ${state.transcript || '(empty)'}`);
    setStatus(state.transcript ? 'complete' : 'empty');
    await estimateStorage();
  } catch (error) {
    setStatus('error');
    log(error?.stack || error?.message || String(error));
  } finally {
    activeProgressCallback = null;
    setRunning(false);
    renderMetrics(); renderResults();
  }
}

async function clearCache() {
  setRunning(true);
  try {
    await pipeline.dispose();
    pipeline = createPipeline();
    state.loadedModelKey = '';
    if (typeof indexedDB !== 'undefined') {
      await new Promise((resolve, reject) => {
        const r = indexedDB.deleteDatabase(IDB_DB_NAME);
        r.onsuccess = () => resolve();
        r.onerror = () => reject(new Error('IndexedDB cache delete failed.'));
        r.onblocked = () => resolve();
      });
    }
    state.progress.clear(); state.metrics.cacheEntries = 0;
    log('Cleared model cache.');
    setStatus('idle');
  } finally {
    setRunning(false);
    renderProgress(); renderMetrics();
  }
}

function applyPreset(key) {
  const preset = PRESETS[key];
  if (!preset) return;
  state.preset = key;
  state.encoderFolder = preset.encoderFolder;
  state.decoderFolder = preset.decoderFolder;
  state.encoderBackend = preset.encoderBackend;
  state.decoderBackend = preset.decoderBackend;
  render();
}

// ── compatibility ──

function getCompatibilityNotes() {
  const notes = [];
  const enc = state.encoderFolder;
  const dec = state.decoderFolder;
  const wantsWebGpu = state.encoderBackend === 'webgpu' || state.decoderBackend === 'webgpu';

  if (!window.isSecureContext) notes.push({ kind: 'bad', text: 'Not a secure context. Use HTTPS or localhost for WebGPU.' });
  if (wantsWebGpu && !navigator.gpu) notes.push({ kind: 'bad', text: 'WebGPU unavailable in this browser.' });
  if (!window.crossOriginIsolated) notes.push({ kind: 'warn', text: 'COI off — threaded WASM may fail.' });
  if (state.encoderBackend === 'wasm' && (enc === 'fp16' || enc === 'fp16_iofp32')) notes.push({ kind: 'bad', text: 'WASM does not support fp16/fp16io encoder.' });
  if (state.decoderBackend === 'wasm' && dec === 'fp16') notes.push({ kind: 'bad', text: 'WASM does not support fp16 decoder.' });
  if (state.encoderBackend === 'wasm' && enc === 'q8') notes.push({ kind: 'bad', text: 'q8 encoder uses ConvInteger — unsupported on browser WASM.' });
  if (state.encoderBackend === 'webgpu' && enc === 'fp16_iofp32') notes.push({ kind: 'good', text: 'fp16io encoder: fp32 I/O + fp16 internals. Optimal for WebGPU.' });
  if (state.encoderBackend === 'webgpu' && enc === 'fp16') notes.push({ kind: 'warn', text: 'Pure fp16 encoder expects float16 input. Use fp16_iofp32 instead.' });
  if (state.decoderBackend === 'webgpu' && dec === 'q8') notes.push({ kind: 'warn', text: 'q8 decoder on WebGPU: known overflow probe.' });
  if (state.decoderBackend === 'webgpu' && dec === 'fp16') notes.push({ kind: 'warn', text: 'fp16 decoder on WebGPU: known NaN probe.' });
  if (enc === 'fp32' && externalFor(enc).encoder) notes.push({ kind: 'warn', text: 'fp32 encoder (2.4GB) exceeds browser fetch limit.' });

  if (!notes.length) notes.push({ kind: 'good', text: 'Selected combination looks valid for the current browser.' });
  return notes;
}

// ── render ──

function render() {
  document.querySelector('#app').innerHTML = `
<div class="app">
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand"><h1>asr.js Whisper 4-Graph</h1><p>Browser demo — library pipeline + IndexedDB cache</p></div>
      <div class="status-strip" id="status-strip"></div>
    </div>
  </header>
  <section class="layout">
    <aside class="stack">
      <section class="panel">
        <div class="panel-header"><h2>Presets</h2></div>
        <div class="panel-body preset-list" id="preset-list"></div>
      </section>

      <section class="panel">
        <div class="panel-header"><h2>Model</h2></div>
        <div class="panel-body grid">
          <div class="grid two">
            <label>Encoder folder
              <select data-field="encoderFolder">${FOLDERS.map(f => `<option value="${f}" ${state.encoderFolder === f ? 'selected' : ''}>${FOLDER_LABEL[f]}</option>`).join('')}</select>
            </label>
            <label>Decoder folder
              <select data-field="decoderFolder">${FOLDERS.map(f => `<option value="${f}" ${state.decoderFolder === f ? 'selected' : ''}>${FOLDER_LABEL[f]}</option>`).join('')}</select>
            </label>
          </div>
          <div class="grid two">
            <label>Encoder backend
              <select data-field="encoderBackend">${BACKENDS.map(b => `<option value="${b}" ${state.encoderBackend === b ? 'selected' : ''}>${b.toUpperCase()}</option>`).join('')}</select>
            </label>
            <label>Decoder backend
              <select data-field="decoderBackend">${BACKENDS.map(b => `<option value="${b}" ${state.decoderBackend === b ? 'selected' : ''}>${b.toUpperCase()}</option>`).join('')}</select>
            </label>
          </div>
          <div id="compatibility"></div>
          <div class="actions">
            <button class="primary" data-action="load" data-run-control>Load Model</button>
            <button data-action="clear-cache" data-run-control>Clear Cache</button>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header"><h2>Audio</h2></div>
        <div class="panel-body grid">
          <div class="file-row">
            <input type="file" accept="audio/*" data-action="file" />
            <button data-action="sample" data-run-control>JFK Sample</button>
          </div>
          <div class="grid two">
            <label>Language <input data-field="language" value="${state.language}" /></label>
            <label>Max tokens <input type="number" min="1" max="448" data-field="maxNewTokens" value="${state.maxNewTokens}" /></label>
          </div>
          <label><span>No timestamps</span> <input type="checkbox" data-field="noTimestamps" ${state.noTimestamps ? 'checked' : ''} /></label>
          <div id="audio-status" class="note"></div>
          <button class="primary" data-action="transcribe" data-run-control>Transcribe</button>
        </div>
      </section>
    </aside>

    <section class="stack">
      <section class="summary-grid" id="metrics"></section>
      <section class="panel">
        <div class="panel-header"><h2>Transcript</h2><span class="pill">Baseline JFK text available</span></div>
        <div class="panel-body"><div class="transcript" id="transcript"></div></div>
      </section>
      <section class="panel">
        <div class="panel-header"><h2>Assets</h2></div>
        <div class="panel-body">
          <div class="table-wrap">
            <table><thead><tr><th>File</th><th>Loaded</th><th>Total</th><th>Progress</th></tr></thead><tbody id="progress-body"></tbody></table>
          </div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-header"><h2>Tokens</h2></div>
        <div class="panel-body"><div class="tokens" id="tokens"></div></div>
      </section>
      <section class="panel">
        <div class="panel-header"><h2>Log</h2></div>
        <div class="panel-body"><pre class="log" id="log"></pre></div>
      </section>
    </section>
  </section>
</div>`;

  bindEvents();
  renderPresetList();
  renderStatus();
  renderCompatibility();
  renderAudio();
  renderMetrics();
  renderProgress();
  renderResults();
  renderLog();
}

function bindEvents() {
  document.querySelectorAll('[data-field]').forEach(node => {
    const field = node.dataset.field;
    if (node.type === 'checkbox') {
      node.checked = Boolean(state[field]);
      node.addEventListener('change', () => { state[field] = node.checked; renderCompatibility(); });
      return;
    }
    if (node.tagName === 'SELECT') node.value = state[field];
    node.addEventListener('change', () => {
      state[field] = node.type === 'number' ? Number(node.value) : node.value;
      state.preset = ''; // custom selection — no preset active
      renderPresetList();
      renderCompatibility();
    });
  });

  document.querySelector('[data-action="load"]').addEventListener('click', () => { loadModelOnly().catch(() => {}); });
  document.querySelector('[data-action="transcribe"]').addEventListener('click', () => { transcribe().catch(() => {}); });
  document.querySelector('[data-action="clear-cache"]').addEventListener('click', () => { clearCache().catch(e => log(e?.message || String(e))); });
  document.querySelector('[data-action="sample"]').addEventListener('click', () => { loadSampleAudio().catch(e => log(e?.message || String(e))); });
  document.querySelector('[data-action="file"]').addEventListener('change', e => { handleFileInput(e.target.files?.[0]).catch(e => log(e?.message || String(e))); });
}

function renderPresetList() {
  const list = document.querySelector('#preset-list');
  list.innerHTML = Object.entries(PRESETS).map(([key, preset]) =>
    `<button class="preset ${state.preset === key ? 'active' : ''}" data-preset="${key}"><strong>${preset.label}</strong><span>${preset.detail}</span></button>`
  ).join('');
  list.querySelectorAll('[data-preset]').forEach(node => {
    node.addEventListener('click', () => applyPreset(node.dataset.preset));
  });
}

function renderStatus() {
  const strip = document.querySelector('#status-strip');
  const webgpu = 'gpu' in navigator;
  const notes = getCompatibilityNotes();
  const kind = state.status === 'error' ? 'bad' : state.status === 'loading' || state.status === 'running' ? 'warn' : state.status === 'complete' || state.status === 'ready' ? 'good' : '';
  strip.innerHTML = `
    <span class="pill ${kind}">${state.status}</span>
    <span class="pill ${webgpu ? 'good' : 'bad'}">WebGPU ${webgpu ? 'available' : 'missing'}</span>
    <span class="pill ${crossOriginIsolated ? 'good' : 'warn'}">COI ${crossOriginIsolated ? 'on' : 'off'}</span>
    <span class="pill ${notes[0]?.kind || ''}">${notes[0]?.text || ''}</span>`;
}

function renderCompatibility() {
  const target = document.querySelector('#compatibility');
  const notes = getCompatibilityNotes();
  target.innerHTML = `<div class="grid">${notes.map(n => `<span class="pill ${n.kind}">${n.text}</span>`).join('')}</div>`;
  renderStatus();
}

function renderAudio() {
  document.querySelector('#audio-status').textContent = state.audio ? state.audio.label : 'No audio loaded.';
}

function renderMetrics() {
  const match = state.transcript && state.transcript.trim() === FP32_BASELINE_TEXT ? 'match' : state.transcript ? 'check' : '-';
  document.querySelector('#metrics').innerHTML = `
    <div class="metric"><span>Load</span><strong>${formatMs(state.metrics.loadMs)}</strong></div>
    <div class="metric"><span>Transcribe</span><strong>${formatMs(state.metrics.transcribeMs)}</strong></div>
    <div class="metric"><span>Total</span><strong>${formatMs(state.metrics.totalMs)}</strong></div>
    <div class="metric"><span>Storage</span><strong>${formatBytes(state.metrics.cacheEntries)}</strong></div>
    <div class="metric"><span>Baseline</span><strong>${match}</strong></div>`;
}

function renderProgress() {
  const body = document.querySelector('#progress-body');
  const rows = [...state.progress.values()].sort((a, b) => String(a.file).localeCompare(String(b.file)));
  if (!rows.length) { body.innerHTML = '<tr><td colspan="4" class="note">No asset activity yet.</td></tr>'; return; }
  body.innerHTML = rows.map(event => {
    const pct = Number.isFinite(event.percent) ? event.percent : event.total ? Math.round((event.loaded / event.total) * 100) : 0;
    return `<tr><td class="mono">${event.file}</td><td>${formatBytes(event.loaded)}</td><td>${formatBytes(event.total)}</td><td><div class="bar"><span style="width:${Math.min(100, pct)}%"></span></div></td></tr>`;
  }).join('');
}

function renderResults() {
  document.querySelector('#transcript').textContent = state.transcript || 'No transcript yet.';
  const tokenTarget = document.querySelector('#tokens');
  const shown = state.tokens.slice(0, 96);
  tokenTarget.innerHTML = shown.length
    ? shown.map(t => `<span class="token">${t.id ?? ''}:${escapeHtml(t.text || '')}</span>`).join('')
    : '<span class="note">No token output yet.</span>';
}

function renderLog() {
  const node = document.querySelector('#log');
  node.textContent = state.log.join('\n');
  node.scrollTop = node.scrollHeight;
}

function escapeHtml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

render();
estimateStorage().catch(() => undefined);
