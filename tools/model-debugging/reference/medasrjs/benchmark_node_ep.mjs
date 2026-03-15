import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import ort from 'onnxruntime-node';
import wavefilePkg from 'wavefile';

import { MedAsrPipeline } from '../src/index.js';

const { WaveFile } = wavefilePkg;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const opts = {
    samples: 10,
    ep: 'cpu',
    datasetDir: process.env.PARROT_DATASET_DIR || 'N:/github/ysdede/scribe-ds/data/PARROT_v1.0/07_audio/labels/_30s',
    onnxPath: path.join(PROJECT_ROOT, 'models', 'medasr', 'model.onnx'),
    tokensPath: path.join(PROJECT_ROOT, 'models', 'medasr', 'tokens.txt'),
    outJson: path.join(PROJECT_ROOT, 'metrics', 'benchmark_node_results.json'),
    summaryJson: path.join(PROJECT_ROOT, 'metrics', 'benchmark_node_summary.json'),
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--samples' && argv[i + 1]) opts.samples = parseInt(argv[++i], 10);
    else if (a === '--ep' && argv[i + 1]) opts.ep = argv[++i];
    else if (a === '--dataset-dir' && argv[i + 1]) opts.datasetDir = argv[++i];
    else if (a === '--onnx-path' && argv[i + 1]) opts.onnxPath = argv[++i];
    else if (a === '--tokens-path' && argv[i + 1]) opts.tokensPath = argv[++i];
    else if (a === '--out-json' && argv[i + 1]) opts.outJson = argv[++i];
    else if (a === '--summary-json' && argv[i + 1]) opts.summaryJson = argv[++i];
  }
  return opts;
}

function loadAudioMono16k(wavPath) {
  const wavBuffer = fs.readFileSync(wavPath);
  const wav = new WaveFile(wavBuffer);
  if (wav.fmt.sampleRate !== 16000) wav.toSampleRate(16000);
  wav.toBitDepth('32f');
  let samples = wav.getSamples(false, Float32Array);

  if (Array.isArray(samples)) {
    if (samples.length === 1) return samples[0];
    const mono = new Float32Array(samples[0].length);
    for (let i = 0; i < mono.length; i++) {
      let acc = 0;
      for (let c = 0; c < samples.length; c++) acc += samples[c][i];
      mono[i] = acc / samples.length;
    }
    return mono;
  }

  return samples;
}

function normalize(s) {
  return (s || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function wordErrorRate(ref, hyp) {
  const a = normalize(ref).split(' ').filter(Boolean);
  const b = normalize(hyp).split(' ').filter(Boolean);
  const dist = levenshtein(a, b);
  return a.length ? dist / a.length : 0;
}

function charErrorRate(ref, hyp) {
  const a = normalize(ref);
  const b = normalize(hyp);
  const dist = levenshtein(a, b);
  return a.length ? dist / a.length : 0;
}

function listSamples(datasetDir, maxSamples) {
  const files = fs.readdirSync(datasetDir).filter((f) => f.endsWith('.json')).sort();
  const out = [];
  for (const f of files) {
    const wav = path.join(datasetDir, f.replace(/\.json$/, '.wav'));
    const jsonPath = path.join(datasetDir, f);
    if (!fs.existsSync(wav)) continue;
    const label = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    out.push({ file: path.basename(wav), wav, reference: label.transcription || '' });
    if (out.length >= maxSamples) break;
  }
  return out;
}

function epOptions(ep) {
  if (ep === 'cpu') return ['cpu'];
  if (ep === 'cuda') return ['cuda'];
  throw new Error(`Unsupported --ep: ${ep}. Use cpu or cuda.`);
}

(async () => {
  const opts = parseArgs(process.argv.slice(2));
  fs.mkdirSync(path.dirname(opts.outJson), { recursive: true });
  fs.mkdirSync(path.dirname(opts.summaryJson), { recursive: true });

  const samples = listSamples(opts.datasetDir, opts.samples);
  if (samples.length === 0) throw new Error(`No wav/json samples found in ${opts.datasetDir}`);

  let session;
  try {
    session = await ort.InferenceSession.create(opts.onnxPath, {
      executionProviders: epOptions(opts.ep),
    });
  } catch (e) {
    throw new Error(
      `Failed to create ONNX session with EP="${opts.ep}". ` +
      `Run 'npm run medasrjs:compat' to inspect runtime support. Original error: ${e?.message || String(e)}`
    );
  }

  const pipeline = MedAsrPipeline.fromNodeSession({
    session,
    ort,
    tokensPath: opts.tokensPath,
    nMels: 128,
  });

  const rows = [];
  let totalAudio = 0;
  let totalInfer = 0;
  let werSum = 0;
  let cerSum = 0;

  for (const s of samples) {
    const audio = loadAudioMono16k(s.wav);
    const duration = audio.length / 16000;
    totalAudio += duration;

    const t0 = performance.now();
    const out = await pipeline.transcribe(audio);
    const inferSec = (performance.now() - t0) / 1000;
    totalInfer += inferSec;

    const prediction = out.utterance_text;
    const wer = wordErrorRate(s.reference, prediction);
    const cer = charErrorRate(s.reference, prediction);
    werSum += wer;
    cerSum += cer;

    rows.push({
      file: s.file,
      reference: s.reference,
      prediction,
      audio_duration: duration,
      inference_time: inferSec,
      wer,
      cer,
    });
  }

  const avgWer = werSum / rows.length;
  const avgCer = cerSum / rows.length;
  const rtf = totalInfer / totalAudio;

  const summary = {
    backend: `onnx-node-${opts.ep}`,
    samples: rows.length,
    dataset_dir: opts.datasetDir,
    onnx_path: opts.onnxPath,
    tokens_path: opts.tokensPath,
    audio_duration_sec: totalAudio,
    inference_time_sec: totalInfer,
    rtf,
    approx_wer: avgWer,
    approx_cer: avgCer,
  };

  fs.writeFileSync(opts.outJson, JSON.stringify(rows, null, 2), 'utf-8');
  fs.writeFileSync(opts.summaryJson, JSON.stringify(summary, null, 2), 'utf-8');

  console.log('--- medasrjs benchmark ---');
  console.log(`EP: ${opts.ep}`);
  console.log(`Samples: ${rows.length}`);
  console.log(`RTF: ${rtf.toFixed(4)}x`);
  console.log(`Approx WER: ${(avgWer * 100).toFixed(2)}%`);
  console.log(`Approx CER: ${(avgCer * 100).toFixed(2)}%`);
  console.log(`Results: ${opts.outJson}`);
  console.log(`Summary: ${opts.summaryJson}`);
})();
