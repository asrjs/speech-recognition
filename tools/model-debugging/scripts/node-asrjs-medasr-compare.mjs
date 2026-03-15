import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..', '..');
const toolsRoot = path.resolve(projectRoot, 'tools');
const resultsRoot = path.join(toolsRoot, 'data', 'results', 'medasr-node-parity');

function parseArgs(argv) {
  const defaults = {
    samples: 10,
    datasetDir:
      'N:/github/ysdede/scribe-ds/data/PARROT_v1.0/07_audio/labels/_30s',
    medasrjsRoot: 'N:/github/ysdede/medasr.js/medasrjs',
    outJson: path.join(resultsRoot, 'latest.json'),
    backend: 'wasm',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--samples' && argv[index + 1]) {
      defaults.samples = Number.parseInt(argv[index + 1], 10);
      index += 1;
      continue;
    }
    if (arg === '--dataset-dir' && argv[index + 1]) {
      defaults.datasetDir = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--medasrjs-root' && argv[index + 1]) {
      defaults.medasrjsRoot = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--out-json' && argv[index + 1]) {
      defaults.outJson = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--backend' && argv[index + 1]) {
      defaults.backend = argv[index + 1];
      index += 1;
    }
  }

  return defaults;
}

function normalize(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
  for (let i = 0; i <= m; i += 1) dp[i][0] = i;
  for (let j = 0; j <= n; j += 1) dp[0][j] = j;
  for (let i = 1; i <= m; i += 1) {
    for (let j = 1; j <= n; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function wordErrorRate(reference, hypothesis) {
  const a = normalize(reference).split(' ').filter(Boolean);
  const b = normalize(hypothesis).split(' ').filter(Boolean);
  return a.length ? levenshtein(a, b) / a.length : 0;
}

function charErrorRate(reference, hypothesis) {
  const a = normalize(reference);
  const b = normalize(hypothesis);
  return a.length ? levenshtein(a, b) / a.length : 0;
}

function listSamples(datasetDir, maxSamples) {
  const files = fs.readdirSync(datasetDir).filter((entry) => entry.endsWith('.json')).sort();
  const rows = [];
  for (const file of files) {
    const wavPath = path.join(datasetDir, file.replace(/\.json$/i, '.wav'));
    const jsonPath = path.join(datasetDir, file);
    if (!fs.existsSync(wavPath)) {
      continue;
    }

    const label = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    rows.push({
      file: path.basename(wavPath),
      wavPath,
      reference: label.transcription || '',
    });
    if (rows.length >= maxSamples) {
      break;
    }
  }
  return rows;
}

function average(values) {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

const options = parseArgs(process.argv.slice(2));
const medasrRequire = createRequire(path.join(options.medasrjsRoot, 'package.json'));
const ortModule = await import(pathToFileURL(medasrRequire.resolve('onnxruntime-node')).href);
const ort = ortModule.default ?? ortModule;
const wavefileModule = await import(pathToFileURL(medasrRequire.resolve('wavefile')).href);
const wavefilePkg = wavefileModule.default ?? wavefileModule;
const { MedAsrModel } = await import(pathToFileURL(path.join(options.medasrjsRoot, 'src', 'medasr.js')).href);
const { loadSpeechModel, PcmAudioBuffer } = await import(
  pathToFileURL(path.join(projectRoot, 'dist', 'index.js')).href
);
const { WaveFile } = wavefilePkg;

function loadAudioMono16k(audioPath) {
  const wavBuffer = fs.readFileSync(audioPath);
  const wav = new WaveFile(wavBuffer);
  if (wav.fmt.sampleRate !== 16000) {
    wav.toSampleRate(16000);
  }
  wav.toBitDepth('32f');
  const samples = wav.getSamples(false, Float32Array);

  if (Array.isArray(samples)) {
    if (samples.length === 1) {
      return samples[0];
    }

    const mono = new Float32Array(samples[0].length);
    for (let frameIndex = 0; frameIndex < mono.length; frameIndex += 1) {
      let sum = 0;
      for (let channelIndex = 0; channelIndex < samples.length; channelIndex += 1) {
        sum += samples[channelIndex][frameIndex] ?? 0;
      }
      mono[frameIndex] = sum / samples.length;
    }
    return mono;
  }

  return samples;
}

const modelDir = path.join(options.medasrjsRoot, 'models', 'medasr');
const onnxPath = path.join(modelDir, 'model.onnx');
const onnxDataPath = path.join(modelDir, 'model.onnx.data');
const tokensPath = path.join(modelDir, 'tokens.txt');
const tokenizerDataUrl = `data:text/plain;base64,${Buffer.from(
  fs.readFileSync(tokensPath, 'utf-8'),
  'utf-8',
).toString('base64')}`;

const comparisonSamples = listSamples(options.datasetDir, options.samples);
if (comparisonSamples.length === 0) {
  throw new Error(`No wav/json samples found in ${options.datasetDir}`);
}

const medasrSession = await ort.InferenceSession.create(onnxPath, {
  executionProviders: ['cpu'],
});
const medasrjsModel = await MedAsrModel.fromSession({
  session: medasrSession,
  ort,
  tokenizerPath: tokensPath,
  nMels: 128,
});

const asrjsModel = await loadSpeechModel({
  family: 'lasr-ctc',
  modelId: 'google/medasr',
  backend: options.backend,
  classification: {
    ecosystem: 'lasr',
    processor: 'kaldi-mel',
    encoder: 'conformer',
    decoder: 'ctc',
    topology: 'ctc',
    task: 'asr',
    family: 'medasr',
  },
  options: {
    source: {
      kind: 'direct',
      artifacts: {
        modelUrl: pathToFileURL(onnxPath).href,
        modelDataUrl: pathToFileURL(onnxDataPath).href,
        modelDataFilename: 'model.onnx.data',
        tokenizerUrl: tokenizerDataUrl,
      },
    },
  },
});

const rows = [];
for (const sample of comparisonSamples) {
  const audio = loadAudioMono16k(sample.wavPath);

  const medasrjsOutput = await medasrjsModel.transcribe(audio, 16000);
  const asrjsOutput = await asrjsModel.transcribe(PcmAudioBuffer.fromMono(audio, 16000), {
    responseFlavor: 'canonical+native',
    detail: 'detailed',
    returnTokenIds: true,
  });

  rows.push({
    file: sample.file,
    reference: sample.reference,
    medasrjs: {
      text: medasrjsOutput.utterance_text,
      heuristicWer: wordErrorRate(sample.reference, medasrjsOutput.utterance_text),
      heuristicCer: charErrorRate(sample.reference, medasrjsOutput.utterance_text),
      wer: wordErrorRate(sample.reference, medasrjsOutput.utterance_text),
      cer: charErrorRate(sample.reference, medasrjsOutput.utterance_text),
      timings: medasrjsOutput.timings,
    },
    asrjs: {
      text: asrjsOutput.canonical.text,
      nativeText: asrjsOutput.native?.utteranceText,
      heuristicWer: wordErrorRate(sample.reference, asrjsOutput.canonical.text),
      heuristicCer: charErrorRate(sample.reference, asrjsOutput.canonical.text),
      wer: wordErrorRate(sample.reference, asrjsOutput.canonical.text),
      cer: charErrorRate(sample.reference, asrjsOutput.canonical.text),
      metrics: asrjsOutput.canonical.meta.metrics,
      warnings: asrjsOutput.native?.warnings,
    },
    diff: {
      textCer: charErrorRate(medasrjsOutput.utterance_text, asrjsOutput.canonical.text),
      textWer: wordErrorRate(medasrjsOutput.utterance_text, asrjsOutput.canonical.text),
      exactMatch:
        normalize(medasrjsOutput.utterance_text) === normalize(asrjsOutput.canonical.text),
    },
  });
}

await asrjsModel.dispose();

const summary = {
  comparedAt: new Date().toISOString(),
  samples: rows.length,
  datasetDir: options.datasetDir,
  medasrjsRoot: options.medasrjsRoot,
  backend: options.backend,
  scoring: {
    method: 'heuristic-lowercase-strip-punctuation',
    comparableToLeaderboardMetrics: false,
    note:
      'Use transcript-to-transcript comparison or the Python benchmark scripts when you need leaderboard-comparable WER/CER.',
  },
  medasrjsAvgHeuristicWer: average(rows.map((row) => row.medasrjs.heuristicWer)),
  medasrjsAvgHeuristicCer: average(rows.map((row) => row.medasrjs.heuristicCer)),
  asrjsAvgHeuristicWer: average(rows.map((row) => row.asrjs.heuristicWer)),
  asrjsAvgHeuristicCer: average(rows.map((row) => row.asrjs.heuristicCer)),
  medasrjsAvgWer: average(rows.map((row) => row.medasrjs.wer)),
  medasrjsAvgCer: average(rows.map((row) => row.medasrjs.cer)),
  asrjsAvgWer: average(rows.map((row) => row.asrjs.wer)),
  asrjsAvgCer: average(rows.map((row) => row.asrjs.cer)),
  crossCer: average(rows.map((row) => row.diff.textCer)),
  exactMatches: rows.filter((row) => row.diff.exactMatch).length,
};

fs.mkdirSync(path.dirname(options.outJson), { recursive: true });
fs.writeFileSync(
  options.outJson,
  JSON.stringify(
    {
      summary,
      rows,
    },
    null,
    2,
  ),
  'utf-8',
);

console.log('--- asrjs vs medasrjs node comparison ---');
console.log(`Samples: ${summary.samples}`);
console.log(`medasrjs heuristic WER: ${(summary.medasrjsAvgHeuristicWer * 100).toFixed(2)}%`);
console.log(`asrjs heuristic WER: ${(summary.asrjsAvgHeuristicWer * 100).toFixed(2)}%`);
console.log(`Cross-text CER: ${(summary.crossCer * 100).toFixed(2)}%`);
console.log(`Exact transcript matches: ${summary.exactMatches}/${summary.samples}`);
console.log(`Results: ${options.outJson}`);
