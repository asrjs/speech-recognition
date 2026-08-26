import fs from 'fs';
import os from 'os';
import path from 'path';
import process from 'process';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

const PYTHON = 'C:\\Users\\steam\\anaconda3\\envs\\nemo\\python.exe';
const DEFAULT_AUDIO = path.resolve(process.cwd(), 'tools/data/fixtures/audio/librivox.org.wav');
const DEFAULT_NODE_MODEL = 'N:\\models\\onnx\\nemo\\parakeet-tdt-0.6b-v2-onnx-tfjs4';
const DEFAULT_ONNX_ASR_MODEL = 'N:\\models\\onnx\\nemo\\parakeet-tdt-0.6b-v2-onnx';
const DEFAULT_NEMO_MODEL = 'nvidia/parakeet-tdt-0.6b-v2';

export function validateCommandArg(val, label = 'argument') {
  if (typeof val !== 'string') {
    throw new Error(`Invalid ${label}: expected string, got ${typeof val}`);
  }
  if (val.includes('\0')) {
    throw new Error(`Null bytes forbidden in ${label}`);
  }
  return val;
}

export function validatePath(val, label = 'path') {
  validateCommandArg(val, label);
  if (!val.trim()) {
    throw new Error(`Empty ${label}`);
  }
  return path.isAbsolute(val) ? path.normalize(val) : path.resolve(process.cwd(), val);
}

export function validateSampleRate(val) {
  const num = typeof val === 'number' ? val : Number(val);
  if (!Number.isFinite(num) || num <= 0 || !Number.isInteger(num)) {
    throw new Error(`Invalid sample rate: expected positive integer, got ${val}`);
  }
  return num;
}

export function parseArgs(rawArgs = process.argv.slice(2)) {
  const args = rawArgs;
  const out = {
    audio: DEFAULT_AUDIO,
    sampleRate: 16000,
    nodeModel: DEFAULT_NODE_MODEL,
    onnxAsrModelPath: DEFAULT_ONNX_ASR_MODEL,
    nemoModel: DEFAULT_NEMO_MODEL,
    keepArtifacts: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--audio') {
      const val = args[++i];
      out.audio = validatePath(val, 'audio');
    } else if (arg === '--sample-rate') {
      const val = args[++i];
      out.sampleRate = validateSampleRate(val);
    } else if (arg === '--node-model') {
      const val = args[++i];
      out.nodeModel = validatePath(val, 'nodeModel');
    } else if (arg === '--onnx-asr-model-path') {
      const val = args[++i];
      out.onnxAsrModelPath = validatePath(val, 'onnxAsrModelPath');
    } else if (arg === '--nemo-model') {
      const val = args[++i];
      out.nemoModel = validateCommandArg(val, 'nemoModel');
    } else if (arg === '--keep-artifacts') {
      out.keepArtifacts = true;
    }
  }

  return out;
}

export function runOrThrow(command, args, label) {
  validateCommandArg(command, 'command');
  if (!Array.isArray(args)) {
    throw new Error('args must be an array');
  }
  for (let i = 0; i < args.length; i++) {
    validateCommandArg(args[i], `argument [${i}]`);
  }

  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    encoding: 'utf8',
    stdio: 'pipe',
  });
  if (result.status !== 0) {
    throw new Error(
      `${label} failed with exit code ${result.status}\n` +
      `stdout:\n${result.stdout || '(empty)'}\n` +
      `stderr:\n${result.stderr || '(empty)'}`,
    );
  }
  return result;
}

function normalizeText(text) {
  return String(text ?? '')
    .normalize('NFKC')
    .replace(/\s+/g, ' ')
    .trim();
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function takeFirst(items, count = 30) {
  return Array.isArray(items) ? items.slice(0, count) : [];
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function summariseNode(payload) {
  return {
    text: payload.output?.text ?? '',
    token_ids: (payload.output?.tokens ?? []).map((token) => token.id),
    token_pieces: (payload.output?.tokens ?? []).map((token) => token.rawToken),
  };
}

function summarisePython(payload) {
  return {
    text: payload.result?.text ?? '',
    token_ids: payload.result?.token_ids ?? [],
    token_pieces: payload.result?.token_pieces ?? payload.result?.tokens ?? [],
  };
}

function compareRuns(reference, candidate) {
  const refFirst30 = takeFirst(reference.token_ids);
  const candFirst30 = takeFirst(candidate.token_ids);

  return {
    normalized_text_matches: normalizeText(reference.text) === normalizeText(candidate.text),
    first_30_token_ids_match: arraysEqual(refFirst30, candFirst30),
    reference_first_30_token_ids: refFirst30,
    candidate_first_30_token_ids: candFirst30,
    reference_first_30_pieces: takeFirst(reference.token_pieces),
    candidate_first_30_pieces: takeFirst(candidate.token_pieces),
  };
}

function main() {
  const opts = parseArgs();
  if (!fs.existsSync(opts.audio)) {
    throw new Error(`Audio file not found: ${opts.audio}`);
  }

  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nemo-shared-audio-'));
  const sharedAudio = path.join(tmpRoot, `shared-${opts.sampleRate}.wav`);
  const nodeJson = path.join(tmpRoot, 'node.json');
  const nemoJson = path.join(tmpRoot, 'nemo.json');
  const onnxAsrJson = path.join(tmpRoot, 'onnx-asr.json');

  runOrThrow(PYTHON, [
    path.resolve(process.cwd(), 'tools/model-debugging/reference/hf-parakeet-port/python-resample-wav.py'),
    '--input', opts.audio,
    '--output', sharedAudio,
    '--sample-rate', String(opts.sampleRate),
    '--subtype', 'PCM_16',
  ], 'python-resample-wav');

  runOrThrow('node', [
    path.resolve(process.cwd(), 'tools/model-debugging/reference/hf-parakeet-port/node-nemo-inspect.mjs'),
    '--audio', sharedAudio,
    '--model', opts.nodeModel,
    '--api', 'direct',
    '--timestamps', 'segments',
    '--return-words',
    '--return-tokens',
    '--output', nodeJson,
  ], 'node-nemo-inspect');

  runOrThrow(PYTHON, [
    path.resolve(process.cwd(), 'tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py'),
    '--audio', sharedAudio,
    '--model', opts.nemoModel,
    '--device', 'cpu',
    '--output', nemoJson,
  ], 'python-nemo-inspect');

  runOrThrow(PYTHON, [
    path.resolve(process.cwd(), 'tools/model-debugging/reference/hf-parakeet-port/python-onnx-asr-inspect.py'),
    '--audio', sharedAudio,
    '--model-path', opts.onnxAsrModelPath,
    '--output', onnxAsrJson,
  ], 'python-onnx-asr-inspect');

  const node = summariseNode(readJson(nodeJson));
  const nemo = summarisePython(readJson(nemoJson));
  const onnxAsr = summarisePython(readJson(onnxAsrJson));

  const payload = {
    shared_audio: sharedAudio,
    source_audio: opts.audio,
    sample_rate: opts.sampleRate,
    comparisons: {
      node_vs_nemo: compareRuns(node, nemo),
      node_vs_onnx_asr: compareRuns(node, onnxAsr),
      nemo_vs_onnx_asr: compareRuns(nemo, onnxAsr),
    },
    texts: {
      node: node.text,
      nemo: nemo.text,
      onnx_asr: onnxAsr.text,
    },
  };

  console.log(JSON.stringify(payload, null, 2));

  if (!opts.keepArtifacts) {
    fs.rmSync(tmpRoot, { recursive: true, force: true });
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  try {
    main();
  } catch (error) {
    console.error('[node-nemo-shared-audio-parity] failed:', error);
    process.exitCode = 1;
  }
}
