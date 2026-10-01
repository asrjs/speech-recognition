import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { compareTokenIds } from './node-compare-stage-captures.mjs';

import {
  PcmAudioBuffer,
  createSpeechRuntime,
  createWasmBackend,
  createNativeCpuBackend,
  loadSpeechModel,
} from '@asrjs/speech-recognition';
import { createDefaultNodeAssetProvider } from '@asrjs/speech-recognition/io/node';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import {
  createParakeetPresetFactory,
  resolveParakeetPresetManifest,
} from '@asrjs/speech-recognition/presets/parakeet';

const DEFAULT_AUDIO = path.resolve(process.cwd(), 'tools/data/fixtures/audio/librivox.org.wav');
const QUANT_SUFFIX = {
  fp32: '.onnx',
  fp16: '.fp16.onnx',
  int8: '.int8.onnx',
};

function readUInt32LE(buffer, offset) {
  return buffer.readUInt32LE(offset);
}

function readUInt16LE(buffer, offset) {
  return buffer.readUInt16LE(offset);
}

export function parseArgs(args = process.argv.slice(2)) {
  const options = {
    modelId: 'parakeet-tdt-0.6b-v2',
    modelDir: process.env.ASRJS_NEMO_MODEL_DIR
      ? path.resolve(process.env.ASRJS_NEMO_MODEL_DIR)
      : null,
    family: null,
    config: null,
    backend: 'wasm',
    gridBatching: true,
    audio: DEFAULT_AUDIO,
    targetSampleRate: 16000,
    encoderQuant: 'fp32',
    decoderQuant: 'fp32',
    preprocessor: 'nemo128',
    preprocessorBackend: 'onnx',
    cpuThreads: 4,
    output: null,
    oracle: null,
    sampleId: null,
    originalWeights: null,
    inputMode: 'source-wav',
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--model-id') options.modelId = args[++index];
    else if (arg === '--family') options.family = args[++index];
    else if (arg === '--config') options.config = path.resolve(args[++index]);
    else if (arg === '--backend') options.backend = args[++index];
    else if (arg === '--no-grid-batching') options.gridBatching = false;
    else if (arg === '--model-dir') options.modelDir = path.resolve(args[++index]);
    else if (arg === '--audio') options.audio = path.resolve(args[++index]);
    else if (arg === '--target-sample-rate') options.targetSampleRate = Number(args[++index]);
    else if (arg === '--encoder-quant') options.encoderQuant = args[++index];
    else if (arg === '--decoder-quant') options.decoderQuant = args[++index];
    else if (arg === '--preprocessor') options.preprocessor = args[++index];
    else if (arg === '--preprocessor-backend') options.preprocessorBackend = args[++index];
    else if (arg === '--cpu-threads') options.cpuThreads = Number(args[++index]);
    else if (arg === '--output') options.output = path.resolve(args[++index]);
    else if (arg === '--oracle') options.oracle = path.resolve(args[++index]);
    else if (arg === '--sample-id') options.sampleId = args[++index];
    else if (arg === '--original-weights') options.originalWeights = path.resolve(args[++index]);
    else if (arg === '--input-mode') options.inputMode = args[++index];
    else throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

export function validateInspectorOptions(options) {
  if (!options.modelDir)
    throw new Error(
      'Provide --model-dir or ASRJS_NEMO_MODEL_DIR; no machine-specific model directory is assumed.',
    );
  if (!['wasm', 'native-cpu'].includes(options.backend))
    throw new Error('This Node inspector supports --backend wasm or native-cpu.');
  if (options.family && options.family !== 'nemo-tdt')
    throw new Error('This inspector currently supports only --family nemo-tdt.');
  if (options.family && !options.config)
    throw new Error(
      'Explicit family inference requires --config with an audited runtime configuration.',
    );
  if (options.oracle && (!options.sampleId || !options.originalWeights))
    throw new Error(
      '--oracle requires --sample-id and --original-weights to bind the official reference.',
    );
  if (!['source-wav', 'oracle-pcm'].includes(options.inputMode))
    throw new Error('--input-mode must be source-wav or oracle-pcm.');
  if (options.inputMode === 'oracle-pcm' && !options.oracle)
    throw new Error('--input-mode oracle-pcm requires --oracle.');
  if (!['js', 'onnx'].includes(options.preprocessorBackend))
    throw new Error('Unsupported preprocessor backend');
  if (!QUANT_SUFFIX[options.encoderQuant] || !QUANT_SUFFIX[options.decoderQuant])
    throw new Error('Unsupported graph precision');
  if (!Number.isInteger(options.cpuThreads) || options.cpuThreads < 1)
    throw new Error('--cpu-threads must be a positive integer');
  if (!Number.isInteger(options.targetSampleRate) || options.targetSampleRate < 1)
    throw new Error('--target-sample-rate must be a positive integer');
  if (
    !options.family &&
    resolveParakeetPresetManifest(options.modelId)?.classification.topology !== 'tdt'
  ) {
    throw new Error(
      'Unknown models require explicit --family nemo-tdt and --config; Parakeet defaults are not generic.',
    );
  }
}

function loadWavAsFloat32(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`Unsupported WAV container: ${filePath}`);
  }

  let offset = 12;
  let audioFormat = null;
  let numChannels = null;
  let sampleRate = null;
  let bitsPerSample = null;
  let dataOffset = null;
  let dataSize = null;

  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = readUInt32LE(buffer, offset + 4);
    const chunkDataStart = offset + 8;
    const next = chunkDataStart + chunkSize + (chunkSize % 2);

    if (chunkId === 'fmt ') {
      audioFormat = readUInt16LE(buffer, chunkDataStart);
      numChannels = readUInt16LE(buffer, chunkDataStart + 2);
      sampleRate = readUInt32LE(buffer, chunkDataStart + 4);
      bitsPerSample = readUInt16LE(buffer, chunkDataStart + 14);
    } else if (chunkId === 'data') {
      dataOffset = chunkDataStart;
      dataSize = chunkSize;
    }

    offset = next;
  }

  if (audioFormat == null || numChannels == null || sampleRate == null || bitsPerSample == null) {
    throw new Error(`Invalid WAV: missing fmt chunk in ${filePath}`);
  }
  if (dataOffset == null || dataSize == null) {
    throw new Error(`Invalid WAV: missing data chunk in ${filePath}`);
  }

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(dataSize / bytesPerSample);
  const totalFrames = Math.floor(totalSamples / numChannels);
  const mono = new Float32Array(totalFrames);
  let cursor = dataOffset;

  for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
    let sum = 0;
    for (let channelIndex = 0; channelIndex < numChannels; channelIndex += 1) {
      let sample = 0;
      if (audioFormat === 1 && bitsPerSample === 16) {
        sample = buffer.readInt16LE(cursor) / 32768;
      } else if (audioFormat === 1 && bitsPerSample === 32) {
        sample = buffer.readInt32LE(cursor) / 2147483648;
      } else if (audioFormat === 3 && bitsPerSample === 32) {
        sample = buffer.readFloatLE(cursor);
      } else {
        throw new Error(`Unsupported WAV format. format=${audioFormat}, bits=${bitsPerSample}`);
      }
      sum += sample;
      cursor += bytesPerSample;
    }
    mono[frameIndex] = sum / numChannels;
  }

  return { audio: mono, sampleRate, numberOfChannels: numChannels };
}

function resampleLinear(audio, fromRate, toRate) {
  if (fromRate === toRate) {
    return audio;
  }
  if (!Number.isFinite(fromRate) || !Number.isFinite(toRate) || fromRate <= 0 || toRate <= 0) {
    throw new Error(`Invalid resample rates: from=${fromRate}, to=${toRate}`);
  }

  const ratio = toRate / fromRate;
  const outputLength = Math.max(1, Math.round(audio.length * ratio));
  const output = new Float32Array(outputLength);
  const scale = fromRate / toRate;

  for (let index = 0; index < outputLength; index += 1) {
    const position = index * scale;
    const left = Math.floor(position);
    const right = Math.min(left + 1, audio.length - 1);
    const fraction = position - left;
    output[index] = audio[left] * (1 - fraction) + audio[right] * fraction;
  }

  return output;
}

function ensureFile(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
  return filePath;
}

function toFileUrl(filePath) {
  return pathToFileURL(filePath).href;
}

export function resolveModelArtifacts(options) {
  const encoderName = `encoder-model${QUANT_SUFFIX[options.encoderQuant]}`;
  const decoderName = `decoder_joint-model${QUANT_SUFFIX[options.decoderQuant]}`;
  const tokenizerName = 'vocab.txt';
  const preprocessorName = `${options.preprocessor}.onnx`;
  const encoderPath = ensureFile(path.join(options.modelDir, encoderName), 'Encoder model');
  const decoderPath = ensureFile(path.join(options.modelDir, decoderName), 'Decoder model');
  const tokenizerPath = ensureFile(path.join(options.modelDir, tokenizerName), 'Tokenizer');
  const preprocessorPath =
    options.preprocessorBackend === 'js'
      ? null
      : ensureFile(path.join(options.modelDir, preprocessorName), 'Preprocessor model');
  const encoderDataPath = path.join(options.modelDir, `${encoderName}.data`);
  const decoderDataPath = path.join(options.modelDir, `${decoderName}.data`);

  return {
    encoderPath,
    decoderPath,
    tokenizerPath,
    preprocessorPath,
    encoderDataPath: fs.existsSync(encoderDataPath) ? encoderDataPath : null,
    decoderDataPath: fs.existsSync(decoderDataPath) ? decoderDataPath : null,
  };
}

function findContext(text, needle, radius = 32) {
  const lowerText = String(text ?? '').toLowerCase();
  const lowerNeedle = String(needle).toLowerCase();
  const index = lowerText.indexOf(lowerNeedle);
  if (index === -1) {
    return null;
  }
  const start = Math.max(0, index - radius);
  const end = Math.min(text.length, index + needle.length + radius);
  return text.slice(start, end);
}

function normalizeText(text) {
  return String(text ?? '')
    .normalize('NFKC')
    .replace(/\s+/g, ' ')
    .trim();
}

export async function describeLocalArtifact(filePath) {
  const absolute = path.resolve(filePath);
  const initial = await fs.promises.stat(absolute);
  if (!initial.isFile() || initial.size === 0)
    throw new Error(`Expected a non-empty regular file: ${absolute}`);
  const handle = await fs.promises.open(absolute, 'r');
  try {
    const before = await handle.stat();
    if (!before.isFile() || before.size === 0)
      throw new Error(`Expected a non-empty regular file: ${absolute}`);
    const hash = createHash('sha256');
    let bytes = 0;
    for await (const chunk of handle.createReadStream({ autoClose: false })) {
      bytes += chunk.length;
      hash.update(chunk);
    }
    const after = await handle.stat();
    if (bytes !== before.size || after.size !== before.size || after.mtimeMs !== before.mtimeMs) {
      throw new Error(`Artifact changed while hashing: ${absolute}`);
    }
    return { path: absolute, bytes, sha256: hash.digest('hex') };
  } finally {
    await handle.close();
  }
}

export async function runInspectorWithCleanup(operation, disposeModel, disposeRuntime) {
  let payload;
  const failures = [];
  try {
    payload = await operation();
  } catch (error) {
    failures.push(error);
  }
  const cleanupStart = performance.now();
  for (const dispose of [disposeModel, disposeRuntime]) {
    try {
      await dispose();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) throw new AggregateError(failures, 'Inspector inference/cleanup failed');
  return {
    ...payload,
    cleanup: { status: 'PASS', durationMs: +(performance.now() - cleanupStart).toFixed(2) },
  };
}

export function compareNemoTdtOutput(reference, payload, sampleId) {
  const candidate = {
    tokens: payload.native?.debug?.tokenIds,
    token_frames: payload.native?.debug?.frameIndices,
    token_duration: payload.native?.debug?.tdtSteps,
    transcript: payload.output?.text,
  };
  const fields = ['tokens', 'token_frames', 'token_duration'];
  for (const [label, value] of [
    ['Reference', reference],
    ['Candidate', candidate],
  ]) {
    for (const field of fields) {
      if (!Array.isArray(value?.[field])) throw new Error(`${label} requires explicit ${field}`);
    }
    if (typeof value?.transcript !== 'string')
      throw new Error(`${label} requires explicit transcript`);
  }
  const checks = Object.fromEntries(
    fields.map((field) => [field, compareTokenIds(reference, candidate, sampleId, field)]),
  );
  checks.transcript = {
    pass: reference.transcript === candidate.transcript,
    reference: reference.transcript,
    candidate: candidate.transcript,
  };
  const firstDivergentField =
    [...fields, 'transcript'].find((field) => !checks[field].pass) ?? null;
  return {
    scope: 'FULL_TDT_OUTPUT_ONLY',
    sampleId,
    pass: firstDivergentField === null,
    firstDivergentField,
    checks,
  };
}

export function validateOracleAudioPreparation(sourceAudio, targetSampleRate) {
  if (sourceAudio.sampleRate !== targetSampleRate || sourceAudio.numberOfChannels !== 1) {
    throw new Error(
      'Official oracle comparison requires same-rate mono audio; linear resampling or channel mixing is not the official input contract. Prepare a shared PCM fixture and recapture its reference.',
    );
  }
}

export function describeModelInputPcm(audio, sampleRate) {
  if (
    !(audio instanceof Float32Array) ||
    audio.length === 0 ||
    !Number.isInteger(sampleRate) ||
    sampleRate < 1
  ) {
    throw new Error('Invalid model-input PCM/rate; expected non-empty mono float32 samples.');
  }
  const hash = createHash('sha256');
  const blockSamples = 16384;
  const block = Buffer.allocUnsafe(blockSamples * 4);
  for (let offset = 0; offset < audio.length; offset += blockSamples) {
    const count = Math.min(blockSamples, audio.length - offset);
    for (let index = 0; index < count; index += 1) {
      const value = audio[offset + index];
      if (!Number.isFinite(value)) throw new Error('Non-finite model-input PCM sample.');
      block.writeFloatLE(value, index * 4);
    }
    hash.update(block.subarray(0, count * 4));
  }
  return {
    sample_rate: sampleRate,
    channels: 1,
    dtype: 'float32',
    encoding: 'f32le',
    shape: [1, audio.length],
    samples: audio.length,
    valid_samples: audio.length,
    right_padding_samples: 0,
    sha256: hash.digest('hex'),
  };
}

function readCapturedWaveform(oracle, sampleRate) {
  const contract = oracle?.sample?.model_input;
  const waveform = oracle?.sample?.stages?.waveform;
  const length = oracle?.sample?.stages?.waveform_length;
  const fail = () => {
    throw new Error(
      'INPUT_CONTRACT_MISMATCH: missing, corrupt or padded official model-input PCM.',
    );
  };
  if (
    !contract ||
    contract.boundary !== 'preprocessor_input_signal' ||
    waveform?.dtype !== 'float32' ||
    !Array.isArray(waveform.shape) ||
    waveform.shape.length !== 2 ||
    waveform.shape[0] !== 1 ||
    !Number.isSafeInteger(waveform.shape[1]) ||
    waveform.shape[1] < 1 ||
    !Array.isArray(waveform.data) ||
    waveform.data.length !== waveform.shape[1] ||
    waveform.data.some(
      (value) =>
        typeof value !== 'number' || !Number.isFinite(value) || Math.fround(value) !== value,
    ) ||
    !['int32', 'int64'].includes(length?.dtype) ||
    !Array.isArray(length.shape) ||
    length.shape.length !== 1 ||
    length.shape[0] !== 1 ||
    !Array.isArray(length.data) ||
    length.data.length !== 1 ||
    length.data[0] !== waveform.shape[1]
  )
    fail();
  const audio = Float32Array.from(waveform.data);
  const actual = describeModelInputPcm(audio, sampleRate);
  if (
    !Array.isArray(contract.shape) ||
    contract.shape.length !== 2 ||
    contract.shape[0] !== 1 ||
    contract.shape[1] !== actual.samples ||
    Object.entries(actual).some(([key, value]) => key !== 'shape' && contract[key] !== value)
  )
    fail();
  return audio;
}

export function prepareInspectorAudio(sourceAudio, sampleRate, oracle, inputMode = 'source-wav') {
  if (!['source-wav', 'oracle-pcm'].includes(inputMode)) throw new Error('Invalid --input-mode.');
  if (inputMode === 'oracle-pcm' && !oracle) throw new Error('oracle-pcm requires --oracle.');
  if (oracle && inputMode === 'source-wav') validateOracleAudioPreparation(sourceAudio, sampleRate);
  const audio =
    inputMode === 'oracle-pcm'
      ? readCapturedWaveform(oracle, sampleRate)
      : resampleLinear(sourceAudio.audio, sourceAudio.sampleRate, sampleRate);
  const modelInput = describeModelInputPcm(audio, sampleRate);
  let waveformComparison = null;
  if (oracle) {
    if (oracle.sample?.model_input) {
      const reference = describeModelInputPcm(readCapturedWaveform(oracle, sampleRate), sampleRate);
      if (modelInput.sha256 !== reference.sha256 || modelInput.samples !== reference.samples) {
        throw new Error(
          'INPUT_CONTRACT_MISMATCH: candidate effective PCM differs from official input.',
        );
      }
      waveformComparison = {
        scope: 'EFFECTIVE_MODEL_INPUT_PCM',
        status: 'PASS',
        pass: true,
        reference,
        candidate: modelInput,
      };
    } else {
      waveformComparison = {
        scope: 'EFFECTIVE_MODEL_INPUT_PCM',
        status: 'UNVERIFIED',
        pass: null,
        reason: 'Legacy oracle does not capture effective model-input PCM.',
      };
    }
  }
  return {
    audio,
    modelInput,
    waveformComparison,
    resampler:
      inputMode === 'oracle-pcm'
        ? 'official-captured-pcm'
        : sourceAudio.sampleRate === sampleRate
          ? 'none'
          : 'linear',
  };
}

export function loadNemoTdtOracle(filePath, expected) {
  const bytes = fs.readFileSync(filePath);
  const capture = JSON.parse(
    (filePath.endsWith('.gz') ? gunzipSync(bytes) : bytes).toString('utf8'),
  );
  if (capture.schema_version !== 1 || !Array.isArray(capture.samples))
    throw new Error('Unsupported official capture schema');
  if (capture.meta?.model !== expected.modelId)
    throw new Error('Official/candidate model identity differs');
  if (!expected.weightSha256 || capture.meta?.weight_sha256 !== expected.weightSha256)
    throw new Error('Official/candidate original weight identity differs');
  const controls = [
    'baseline_tokens_match',
    'baseline_text_match',
    'baseline_frames_match',
    'after_tokens_match',
    'after_text_match',
    'after_durations_match',
  ];
  if (controls.some((key) => capture.instrumentation_control?.[key] !== true))
    throw new Error('Official instrumentation control is missing or failed');
  const matches = capture.samples.filter((sample) => sample.sample_id === expected.sampleId);
  if (matches.length !== 1)
    throw new Error('Exactly one official sample with the requested ID is required');
  const sample = matches[0];
  if (!expected.audioSha256 || sample.audio?.sha256 !== expected.audioSha256)
    throw new Error('Official/candidate audio identity differs');
  return { sample, meta: capture.meta, instrumentationControl: capture.instrumentation_control };
}

async function main() {
  const options = parseArgs();
  validateInspectorOptions(options);
  ensureFile(options.audio, 'Audio file');
  ensureFile(options.modelDir, 'Model directory');
  const artifacts = resolveModelArtifacts(options);
  const hashStart = performance.now();
  const inputs = {
    ...artifacts,
    audio: options.audio,
    config: options.config,
    originalWeights: options.originalWeights,
    oracle: options.oracle,
  };
  const provenance = {};
  for (const [key, file] of Object.entries(inputs)) {
    if (file) provenance[key] = await describeLocalArtifact(file);
  }
  if (options.output) {
    const output = fs.existsSync(options.output) ? fs.realpathSync(options.output) : options.output;
    if (Object.values(provenance).some((input) => fs.realpathSync(input.path) === output)) {
      throw new Error('Output must not overwrite an input artifact or oracle');
    }
  }
  const oracle = options.oracle
    ? loadNemoTdtOracle(options.oracle, {
        sampleId: options.sampleId,
        modelId: options.modelId,
        audioSha256: provenance.audio.sha256,
        weightSha256: provenance.originalWeights.sha256,
      })
    : null;
  const artifactHashMs = performance.now() - hashStart;

  const audioStart = performance.now();
  const sourceAudio = loadWavAsFloat32(options.audio);
  const resampleStart = performance.now();
  const preparedAudio = prepareInspectorAudio(
    sourceAudio,
    options.targetSampleRate,
    oracle,
    options.inputMode,
  );
  const audioForModel = preparedAudio.audio;
  const resampleMs = performance.now() - resampleStart;

  const runtime = createSpeechRuntime({
    assetProvider: createDefaultNodeAssetProvider(),
  });
  runtime.registerBackend(
    options.backend === 'native-cpu' ? createNativeCpuBackend() : createWasmBackend(),
  );
  runtime.registerModelFamily(createNemoTdtModelFamily());
  runtime.registerPreset(
    createParakeetPresetFactory({
      useManifestSource: false,
    }),
  );

  let loaded;
  const payload = await runInspectorWithCleanup(
    async () => {
      const loadStart = performance.now();
      loaded = await loadSpeechModel({
        runtime,
        ...(options.family ? { family: options.family } : { preset: 'parakeet' }),
        modelId: options.modelId,
        backend: options.backend,
        selectionCriteria: { allowExperimental: options.backend === 'native-cpu' },
        options: {
          config: options.config
            ? JSON.parse(fs.readFileSync(ensureFile(options.config, 'Runtime config'), 'utf8'))
            : undefined,
          source: {
            kind: 'direct',
            artifacts: {
              encoderUrl: toFileUrl(artifacts.encoderPath),
              decoderUrl: toFileUrl(artifacts.decoderPath),
              tokenizerUrl: toFileUrl(artifacts.tokenizerPath),
              preprocessorUrl: artifacts.preprocessorPath
                ? toFileUrl(artifacts.preprocessorPath)
                : undefined,
              encoderDataUrl: artifacts.encoderDataPath
                ? toFileUrl(artifacts.encoderDataPath)
                : undefined,
              decoderDataUrl: artifacts.decoderDataPath
                ? toFileUrl(artifacts.decoderDataPath)
                : undefined,
              encoderFilename: path.basename(artifacts.encoderPath),
              decoderFilename: path.basename(artifacts.decoderPath),
            },
            preprocessorBackend: options.preprocessorBackend,
            cpuThreads: options.cpuThreads,
          },
        },
      });
      const loadMs = performance.now() - loadStart;

      const transcribeStart = performance.now();
      const pcmInput = PcmAudioBuffer.fromMono(audioForModel, options.targetSampleRate);
      const publicInput = describeModelInputPcm(pcmInput.channels[0], pcmInput.sampleRate);
      if (publicInput.sha256 !== preparedAudio.modelInput.sha256)
        throw new Error('INPUT_CONTRACT_MISMATCH: public PCM container changed effective samples.');
      const result = await loaded.transcribe(pcmInput, {
        detail: 'detailed',
        responseFlavor: 'canonical+native',
        returnTokenIds: true,
        returnFrameIndices: true,
        returnLogProbs: true,
        returnTdtSteps: true,
        gridBatching: options.gridBatching,
      });
      const transcribeMs = performance.now() - transcribeStart;

      const text = result.canonical.text ?? result.native?.utteranceText ?? '';
      const ort = await import(
        options.backend === 'native-cpu' ? 'onnxruntime-node' : 'onnxruntime-web'
      );
      const payload = {
        schema_version: 1,
        generated_at: new Date().toISOString(),
        status: 'UNCOMPARED_INFERENCE',
        environment: {
          host: process.versions.bun ? 'bun' : 'node',
          versions: { ...process.versions },
          platform: process.platform,
          arch: process.arch,
          ort: { ...ort.env.versions },
          cpuThreads: options.cpuThreads,
        },
        provenance,
        model: {
          modelId: options.modelId,
          modelDir: options.modelDir,
          backend: loaded.model.backend.id,
          family: loaded.model.info.family,
          engine: options.backend === 'native-cpu' ? 'onnxruntime-node' : 'onnxruntime-web',
          provider: options.backend === 'native-cpu' ? 'cpu' : 'wasm',
          configFile: options.config,
          gridBatching: options.gridBatching,
          encoderQuant: options.encoderQuant,
          decoderQuant: options.decoderQuant,
          preprocessor: options.preprocessor,
          preprocessorBackend: options.preprocessorBackend,
        },
        audio: {
          file: options.audio,
          sourceSampleRate: sourceAudio.sampleRate,
          targetSampleRate: options.targetSampleRate,
          sourceSeconds: +(sourceAudio.audio.length / sourceAudio.sampleRate).toFixed(3),
          modelSeconds: +(audioForModel.length / options.targetSampleRate).toFixed(3),
          channels: sourceAudio.numberOfChannels,
          modelInput: publicInput,
        },
        preparation: {
          artifactHashMs: +artifactHashMs.toFixed(2),
          loadAudioMs: +(resampleStart - audioStart).toFixed(2),
          resampleMs: +resampleMs.toFixed(2),
          inputMode: options.inputMode,
          resampler: preparedAudio.resampler,
        },
        timing: {
          loadMs: +loadMs.toFixed(2),
          transcribeMs: +transcribeMs.toFixed(2),
          totalMs: +(performance.now() - audioStart).toFixed(2),
        },
        output: {
          text,
          normalizedText: normalizeText(text),
          contextLibrivox: findContext(text, 'librivox'),
          contextDomain: findContext(text, 'librivox.org'),
          words: result.canonical.words?.length ?? result.native?.words?.length ?? 0,
          tokens: result.canonical.tokens?.length ?? result.native?.tokens?.length ?? 0,
          first30TokenIds: (result.native?.tokens ?? [])
            .map((token) => token.id)
            .filter((id) => Number.isFinite(id))
            .slice(0, 30),
          first30TokenPieces: (result.native?.tokens ?? [])
            .map((token) => token.rawText ?? token.text)
            .slice(0, 30),
        },
        canonical: result.canonical,
        native: result.native,
        waveformComparison: preparedAudio.waveformComparison,
      };
      payload.oracleComparison = oracle
        ? compareNemoTdtOutput(oracle.sample, payload, options.sampleId)
        : null;
      if (oracle) {
        payload.reference = {
          meta: oracle.meta,
          instrumentationControl: oracle.instrumentationControl,
        };
        payload.status = payload.oracleComparison.pass
          ? 'PARTIAL_PASS_OUTPUT_ONLY'
          : 'FAIL_OUTPUT_COMPARISON';
      }
      return payload;
    },
    async () => {
      await loaded?.dispose();
    },
    async () => {
      await runtime.dispose();
    },
  );

  const encoded = JSON.stringify(payload, null, 2);
  if (options.output) {
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, encoded, 'utf8');
  } else {
    console.log(encoded);
  }

  if (payload.oracleComparison && !payload.oracleComparison.pass) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error('[node-asrjs-nemo-inspect] failed:', error);
    process.exitCode = 1;
  });
}
