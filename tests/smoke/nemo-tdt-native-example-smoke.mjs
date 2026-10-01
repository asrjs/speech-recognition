/** Real CLI acceptance for the native consumer example; no inference implementation here. */
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { decodeWavToMonoPcm } from '@asrjs/speech-recognition/io/node';
import {
  prepareNativeExampleInput,
  parseNativeExampleArgs,
} from '../../examples/nemo-tdt-native-local.mjs';
import {
  describeLocalArtifact,
  loadNemoTdtOracle,
  compareNemoTdtOutput,
  prepareInspectorAudio,
} from '../../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

/** This recipe's family-owned layout, including either graph's optional sidecar. */
export function nativeExampleArtifactPaths(modelDir) {
  return [
    'encoder-model.onnx',
    'encoder-model.onnx.data',
    'decoder_joint-model.onnx',
    'decoder_joint-model.onnx.data',
    'vocab.txt',
  ]
    .map((file) => path.join(modelDir, file))
    .filter((file) => fs.existsSync(file));
}

/** Validate prepared input with the existing adapter, without resampling it. */
export function compareNativeExamplePcm(audio, sampleRate, oracle) {
  if (!oracle) throw new Error('Native example smoke requires an official oracle.');
  return prepareInspectorAudio(
    { audio, sampleRate, numberOfChannels: 1 },
    sampleRate,
    oracle,
    'source-wav',
  ).waveformComparison;
}

/** Enumerate the actual compiled build, not assumed source-to-dist equivalents. */
export function nativeExampleBuildFiles(buildDir) {
  const files = [];
  function visit(directory) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const filename = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(filename);
      else if (entry.isFile()) files.push(filename);
      else throw new Error(`Unsupported build entry: ${filename}`);
    }
  }
  visit(buildDir);
  if (!files.length) throw new Error('Native example smoke requires a nonempty compiled build.');
  return files.sort();
}

async function main() {
  const { values } = parseArgs({
    options: Object.fromEntries(
      [
        'model-dir',
        'config',
        'model-id',
        'audio',
        'oracle',
        'sample-id',
        'original-weights',
        'output',
        'executable',
        'prepared-wav',
        'threads',
      ].map((key) => [key, { type: 'string' }]),
    ),
    strict: true,
    allowPositionals: false,
  });
  for (const key of [
    'model-dir',
    'config',
    'model-id',
    'audio',
    'oracle',
    'sample-id',
    'original-weights',
    'output',
  ]) {
    if (!values[key]?.trim()) throw new Error(`Required --${key}.`);
  }
  const output = path.resolve(values.output);
  if (fs.existsSync(output)) throw new Error(`Refusing to overwrite receipt: ${output}`);
  const originalWeights = await describeLocalArtifact(values['original-weights']);
  const sourceAudio = await describeLocalArtifact(values.audio);
  const oracle = loadNemoTdtOracle(values.oracle, {
    modelId: values['model-id'],
    sampleId: values['sample-id'],
    audioSha256: sourceAudio.sha256,
    weightSha256: originalWeights.sha256,
  });
  let audioPath = path.resolve(values.audio);
  let derivation = null;
  if (values['prepared-wav']) {
    const source = decodeWavToMonoPcm(fs.readFileSync(audioPath));
    const rate = JSON.parse(fs.readFileSync(values.config, 'utf8')).sampleRate;
    const captured = prepareInspectorAudio(
      {
        audio: source.pcm,
        sampleRate: source.sampleRate,
        numberOfChannels: source.numberOfChannels,
      },
      rate,
      oracle,
      'oracle-pcm',
    );
    const pcm = captured.audio;
    const bytes = Buffer.alloc(44 + pcm.byteLength);
    bytes.write('RIFF');
    bytes.writeUInt32LE(bytes.length - 8, 4);
    bytes.write('WAVEfmt ', 8);
    bytes.writeUInt32LE(16, 16);
    bytes.writeUInt16LE(3, 20);
    bytes.writeUInt16LE(1, 22);
    bytes.writeUInt32LE(rate, 24);
    bytes.writeUInt32LE(rate * 4, 28);
    bytes.writeUInt16LE(4, 32);
    bytes.writeUInt16LE(32, 34);
    bytes.write('data', 36);
    bytes.writeUInt32LE(pcm.byteLength, 40);
    for (let i = 0; i < pcm.length; i++) bytes.writeFloatLE(pcm[i], 44 + i * 4);
    audioPath = path.resolve(values['prepared-wav']);
    if (fs.existsSync(audioPath)) {
      if (!fs.readFileSync(audioPath).equals(bytes))
        throw new Error('Existing derived WAV differs from official PCM.');
    } else fs.writeFileSync(audioPath, bytes, { flag: 'wx' });
    derivation = {
      method: 'official captured float32 PCM to IEEE float WAV, no resampling',
      sourceAudio,
      capturedModelInput: captured.modelInput,
    };
  }
  const argv = [
    '--model-dir',
    values['model-dir'],
    '--config',
    values.config,
    '--model-id',
    values['model-id'],
    '--audio',
    audioPath,
    '--threads',
    values.threads ?? '4',
  ];
  const prepared = await prepareNativeExampleInput(parseNativeExampleArgs(argv));
  const waveform = compareNativeExamplePcm(
    prepared.audio.channels[0],
    prepared.audio.sampleRate,
    oracle,
  );
  const files = [
    'examples/nemo-tdt-native-local.mjs',
    'src/audio/wav.ts',
    'dist/audio/wav.js',
    'src/runtime/media.ts',
    'src/io-node.ts',
    'dist/io-node.js',
    'src/models/nemo-tdt/executor.ts',
    'src/models/nemo-tdt/model.ts',
    'src/models/nemo-tdt/ort.ts',
    'src/models/nemo-tdt/preprocessor.ts',
    'src/runtime/session.ts',
    'src/runtime/owned-session.ts',
    'src/audio/js-mel.ts',
    'dist/audio/js-mel.js',
    'tests/smoke/nemo-tdt-native-example-smoke.mjs',
    'package.json',
    values.config,
    values.oracle,
    audioPath,
    ...nativeExampleArtifactPaths(values['model-dir']),
  ];
  const provenance = await Promise.all(files.map(describeLocalArtifact));
  const buildDir = path.resolve('dist');
  const buildFiles = [];
  // Bound open streams: snapshots must not exhaust descriptors on large builds.
  for (const filename of nativeExampleBuildFiles(buildDir)) {
    const artifact = await describeLocalArtifact(filename);
    buildFiles.push({
      relativePath: path.relative(buildDir, filename).replaceAll('\\', '/'),
      bytes: artifact.bytes,
      sha256: artifact.sha256,
    });
  }
  const libraryBuild = {
    directory: buildDir,
    files: buildFiles,
    sha256: createHash('sha256').update(JSON.stringify(buildFiles)).digest('hex'),
    note: 'Actual compiled snapshot before launch; hashes alone do not prove source-to-build equivalence.',
  };
  const executable = values.executable ?? process.execPath;
  const command = [executable, 'examples/nemo-tdt-native-local.mjs', ...argv];
  const start = performance.now();
  const child = spawnSync(executable, command.slice(1), {
    cwd: process.cwd(),
    encoding: 'utf8',
    timeout: 240000,
    maxBuffer: 8 * 1024 * 1024,
    windowsHide: true,
  });
  const execution = {
    exitCode: child.status,
    signal: child.signal,
    error: child.error?.message ?? null,
    wallMilliseconds: performance.now() - start,
  };
  let result = null;
  let comparison = null;
  let error = null;
  try {
    if (child.error || child.status !== 0)
      throw new Error('Example CLI did not exit successfully.');
    result = JSON.parse(child.stdout);
    if (
      result.status !== 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE' ||
      result.backend.id !== 'native-cpu' ||
      result.backend.engine !== 'onnxruntime-node' ||
      result.backend.provider !== 'cpu' ||
      !result.backend.versions.node
    )
      throw new Error('Actual native backend identity missing.');
    comparison = compareNemoTdtOutput(
      oracle.sample,
      {
        native: result.transcript.native,
        output: result.transcript.canonical,
      },
      values['sample-id'],
    );
    if (!comparison.pass) throw new Error('Full output differs from official inference.');
  } catch (failure) {
    error = failure.message;
  }
  const receipt = {
    status: error ? 'FAIL_NATIVE_EXAMPLE' : 'PARTIAL_PASS_OUTPUT_ONLY',
    createdAt: new Date().toISOString(),
    command,
    execution,
    originalWeights,
    sourceAudio,
    provenance,
    libraryBuild,
    derivation,
    waveformComparison: waveform,
    oracleComparison: comparison,
    cleanup: {
      status: result && child.status === 0 ? 'PASS' : 'UNVERIFIED',
      basis:
        'Example JSON is returned only after awaited runtime-owned cleanup; child must exit normally.',
    },
    result,
    error,
    stderr: child.stderr,
    stdoutOnFailure: error ? child.stdout : null,
    remainingGates: [
      'numerical feature parity',
      'source resampling acceptance',
      'representative quality',
      'browser WASM/WebGPU/hybrid',
      'fp16 promotion',
    ],
  };
  fs.writeFileSync(output, JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
  console.log(
    JSON.stringify({
      output,
      status: receipt.status,
      host: result?.host,
      tokens: result?.transcript.native.debug.tokenIds.length,
      waveform: waveform.status,
      outputComparison: comparison?.pass,
      wallMilliseconds: execution.wallMilliseconds,
    }),
  );
  if (error) throw new Error(error);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
