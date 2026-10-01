/**
 * Local fp32 NeMo TDT inference through the public API in Node.js or Bun.
 * Run `npm run build`, then `node examples/nemo-tdt-native-local.mjs --help`.
 * This experimental recipe does not download models or resample audio.
 */
import { readFile, stat } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import {
  createSpeechRuntime,
  createNativeCpuBackend,
  PcmAudioBuffer,
} from '@asrjs/speech-recognition';
import {
  createNemoTdtModelFamily,
  parseNemoTdtConfig,
} from '@asrjs/speech-recognition/models/nemo-tdt';
import {
  createNodeFileSystemAssetProvider,
  decodeWavToMonoPcm,
} from '@asrjs/speech-recognition/io/node';

export const HELP = `Usage: node (or bun) examples/nemo-tdt-native-local.mjs
  --model-dir <local fp32 directory> --config <audited runtime config.json>
  --model-id <candidate ID> --audio <WAV at the configured sample rate>
  [--threads <positive integer, default 4>] [--help]

Expected export layout: encoder-model.onnx, decoder_joint-model.onnx, vocab.txt.
Optional external data must be colocated as <graph filename>.data.
Config must describe this candidate's dimensions and tokenizer, not raw NeMo YAML.
Native CPU is explicit and experimental: no WASM fallback, no automatic download.
PCM WAV 8/16/24/32 and IEEE float WAV 32/64 are supported. Channels are averaged;
sample rate and loudness are unchanged. Resample externally when required.
Progress goes to stderr; full canonical/native JSON is printed after cleanup.
Successful execution is not numerical model parity or browser acceptance.
`;

export function parseNativeExampleArgs(args) {
  const { values } = parseArgs({
    args,
    strict: true,
    allowPositionals: false,
    options: {
      'model-dir': { type: 'string' },
      config: { type: 'string' },
      'model-id': { type: 'string' },
      audio: { type: 'string' },
      threads: { type: 'string', default: '4' },
      help: { type: 'boolean' },
    },
  });
  if (values.help) return { help: true };
  for (const key of ['model-dir', 'config', 'model-id', 'audio']) {
    if (typeof values[key] !== 'string' || !values[key].trim()) {
      throw new Error(`Required --${key} is missing or empty. Use --help.`);
    }
  }
  const threads = Number(values.threads);
  if (!/^\d+$/.test(values.threads) || !Number.isSafeInteger(threads) || threads < 1) {
    throw new Error('--threads must be a positive safe integer.');
  }
  return {
    modelDir: resolve(values['model-dir']),
    configPath: resolve(values.config),
    modelId: values['model-id'],
    audioPath: resolve(values.audio),
    threads,
  };
}

/** @param {import('@asrjs/speech-recognition').RuntimeProgressEvent | import('@asrjs/speech-recognition').TranscriptionProgressEvent} event */
export function formatNativeExampleProgress(event) {
  const label = 'phase' in event ? event.phase : event.stage;
  return `[${label}] ${event.message ?? ''}`;
}

async function localFile(path, optional = false) {
  let info;
  try {
    info = await stat(path);
  } catch (error) {
    if (optional && error.code === 'ENOENT') return undefined;
    throw new Error(`Cannot read required local file: ${path}`, { cause: error });
  }
  if (!info.isFile() || info.size === 0) throw new Error(`Expected a nonempty local file: ${path}`);
  return pathToFileURL(path).href;
}

export async function prepareNativeExampleInput(options) {
  const encoderFilename = 'encoder-model.onnx';
  const decoderFilename = 'decoder_joint-model.onnx';
  const [encoderUrl, decoderUrl, tokenizerUrl, encoderDataUrl, decoderDataUrl] = await Promise.all([
    localFile(join(options.modelDir, encoderFilename)),
    localFile(join(options.modelDir, decoderFilename)),
    localFile(join(options.modelDir, 'vocab.txt')),
    localFile(join(options.modelDir, `${encoderFilename}.data`), true),
    localFile(join(options.modelDir, `${decoderFilename}.data`), true),
    localFile(options.configPath),
    localFile(options.audioPath),
  ]);
  const override = JSON.parse(await readFile(options.configPath, 'utf8'));
  for (const key of [
    'sampleRate',
    'frameShiftSeconds',
    'subsamplingFactor',
    'melBins',
    'predictionHiddenSize',
    'predictionLayers',
    'maxSymbolsPerStep',
    'vocabularySize',
  ]) {
    if (
      !Number.isFinite(override?.[key]) ||
      override[key] <= 0 ||
      (key !== 'frameShiftSeconds' && !Number.isSafeInteger(override[key]))
    ) {
      throw new Error(`Audited runtime config requires positive ${key}.`);
    }
  }
  if (
    !Number.isSafeInteger(override.tokenizer?.blankTokenId) ||
    override.tokenizer.blankTokenId < 0 ||
    override.tokenizer.blankTokenId >= override.vocabularySize ||
    override.tokenizer.kind !== 'sentencepiece'
  ) {
    throw new Error('Audited runtime config requires the candidate sentencepiece blank token ID.');
  }
  const config = parseNemoTdtConfig(options.modelId, override);
  const decoded = decodeWavToMonoPcm(await readFile(options.audioPath));
  if (!decoded.pcm.length) throw new Error('Input WAV contains no audio frames.');
  if (decoded.sampleRate !== config.sampleRate) {
    throw new Error(
      `Input WAV is ${decoded.sampleRate} Hz; model requires ${config.sampleRate} Hz. ` +
        'This example does not silently resample or relabel PCM.',
    );
  }
  return {
    config,
    audio: PcmAudioBuffer.fromMono(decoded.pcm, decoded.sampleRate),
    sourceChannels: decoded.numberOfChannels,
    source: {
      kind: 'direct',
      encoderBackend: 'native-cpu',
      decoderBackend: 'native-cpu',
      preprocessorBackend: 'js',
      cpuThreads: options.threads,
      artifacts: {
        encoderUrl,
        decoderUrl,
        tokenizerUrl,
        encoderDataUrl,
        decoderDataUrl,
        encoderFilename,
        decoderFilename,
      },
    },
  };
}

async function getEngineVersions() {
  const ort = await import('onnxruntime-node');
  return { ...ort.env.versions };
}

/** Optional dependency injection is for lifecycle simulations, not real acceptance. */
export async function runNativeLocalExample(
  options,
  {
    createRuntime = createSpeechRuntime,
    getEngineVersions: versions = getEngineVersions,
    onProgress = () => {},
  } = {},
) {
  const prepared = await prepareNativeExampleInput(options);
  const runtime = createRuntime({
    backends: [createNativeCpuBackend()],
    modelFamilies: [createNemoTdtModelFamily()],
    assetProvider: createNodeFileSystemAssetProvider(),
    hooks: { onProgress },
  });
  let failed = false;
  let primaryError;
  try {
    const model = await runtime.loadModel({
      family: 'nemo-tdt',
      modelId: options.modelId,
      backend: 'native-cpu',
      selectionCriteria: {
        allowExperimental: true,
        environment: 'node',
        requiredPrecision: 'fp32',
      },
      options: { config: prepared.config, source: prepared.source },
    });
    if (model.backend.id !== 'native-cpu')
      throw new Error('Requested native CPU backend was not selected.');
    const session = await model.createSession();
    const transcript = await session.transcribe(prepared.audio, {
      detail: 'detailed',
      responseFlavor: 'canonical+native',
      gridBatching: false,
      returnTokenIds: true,
      returnFrameIndices: true,
      returnTdtSteps: true,
      onProgress,
    });
    return {
      status: 'RUN_COMPLETED_NOT_MODEL_ACCEPTANCE',
      host: {
        runtime: process.versions.bun ? 'bun' : 'node',
        version: process.versions.bun ?? process.versions.node,
      },
      backend: {
        id: model.backend.id,
        engine: 'onnxruntime-node',
        provider: 'cpu',
        requestedPrecision: 'fp32',
        versions: await versions(),
      },
      model: {
        family: 'nemo-tdt',
        id: options.modelId,
        directory: options.modelDir,
        configPath: options.configPath,
      },
      input: {
        path: options.audioPath,
        sampleRate: prepared.audio.sampleRate,
        sourceChannels: prepared.sourceChannels,
        modelChannels: 1,
        samples: prepared.audio.numberOfFrames,
        durationSeconds: prepared.audio.durationSeconds,
      },
      transcript,
    };
  } catch (error) {
    failed = true;
    primaryError = error;
    throw error;
  } finally {
    // Runtime owns loaded models; models own sessions. Await the ownership chain.
    try {
      await runtime.dispose();
    } catch (cleanupError) {
      if (failed)
        throw new AggregateError([primaryError, cleanupError], 'Inference and cleanup failed.');
      throw cleanupError;
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const options = parseNativeExampleArgs(process.argv.slice(2));
    if (options.help) console.log(HELP);
    else {
      const result = await runNativeLocalExample(options, {
        onProgress: (event) => console.error(formatNativeExampleProgress(event)),
      });
      console.log(JSON.stringify(result, null, 2));
    }
  } catch (error) {
    console.error(error);
    process.exitCode = 1;
  }
}
