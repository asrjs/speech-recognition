import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

import { PcmAudioBuffer, createSpeechRuntime, createWasmBackend, loadSpeechModel } from '@asrjs/speech-recognition';
import { createDefaultNodeAssetProvider } from '@asrjs/speech-recognition/io/node';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

const DEFAULT_AUDIO = path.resolve(
  process.cwd(),
  'tools/data/fixtures/audio/librivox.org.wav',
);
const DEFAULT_MODEL_DIR = 'N:\\models\\onnx\\nemo\\parakeet-tdt-0.6b-v2-onnx';
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

function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    modelId: 'parakeet-tdt-0.6b-v2',
    modelDir: DEFAULT_MODEL_DIR,
    audio: DEFAULT_AUDIO,
    targetSampleRate: 16000,
    encoderQuant: 'fp32',
    decoderQuant: 'fp32',
    preprocessor: 'nemo128',
    preprocessorBackend: 'onnx',
    cpuThreads: 4,
    output: null,
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === '--model-id') options.modelId = args[++index];
    else if (arg === '--model-dir') options.modelDir = path.resolve(args[++index]);
    else if (arg === '--audio') options.audio = path.resolve(args[++index]);
    else if (arg === '--target-sample-rate') options.targetSampleRate = Number(args[++index]);
    else if (arg === '--encoder-quant') options.encoderQuant = args[++index];
    else if (arg === '--decoder-quant') options.decoderQuant = args[++index];
    else if (arg === '--preprocessor') options.preprocessor = args[++index];
    else if (arg === '--preprocessor-backend') options.preprocessorBackend = args[++index];
    else if (arg === '--cpu-threads') options.cpuThreads = Number(args[++index]);
    else if (arg === '--output') options.output = path.resolve(args[++index]);
  }

  return options;
}

function loadWavAsFloat32(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (
    buffer.toString('ascii', 0, 4) !== 'RIFF' ||
    buffer.toString('ascii', 8, 12) !== 'WAVE'
  ) {
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

function resolveModelArtifacts(options) {
  const encoderName = `encoder-model${QUANT_SUFFIX[options.encoderQuant]}`;
  const decoderName = `decoder_joint-model${QUANT_SUFFIX[options.decoderQuant]}`;
  const tokenizerName = 'vocab.txt';
  const preprocessorName = `${options.preprocessor}.onnx`;
  const encoderPath = ensureFile(path.join(options.modelDir, encoderName), 'Encoder model');
  const decoderPath = ensureFile(path.join(options.modelDir, decoderName), 'Decoder model');
  const tokenizerPath = ensureFile(path.join(options.modelDir, tokenizerName), 'Tokenizer');
  const preprocessorPath = ensureFile(
    path.join(options.modelDir, preprocessorName),
    'Preprocessor model',
  );
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

async function main() {
  const options = parseArgs();
  ensureFile(options.audio, 'Audio file');
  ensureFile(options.modelDir, 'Model directory');
  const artifacts = resolveModelArtifacts(options);

  const audioStart = performance.now();
  const sourceAudio = loadWavAsFloat32(options.audio);
  const resampleStart = performance.now();
  const audioForModel = resampleLinear(
    sourceAudio.audio,
    sourceAudio.sampleRate,
    options.targetSampleRate,
  );
  const resampleMs = performance.now() - resampleStart;

  const runtime = createSpeechRuntime({
    assetProvider: createDefaultNodeAssetProvider(),
  });
  runtime.registerBackend(createWasmBackend());
  runtime.registerModelFamily(createNemoTdtModelFamily());
  runtime.registerPreset(
    createParakeetPresetFactory({
      useManifestSource: false,
    }),
  );

  const loadStart = performance.now();
  const loaded = await loadSpeechModel({
    runtime,
    preset: 'parakeet',
    modelId: options.modelId,
    backend: 'wasm',
    options: {
      source: {
        kind: 'direct',
        artifacts: {
          encoderUrl: toFileUrl(artifacts.encoderPath),
          decoderUrl: toFileUrl(artifacts.decoderPath),
          tokenizerUrl: toFileUrl(artifacts.tokenizerPath),
          preprocessorUrl: toFileUrl(artifacts.preprocessorPath),
          encoderDataUrl: artifacts.encoderDataPath ? toFileUrl(artifacts.encoderDataPath) : undefined,
          decoderDataUrl: artifacts.decoderDataPath ? toFileUrl(artifacts.decoderDataPath) : undefined,
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
  const result = await loaded.transcribe(
    PcmAudioBuffer.fromMono(audioForModel, options.targetSampleRate),
    {
      detail: 'detailed',
      responseFlavor: 'canonical+native',
      returnTokenIds: true,
      returnFrameIndices: true,
      returnLogProbs: true,
      returnTdtSteps: true,
    },
  );
  const transcribeMs = performance.now() - transcribeStart;

  const text = result.canonical.text ?? result.native?.utteranceText ?? '';
  const payload = {
    model: {
      modelId: options.modelId,
      modelDir: options.modelDir,
      backend: loaded.model.backend.id,
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
    },
    preparation: {
      loadAudioMs: +(resampleStart - audioStart).toFixed(2),
      resampleMs: +resampleMs.toFixed(2),
      resampler: sourceAudio.sampleRate === options.targetSampleRate ? 'none' : 'linear',
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
  };

  const encoded = JSON.stringify(payload, null, 2);
  if (options.output) {
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, encoded, 'utf8');
  } else {
    console.log(encoded);
  }

  await loaded.dispose();
  await runtime.dispose();
}

main().catch((error) => {
  console.error('[node-asrjs-nemo-inspect] failed:', error);
  process.exitCode = 1;
});
