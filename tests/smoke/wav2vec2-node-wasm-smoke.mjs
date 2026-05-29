#!/usr/bin/env node
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function readValue(argv, index, flag) {
  const value = argv[index];
  if (!value || value.startsWith('--')) {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function parseArgs(argv) {
  const args = {
    model: process.env.WAV2VEC2_ONNX_MODEL ?? '/tmp/wav2vec2-base-960h.onnx',
    audio: process.env.WAV2VEC2_SMOKE_AUDIO ?? 'tests/fixtures/jfk2.en.wav',
    tokenizer:
      process.env.WAV2VEC2_TOKENIZER_URL ??
      'https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json',
    expects: [],
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = readValue(argv, ++i, arg);
    } else if (arg === '--audio') {
      args.audio = readValue(argv, ++i, arg);
    } else if (arg === '--tokenizer') {
      args.tokenizer = readValue(argv, ++i, arg);
    } else if (arg === '--expect') {
      args.expects.push(readValue(argv, ++i, arg));
    } else if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function usage() {
  return `Usage: node tests/smoke/wav2vec2-node-wasm-smoke.mjs [options]\n\nOptions:\n  --model <path>       ONNX model path. Default: /tmp/wav2vec2-base-960h.onnx\n  --audio <path>       WAV fixture path. Default: tests/fixtures/jfk2.en.wav\n  --tokenizer <url>    vocab.json URL or file URL. Default: facebook/wav2vec2-base-960h vocab\n  --expect <text>      Expected lowercase snippet. Can be repeated. Default: non-empty transcript\n`;
}

async function requireFile(filePath, label) {
  const absolute = path.resolve(filePath);
  await access(absolute);
  return absolute;
}

function decodeWav(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Only RIFF/WAVE fixtures are supported.');
  }

  let offset = 12;
  let format = null;
  let data = null;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + size;
    if (id === 'fmt ') {
      format = {
        audioFormat: buffer.readUInt16LE(start),
        channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4),
        bitsPerSample: buffer.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      data = buffer.subarray(start, end);
    }
    offset = end + (size % 2);
  }

  if (!format || !data) {
    throw new Error('WAV fixture must contain fmt and data chunks.');
  }
  if (format.channels <= 0) {
    throw new Error('WAV fixture has invalid channel count.');
  }

  const bytesPerSample = format.bitsPerSample / 8;
  const frameCount = Math.floor(data.length / (bytesPerSample * format.channels));
  const pcm = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const sampleOffset = (frame * format.channels + channel) * bytesPerSample;
      sum += readPcmSample(data, sampleOffset, format.audioFormat, format.bitsPerSample);
    }
    pcm[frame] = sum / format.channels;
  }

  return { pcm, sampleRate: format.sampleRate, durationSeconds: frameCount / format.sampleRate };
}

function readPcmSample(data, offset, audioFormat, bitsPerSample) {
  if (audioFormat === 1 && bitsPerSample === 16) {
    return data.readInt16LE(offset) / 32768;
  }
  if (audioFormat === 1 && bitsPerSample === 24) {
    return data.readIntLE(offset, 3) / 8388608;
  }
  if (audioFormat === 1 && bitsPerSample === 32) {
    return data.readInt32LE(offset) / 2147483648;
  }
  if (audioFormat === 3 && bitsPerSample === 32) {
    return data.readFloatLE(offset);
  }
  throw new Error(`Unsupported WAV encoding: format=${audioFormat}, bits=${bitsPerSample}.`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(usage());
    return;
  }

  const modelPath = await requireFile(args.model, 'model');
  await requireFile(`${modelPath}.data`, 'external data');
  const audioPath = await requireFile(args.audio, 'audio');
  const { pcm, sampleRate, durationSeconds } = decodeWav(await readFile(audioPath));

  const { loadSpeechModel } = await import('../../dist/index.js');
  const loaded = await loadSpeechModel({
    family: 'wav2vec2',
    modelId: 'facebook/wav2vec2-base-960h',
    backend: 'wasm',
    options: {
      source: {
        kind: 'direct',
        artifacts: {
          modelUrl: pathToFileURL(modelPath).href,
          modelDataUrl: pathToFileURL(`${modelPath}.data`).href,
          modelDataFilename: `${path.basename(modelPath)}.data`,
          tokenizerUrl: args.tokenizer,
        },
        cpuThreads: 1,
      },
    },
  });

  try {
    const started = performance.now();
    const result = await loaded.transcribeMonoPcm(pcm, sampleRate, {
      detail: 'words',
      responseFlavor: 'canonical+native',
      returnTokenIds: true,
      returnConfidence: true,
    });
    const elapsedMs = performance.now() - started;
    const text = String(result.canonical.text ?? '');
    const normalized = text.toLowerCase();
    const missing = args.expects.filter((expected) => !normalized.includes(expected.toLowerCase()));
    if (args.expects.length === 0 && normalized.trim().length === 0) {
      throw new Error('Wav2Vec2 smoke transcript was empty.');
    }
    if (missing.length > 0) {
      throw new Error(`Missing expected snippet(s): ${missing.join(', ')}\nTranscript: ${text}`);
    }

    console.log('wav2vec2 node/wasm smoke passed');
    console.log(`model=${modelPath}`);
    console.log(`audio=${audioPath}`);
    console.log(`sampleRate=${sampleRate} duration=${durationSeconds.toFixed(3)}s elapsed=${elapsedMs.toFixed(1)}ms`);
    console.log(`words=${result.canonical.words?.length ?? 0} tokens=${result.native?.tokens?.length ?? 0}`);
    console.log(text);
  } finally {
    await loaded.dispose();
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exit(1);
});
