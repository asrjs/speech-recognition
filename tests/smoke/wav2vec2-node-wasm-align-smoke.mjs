#!/usr/bin/env node
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const DEFAULT_TRANSCRIPT =
  'and so my fellow americans ask not what your country can do for you ask what you can do for your country';

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
    transcript: process.env.WAV2VEC2_ALIGN_TRANSCRIPT ?? DEFAULT_TRANSCRIPT,
    expectWords: ['and', 'country'],
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = readValue(argv, ++i, arg);
    } else if (arg === '--audio') {
      args.audio = readValue(argv, ++i, arg);
    } else if (arg === '--tokenizer') {
      args.tokenizer = readValue(argv, ++i, arg);
    } else if (arg === '--transcript') {
      args.transcript = readValue(argv, ++i, arg);
    } else if (arg === '--expect-word') {
      args.expectWords.push(readValue(argv, ++i, arg).toLowerCase());
    } else if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function usage() {
  return `Usage: node tests/smoke/wav2vec2-node-wasm-align-smoke.mjs [options]\n\nOptions:\n  --model <path>        ONNX model path. Default: /tmp/wav2vec2-base-960h.onnx\n  --audio <path>        WAV fixture path. Default: tests/fixtures/jfk2.en.wav\n  --tokenizer <url>     vocab.json URL or file URL. Default: facebook/wav2vec2-base-960h vocab\n  --transcript <text>   Transcript to force-align. Default: JFK fixture transcript\n  --expect-word <word>  Expected aligned word. Can be repeated. Defaults: and,country\n`;
}

async function requireFile(filePath) {
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

function verifyAlignment(words, expectWords) {
  if (words.length === 0) {
    throw new Error('Forced alignment produced no words.');
  }

  let previousStart = -Infinity;
  let previousEnd = -Infinity;
  for (const word of words) {
    if (!(word.start >= previousStart && word.end >= word.start && word.start >= previousEnd - 0.05)) {
      throw new Error(`Non-monotonic aligned word: ${JSON.stringify(word)}`);
    }
    previousStart = word.start;
    previousEnd = word.end;
  }

  const alignedWords = new Set(words.map((word) => word.text.toLowerCase()));
  const missing = expectWords.filter((word) => !alignedWords.has(word.toLowerCase()));
  if (missing.length > 0) {
    throw new Error(`Missing expected aligned word(s): ${missing.join(', ')}\nAligned: ${words.map((w) => w.text).join(' ')}`);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(usage());
    return;
  }

  const modelPath = await requireFile(args.model);
  await requireFile(`${modelPath}.data`);
  const audioPath = await requireFile(args.audio);
  const { pcm, sampleRate, durationSeconds } = decodeWav(await readFile(audioPath));

  const {
    DEFAULT_WAV2VEC2_CLASSIFICATION,
    DEFAULT_WAV2VEC2_CONFIG,
    OrtWav2Vec2Executor,
  } = await import('../../dist/models/wav2vec2/index.js');
  const { createWav2Vec2AlignerFromLogits } = await import('../../dist/alignment.js');

  const executor = new OrtWav2Vec2Executor(
    'facebook/wav2vec2-base-960h',
    DEFAULT_WAV2VEC2_CLASSIFICATION,
    DEFAULT_WAV2VEC2_CONFIG,
    'wasm',
    {
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
  );

  try {
    await executor.ready();
    const started = performance.now();
    const logits = await executor.extractLogits({
      data: pcm,
      channels: [pcm],
      numberOfChannels: 1,
      numberOfFrames: pcm.length,
      sampleRate,
      durationSeconds,
      format: 'f32-planar',
    });
    const aligner = createWav2Vec2AlignerFromLogits(logits);
    const alignment = aligner.align({ transcript: args.transcript });
    const elapsedMs = performance.now() - started;
    verifyAlignment(alignment.words, args.expectWords);

    console.log('wav2vec2 node/wasm forced-alignment smoke passed');
    console.log(`model=${modelPath}`);
    console.log(`audio=${audioPath}`);
    console.log(`sampleRate=${sampleRate} duration=${durationSeconds.toFixed(3)}s elapsed=${elapsedMs.toFixed(1)}ms`);
    console.log(`frames=${logits.frameCount} vocab=${logits.vocabSize} chars=${alignment.totalChars} words=${alignment.words.length}`);
    const preview = alignment.words
      .slice(0, 12)
      .map((word) => `${word.text}:${word.start.toFixed(2)}-${word.end.toFixed(2)}`)
      .join(' ');
    console.log(preview);
  } finally {
    await executor.dispose();
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exit(1);
});
