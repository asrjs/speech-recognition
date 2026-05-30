#!/usr/bin/env node
/**
 * Smoke test: load Wav2Vec2 base-960h from published HuggingFace ONNX repo.
 *
 * Verifies that loadSpeechModel('facebook/wav2vec2-base-960h') resolves the
 * published ONNX artifact from ysdede/wav2vec2-base-960h-onnx without any
 * local direct source.
 */
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';

function readValue(argv, index, flag) {
  const value = argv[index];
  if (!value || value.startsWith('--')) {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function parseArgs(argv) {
  const args = {
    audio: process.env.WAV2VEC2_SMOKE_AUDIO ?? 'tests/fixtures/jfk2.en.wav',
    expectWords: ['and', 'country'],
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--audio') {
      args.audio = readValue(argv, ++i, arg);
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

  const bytesPerSample = format.bitsPerSample / 8;
  const frameCount = Math.floor(data.length / (bytesPerSample * format.channels));
  const pcm = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      const sampleOffset = (frame * format.channels + channel) * bytesPerSample;
      if (format.audioFormat === 1 && format.bitsPerSample === 16) {
        sum += data.readInt16LE(sampleOffset) / 32768;
      } else if (format.audioFormat === 3 && format.bitsPerSample === 32) {
        sum += data.readFloatLE(sampleOffset);
      }
    }
    pcm[frame] = sum / format.channels;
  }

  return { pcm, sampleRate: format.sampleRate, durationSeconds: frameCount / format.sampleRate };
}

async function requireFile(filePath) {
  const absolute = path.resolve(filePath);
  await access(absolute);
  return absolute;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(`Usage: node tests/smoke/wav2vec2-node-wasm-hf-smoke.mjs [options]

Options:
  --audio <path>        WAV fixture path. Default: tests/fixtures/jfk2.en.wav
  --expect-word <word>  Expected word in transcript. Can be repeated. Defaults: and,country
`);
    return;
  }

  const audioPath = await requireFile(args.audio);
  const { pcm, sampleRate, durationSeconds } = decodeWav(await readFile(audioPath));

  const { loadSpeechModel } = await import('../../dist/runtime/load.js');

  console.log('Loading Wav2Vec2 from HuggingFace ONNX repo...');
  const started = performance.now();

  const loaded = await loadSpeechModel({
    modelId: 'facebook/wav2vec2-base-960h',
    useManifestSources: true,
  });

  const loadMs = performance.now() - started;
  console.log(`Model loaded in ${loadMs.toFixed(0)}ms`);

  try {
    const transcribeStart = performance.now();
    const result = await loaded.transcribe({
      data: pcm,
      channels: [pcm],
      numberOfChannels: 1,
      numberOfFrames: pcm.length,
      sampleRate,
      durationSeconds,
      format: 'f32-planar',
    });
    const transcribeMs = performance.now() - transcribeStart;
    const transcript = result.text ?? '';
    const words = transcript.split(/\s+/).filter(Boolean);

    const missing = args.expectWords.filter(
      (w) => !words.map((x) => x.toLowerCase()).includes(w.toLowerCase()),
    );

    if (missing.length > 0) {
      throw new Error(`Missing expected word(s): ${missing.join(', ')}\nTranscript: ${transcript}`);
    }

    console.log('wav2vec2 HF smoke passed');
    console.log(`audio=${audioPath}`);
    console.log(`sampleRate=${sampleRate} duration=${durationSeconds.toFixed(3)}s`);
    console.log(`load=${loadMs.toFixed(0)}ms transcribe=${transcribeMs.toFixed(0)}ms`);
    console.log(`words=${words.length} transcript=${transcript}`);
  } finally {
    await loaded.dispose();
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exit(1);
});
