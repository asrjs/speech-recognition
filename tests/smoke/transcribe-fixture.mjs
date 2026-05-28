#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const ENABLED_VALUES = new Set(['1', 'true', 'yes', 'on']);

function usage() {
  return `Usage: node tests/smoke/transcribe-fixture.mjs --audio <path> --model <id-or-preset> --expect <text> [options]

Offline fixture transcription smoke test.

Required:
  --audio <path>       Local WAV fixture path.
  --model <id>         Built-in model id or preset alias.
  --expect <text>      Expected text snippet. Can be repeated.

Options:
  --preset <name>      Explicit preset name.
  --family <name>      Explicit model family. Requires --model as modelId.
  --backend <id>       Runtime backend id, e.g. wasm, webgpu, webgpu-hybrid.
  --language <code>    Transcription language option.
  --detail <level>     Transcript detail level. Default: words.
  --force              Fail instead of skip when fixture smoke is disabled or model assets are unavailable.
  --help               Show this help.

Environment:
  ASRJS_FIXTURE_SMOKE=1        Enable the harness.
  ASRJS_FIXTURE_SMOKE_FORCE=1  Treat unavailable model/assets as failure.

Example:
  npm run test:fixture-smoke -- --audio tests/fixtures/sample.wav --model parakeet-tdt-0.6b-v2 --expect "hello"
`;
}

function parseArgs(argv) {
  const args = {
    expects: [],
    detail: 'words',
    force: isEnabled(process.env.ASRJS_FIXTURE_SMOKE_FORCE),
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else if (arg === '--force') {
      args.force = true;
    } else if (arg === '--audio') {
      args.audio = readValue(argv, ++i, arg);
    } else if (arg === '--model') {
      args.model = readValue(argv, ++i, arg);
    } else if (arg === '--expect') {
      args.expects.push(readValue(argv, ++i, arg));
    } else if (arg === '--preset') {
      args.preset = readValue(argv, ++i, arg);
    } else if (arg === '--family') {
      args.family = readValue(argv, ++i, arg);
    } else if (arg === '--backend') {
      args.backend = readValue(argv, ++i, arg);
    } else if (arg === '--language') {
      args.language = readValue(argv, ++i, arg);
    } else if (arg === '--detail') {
      args.detail = readValue(argv, ++i, arg);
    } else {
      throw new CliError(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function readValue(argv, index, flag) {
  const value = argv[index];
  if (!value || value.startsWith('--')) {
    throw new CliError(`Missing value for ${flag}.`);
  }
  return value;
}

function validateArgs(args) {
  if (args.help) {
    return;
  }
  if (!args.audio) {
    throw new CliError('Missing required --audio.');
  }
  if (!args.model && !args.preset) {
    throw new CliError('Missing required --model or --preset.');
  }
  if (args.expects.length === 0) {
    throw new CliError('Missing required --expect.');
  }
  if (args.family && !args.model) {
    throw new CliError('--family requires --model to provide the concrete modelId.');
  }
}

function isEnabled(value) {
  return ENABLED_VALUES.has(String(value ?? '').toLowerCase());
}

function skip(message) {
  console.log(`fixture transcription smoke skipped: ${message}`);
}

class CliError extends Error {}

function decodeWav(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Only RIFF/WAVE fixtures are supported by this smoke harness.');
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

  return { pcm, sampleRate: format.sampleRate };
}

function readPcmSample(data, offset, audioFormat, bitsPerSample) {
  if (audioFormat === 1 && bitsPerSample === 16) {
    return data.readInt16LE(offset) / 32768;
  }
  if (audioFormat === 1 && bitsPerSample === 24) {
    const raw = data.readIntLE(offset, 3);
    return raw / 8388608;
  }
  if (audioFormat === 1 && bitsPerSample === 32) {
    return data.readInt32LE(offset) / 2147483648;
  }
  if (audioFormat === 3 && bitsPerSample === 32) {
    return data.readFloatLE(offset);
  }
  throw new Error(`Unsupported WAV encoding: format=${audioFormat}, bits=${bitsPerSample}.`);
}

function looksLikeUnavailableAssetError(error) {
  const text = `${error?.message ?? error}`.toLowerCase();
  return (
    text.includes('fetch') ||
    text.includes('network') ||
    text.includes('enoent') ||
    text.includes('not found') ||
    text.includes('404') ||
    text.includes('asset') ||
    text.includes('manifest') ||
    text.includes('model')
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  validateArgs(args);

  if (args.help) {
    process.stdout.write(usage());
    return;
  }

  if (!args.force && !isEnabled(process.env.ASRJS_FIXTURE_SMOKE)) {
    skip('set ASRJS_FIXTURE_SMOKE=1 or pass --force to run local model inference');
    return;
  }

  const audioPath = path.resolve(args.audio);
  const { pcm, sampleRate } = decodeWav(await readFile(audioPath));
  const { transcribeSpeechFromMonoPcm } = await import('../../dist/index.js');

  const loadOptions = {
    backend: args.backend ?? 'wasm',
    useManifestSources: true,
  };
  if (args.family) {
    loadOptions.family = args.family;
    loadOptions.modelId = args.model;
  } else if (args.preset) {
    loadOptions.preset = args.preset;
    if (args.model) {
      loadOptions.modelId = args.model;
    }
  } else {
    loadOptions.modelId = args.model;
  }

  try {
    const transcript = await transcribeSpeechFromMonoPcm(pcm, sampleRate, {
      ...loadOptions,
      transcribeOptions: {
        detail: args.detail,
        returnTimestamps: 'word',
        ...(args.language ? { language: args.language } : {}),
      },
    });

    const text = String(transcript.text ?? '');
    const normalized = text.toLowerCase();
    const missing = args.expects.filter((expected) => !normalized.includes(expected.toLowerCase()));
    if (missing.length > 0) {
      throw new Error(`Transcript did not include expected snippet(s): ${missing.join(', ')}\nTranscript: ${text}`);
    }

    console.log(`fixture transcription smoke passed: ${args.expects.length} expected snippet(s) found`);
    console.log(text);
  } catch (error) {
    if (!args.force && looksLikeUnavailableAssetError(error)) {
      skip(`model assets unavailable (${error?.message ?? error})`);
      return;
    }
    throw error;
  }
}

main().catch((error) => {
  if (error instanceof CliError) {
    console.error(error.message);
    console.error('Run with --help for usage.');
    process.exit(2);
  }
  console.error(error?.stack ?? error);
  process.exit(1);
});
