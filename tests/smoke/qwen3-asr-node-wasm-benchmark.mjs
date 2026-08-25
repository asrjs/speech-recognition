#!/usr/bin/env node

import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function valueAfter(argv, index, flag) {
  const value = argv[index];
  if (!value || value.startsWith('--')) throw new Error(`Missing value for ${flag}.`);
  return value;
}

function parseArgs(argv) {
  const args = {
    modelDir: process.env.QWEN3_ASR_MODEL_DIR,
    audio: process.env.QWEN3_ASR_AUDIO ?? 'tests/fixtures/jfk2.en.wav',
    backend: process.env.QWEN3_ASR_BACKEND ?? 'wasm',
    warmup: 1,
    runs: 3,
    language: undefined,
    maxNewTokens: 256,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--model-dir') args.modelDir = valueAfter(argv, ++index, arg);
    else if (arg === '--audio') args.audio = valueAfter(argv, ++index, arg);
    else if (arg === '--backend') args.backend = valueAfter(argv, ++index, arg);
    else if (arg === '--warmup') args.warmup = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--runs') args.runs = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--language') args.language = valueAfter(argv, ++index, arg);
    else if (arg === '--max-new-tokens') args.maxNewTokens = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--help' || arg === '-h') args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function usage() {
  return `Usage: npm run build && node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs --model-dir <artifact-dir> [options]\n\nOptions:\n  --model-dir <dir>       Local goryodog-style artifact directory (required; no download)\n  --audio <wav>           WAV fixture (default: tests/fixtures/jfk2.en.wav)\n  --backend <id>          wasm or webgpu (default: wasm)\n  --warmup <n>            Warmup runs (default: 1)\n  --runs <n>              Measured runs (default: 3)\n  --language <name|code>  Optional forced language, e.g. Turkish or tr\n  --max-new-tokens <n>    Generation cap (default: 256)\n`;
}

async function firstExisting(candidates, label) {
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next known layout without guessing a remote source.
    }
  }
  throw new Error(`Could not find ${label}. Tried:\n${candidates.join('\n')}`);
}

function readPcmSample(data, offset, format, bits) {
  if (format === 1 && bits === 16) return data.readInt16LE(offset) / 32768;
  if (format === 1 && bits === 24) return data.readIntLE(offset, 3) / 8388608;
  if (format === 1 && bits === 32) return data.readInt32LE(offset) / 2147483648;
  if (format === 3 && bits === 32) return data.readFloatLE(offset);
  throw new Error(`Unsupported WAV encoding: format=${format}, bits=${bits}.`);
}

function decodeWav(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Only RIFF/WAVE fixtures are supported.');
  }
  let offset = 12;
  let format;
  let data;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    if (id === 'fmt ') {
      format = {
        audioFormat: buffer.readUInt16LE(start),
        channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4),
        bitsPerSample: buffer.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      data = buffer.subarray(start, Math.min(buffer.length, start + size));
    }
    offset = start + size + (size % 2);
  }
  if (!format || !data || format.channels < 1)
    throw new Error('WAV fixture must contain valid fmt/data chunks.');
  const bytesPerSample = format.bitsPerSample / 8;
  const frameCount = Math.floor(data.length / (bytesPerSample * format.channels));
  const pcm = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      sum += readPcmSample(
        data,
        (frame * format.channels + channel) * bytesPerSample,
        format.audioFormat,
        format.bitsPerSample,
      );
    }
    pcm[frame] = sum / format.channels;
  }
  return { pcm, sampleRate: format.sampleRate, durationSeconds: frameCount / format.sampleRate };
}

function percentile(values, fraction) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(usage());
    return;
  }
  if (!args.modelDir)
    throw new Error('--model-dir is required; this harness never downloads Qwen weights.');
  if (!['wasm', 'webgpu'].includes(args.backend))
    throw new Error('--backend must be wasm or webgpu.');
  if (
    !Number.isInteger(args.runs) ||
    args.runs < 1 ||
    !Number.isInteger(args.warmup) ||
    args.warmup < 0
  ) {
    throw new Error('--runs must be >= 1 and --warmup must be >= 0.');
  }
  if (!Number.isInteger(args.maxNewTokens) || args.maxNewTokens < 1)
    throw new Error('--max-new-tokens must be positive.');

  const modelDir = path.resolve(args.modelDir);
  const encoder = await firstExisting(
    [
      path.join(modelDir, 'onnx', 'audio_encoder_fp16.onnx'),
      path.join(modelDir, 'audio_encoder_fp16.onnx'),
    ],
    'Qwen encoder graph',
  );
  const decoder = await firstExisting(
    [
      path.join(modelDir, 'onnx', 'decoder_with_past_fp16.onnx'),
      path.join(modelDir, 'decoder_with_past_fp16.onnx'),
    ],
    'Qwen decoder graph',
  );
  const tokenizer = await firstExisting(
    [path.join(modelDir, 'processor', 'tokenizer.json'), path.join(modelDir, 'tokenizer.json')],
    'Qwen tokenizer',
  );
  const encoderData = await firstExisting(
    [
      path.join(modelDir, 'onnx', 'audio_encoder_fp16.onnx_data'),
      path.join(modelDir, 'audio_encoder_fp16.onnx_data'),
    ],
    'Qwen encoder external data',
  );
  const decoderData = await firstExisting(
    [
      path.join(modelDir, 'onnx', 'decoder_with_past_fp16.onnx_data'),
      path.join(modelDir, 'decoder_with_past_fp16.onnx_data'),
    ],
    'Qwen decoder external data',
  );
  const audioPath = path.resolve(args.audio);
  await access(audioPath);
  const audio = decodeWav(await readFile(audioPath));

  const { loadSpeechModel } = await import('../../dist/index.js');
  const loaded = await loadSpeechModel({
    family: 'qwen-asr',
    modelId: 'Qwen/Qwen3-ASR-0.6B-hf',
    backend: args.backend,
    options: {
      source: {
        kind: 'direct',
        artifacts: {
          encoderUrl: pathToFileURL(encoder).href,
          decoderUrl: pathToFileURL(decoder).href,
          tokenizerUrl: pathToFileURL(tokenizer).href,
          encoderDataUrl: pathToFileURL(encoderData).href,
          decoderDataUrl: pathToFileURL(decoderData).href,
          encoderDataPath: path.basename(encoderData),
          decoderDataPath: path.basename(decoderData),
        },
        encoderBackend: args.backend,
        decoderBackend: args.backend,
        cpuThreads: 1,
      },
    },
  });
  const transcribeOptions = {
    detail: 'segments',
    language: args.language,
    maxNewTokens: args.maxNewTokens,
  };
  try {
    for (let index = 0; index < args.warmup; index += 1) {
      await loaded.transcribeMonoPcm(audio.pcm, audio.sampleRate, transcribeOptions);
    }
    const elapsed = [];
    const results = [];
    for (let index = 0; index < args.runs; index += 1) {
      const started = performance.now();
      const result = await loaded.transcribeMonoPcm(audio.pcm, audio.sampleRate, transcribeOptions);
      const elapsedMs = performance.now() - started;
      elapsed.push(elapsedMs);
      results.push(result);
    }
    const last = results.at(-1);
    const meanMs = elapsed.reduce((sum, value) => sum + value, 0) / elapsed.length;
    console.log(
      JSON.stringify(
        {
          model: 'Qwen3-ASR-0.6B',
          backend: args.backend,
          audio: audioPath,
          durationSeconds: audio.durationSeconds,
          warmup: args.warmup,
          runs: args.runs,
          elapsedMs: {
            mean: meanMs,
            p50: percentile(elapsed, 0.5),
            p95: percentile(elapsed, 0.95),
          },
          rtfx: audio.durationSeconds > 0 ? audio.durationSeconds / (meanMs / 1000) : null,
          text: last?.text ?? '',
          language: last?.meta?.language,
          metrics: last?.meta?.metrics,
        },
        null,
        2,
      ),
    );
  } finally {
    await loaded.dispose();
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exit(1);
});
