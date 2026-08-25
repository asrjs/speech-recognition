#!/usr/bin/env node
/**
 * Run a real merged Whisper decoder with word timestamps.
 *
 * This smoke is intentionally artifact-local. It validates merged decode and
 * timestamp-token fallback bounds; it does not turn interpolation into an
 * attention-DTW claim when the decoder has no cross_attentions.* outputs.
 *
 * Usage:
 *   WHISPER_MERGED_FIXTURE_DIR=<model-root> \
 *   WHISPER_MERGED_AUDIO=<wav> \
 *   ASRJS_FIXTURE_SMOKE=1 \
 *   node tests/smoke/whisper-merged-timestamp-smoke.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

const ENABLED_VALUES = new Set(['1', 'true', 'yes', 'on']);

function enabled(value) {
  return ENABLED_VALUES.has(String(value ?? '').toLowerCase());
}

function usage() {
  return [
    'Usage: node tests/smoke/whisper-merged-timestamp-smoke.mjs [options]',
    '',
    'Environment:',
    '  WHISPER_MERGED_FIXTURE_DIR  Local Whisper model root containing config/tokenizer/onnx',
    '  WHISPER_MERGED_AUDIO        Local 16 kHz PCM16 WAV fixture',
    '  WHISPER_MERGED_LANGUAGE     Language prompt (default: config language or tr)',
    '  ASRJS_FIXTURE_SMOKE=1       Enable the smoke; otherwise it skips',
    '',
    'Options:',
    '  --model-dir <directory>     Override WHISPER_MERGED_FIXTURE_DIR',
    '  --audio <file>              Override WHISPER_MERGED_AUDIO',
    '  --max-new-tokens <count>    Decode limit (default: 96)',
    '  --force                     Run without ASRJS_FIXTURE_SMOKE=1',
    '  --help                      Show this help',
  ].join('\n');
}

function parseArgs(argv) {
  const args = {
    modelDir: process.env.WHISPER_MERGED_FIXTURE_DIR,
    audioPath: process.env.WHISPER_MERGED_AUDIO,
    maxNewTokens: Number(process.env.WHISPER_MERGED_MAX_NEW_TOKENS ?? 96),
    force: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--model-dir') args.modelDir = argv[++index];
    else if (arg === '--audio') args.audioPath = argv[++index];
    else if (arg === '--max-new-tokens') args.maxNewTokens = Number(argv[++index]);
    else if (arg === '--force') args.force = true;
    else if (arg === '--help' || arg === '-h') {
      console.log(usage());
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function firstExisting(paths) {
  return paths.find((candidate) => fs.existsSync(candidate));
}

function readPcm16Wav(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`Not a RIFF/WAVE file: ${filePath}`);
  }

  let offset = 12;
  let format;
  let data;
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkStart = offset + 8;
    if (chunkId === 'fmt ') {
      format = {
        audioFormat: buffer.readUInt16LE(chunkStart),
        channels: buffer.readUInt16LE(chunkStart + 2),
        sampleRate: buffer.readUInt32LE(chunkStart + 4),
        bitsPerSample: buffer.readUInt16LE(chunkStart + 14),
      };
    } else if (chunkId === 'data') {
      data = buffer.subarray(chunkStart, chunkStart + chunkSize);
    }
    offset = chunkStart + chunkSize + (chunkSize % 2);
  }

  if (!format || !data || format.audioFormat !== 1 || format.bitsPerSample !== 16) {
    throw new Error('Only PCM16 WAV fixtures are supported.');
  }
  if (format.sampleRate !== 16000 || format.channels < 1) {
    throw new Error(`Expected 16 kHz WAV with at least one channel, got ${format.sampleRate} Hz.`);
  }

  const frameCount = Math.floor(data.length / (2 * format.channels));
  const pcm = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < format.channels; channel += 1) {
      sum += data.readInt16LE((frame * format.channels + channel) * 2) / 32768;
    }
    pcm[frame] = sum / format.channels;
  }
  return {
    sampleRate: format.sampleRate,
    durationSeconds: frameCount / format.sampleRate,
    channels: [pcm],
    numberOfChannels: 1,
    numberOfFrames: frameCount,
  };
}

function maxEnd(values, key) {
  return Math.max(0, ...values.map((value) => Number(value[key] ?? 0)));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.force && !enabled(process.env.ASRJS_FIXTURE_SMOKE)) {
    console.log('merged timestamp smoke skipped: set ASRJS_FIXTURE_SMOKE=1 or pass --force');
    return;
  }
  if (!args.modelDir || !args.audioPath) {
    throw new Error(
      'Set WHISPER_MERGED_FIXTURE_DIR and WHISPER_MERGED_AUDIO, or pass --model-dir and --audio.',
    );
  }
  if (!Number.isInteger(args.maxNewTokens) || args.maxNewTokens < 1) {
    throw new Error('--max-new-tokens must be a positive integer.');
  }

  const modelDir = path.resolve(args.modelDir);
  const audioPath = path.resolve(args.audioPath);
  const configPath = path.join(modelDir, 'config.json');
  const tokenizerPath = path.join(modelDir, 'tokenizer.json');
  const encoderPath = firstExisting([
    path.join(modelDir, 'onnx', 'encoder_model.onnx'),
    path.join(modelDir, 'encoder_model.onnx'),
  ]);
  const decoderPath = firstExisting([
    path.join(modelDir, 'onnx', 'decoder_model_merged.onnx'),
    path.join(modelDir, 'decoder_model_merged.onnx'),
  ]);
  for (const required of [configPath, tokenizerPath, encoderPath, decoderPath, audioPath]) {
    if (!required || !fs.existsSync(required))
      throw new Error(`Missing merged smoke artifact: ${required}`);
  }

  const { WhisperOnnxExecutor } = await import('../../dist/models/whisper-seq2seq/executor.js');
  const { parseWhisperModelConfig } =
    await import('../../dist/models/whisper-seq2seq/generation-config.js');
  const configRaw = readJson(configPath);
  const modelPart = parseWhisperModelConfig(configRaw);
  const modelConfig = {
    ecosystem: 'openai',
    architecture: 'whisper-seq2seq',
    processorArchitecture: 'whisper-mel',
    encoderArchitecture: 'whisper-transformer',
    decoderArchitecture: 'transformer-decoder',
    sampleRate: 16000,
    melBins: modelPart.numMelBins ?? 80,
    maxSourcePositions: Number(configRaw.max_source_positions ?? 1500),
    maxTargetPositions: Number(configRaw.max_target_positions ?? 448),
    vocabularySize: Number(configRaw.vocab_size ?? 51865),
    languages: ['auto', 'en', 'tr'],
    tokenizer: { kind: 'tiktoken', vocabSize: Number(configRaw.vocab_size ?? 51865) },
    ...modelPart,
  };
  const source = {
    kind: 'direct',
    artifacts: {
      encoderUrl: pathToFileURL(encoderPath).href,
      decoderUrl: pathToFileURL(decoderPath).href,
      tokenizerUrl: pathToFileURL(tokenizerPath).href,
    },
  };
  const audio = readPcm16Wav(audioPath);
  const executor = new WhisperOnnxExecutor(
    path.basename(modelDir),
    { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
    modelConfig,
    'wasm',
    { source },
  );

  const started = performance.now();
  try {
    const result = await executor.transcribe(
      audio,
      {
        language: process.env.WHISPER_MERGED_LANGUAGE ?? configRaw.language ?? 'tr',
        detail: 'words',
        returnTimestamps: 'word',
        maxNewTokens: args.maxNewTokens,
      },
      { modelId: path.basename(modelDir), config: modelConfig },
    );
    const words = result.words ?? [];
    const segments = result.segments ?? [];
    const maxWordEnd = maxEnd(words, 'endTime');
    const maxSegmentEnd = maxEnd(segments, 'endTime');
    const invalidBounds = [...words, ...segments].some(
      (value) =>
        value.startTime < -0.001 ||
        value.endTime < value.startTime - 0.001 ||
        value.endTime > audio.durationSeconds + 0.001,
    );
    const summary = {
      elapsedMs: Number((performance.now() - started).toFixed(1)),
      audioSeconds: Number(audio.durationSeconds.toFixed(3)),
      text: result.utteranceText,
      wordCount: words.length,
      maxWordEnd: Number(maxWordEnd.toFixed(3)),
      maxSegmentEnd: Number(maxSegmentEnd.toFixed(3)),
      warnings: result.warnings ?? [],
    };
    console.log(JSON.stringify(summary, null, 2));
    if (!result.utteranceText.trim()) throw new Error('Merged decoder returned empty text.');
    if (words.length === 0) throw new Error('Merged timestamp smoke returned no words.');
    if (invalidBounds) throw new Error('Merged timestamp output escaped the input duration.');
  } finally {
    await executor.dispose();
  }
  console.log('merged timestamp smoke passed');
}

main().catch((error) => {
  console.error(error?.stack ?? String(error));
  process.exitCode = 1;
});
