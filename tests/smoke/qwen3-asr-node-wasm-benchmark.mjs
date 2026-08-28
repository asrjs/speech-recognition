#!/usr/bin/env node

import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
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
    encoder: process.env.QWEN3_ASR_ENCODER ?? 'dynamic',
    dtype: process.env.QWEN3_ASR_DTYPE ?? 'fp16',
    warmup: 1,
    runs: 3,
    language: undefined,
    maxNewTokens: 256,
    windowSeconds: undefined,
    overlapSeconds: undefined,
    reference: undefined,
    output: process.env.QWEN3_ASR_OUTPUT,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--model-dir') args.modelDir = valueAfter(argv, ++index, arg);
    else if (arg === '--audio') args.audio = valueAfter(argv, ++index, arg);
    else if (arg === '--backend') args.backend = valueAfter(argv, ++index, arg);
    else if (arg === '--encoder') args.encoder = valueAfter(argv, ++index, arg);
    else if (arg === '--dtype') args.dtype = valueAfter(argv, ++index, arg);
    else if (arg === '--warmup') args.warmup = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--runs') args.runs = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--language') args.language = valueAfter(argv, ++index, arg);
    else if (arg === '--max-new-tokens') args.maxNewTokens = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--window-seconds')
      args.windowSeconds = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--overlap-seconds')
      args.overlapSeconds = Number(valueAfter(argv, ++index, arg));
    else if (arg === '--reference') args.reference = valueAfter(argv, ++index, arg);
    else if (arg === '--output') args.output = valueAfter(argv, ++index, arg);
    else if (arg === '--help' || arg === '-h') args.help = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  return args;
}

function usage() {
  return [
    'Usage: npm run build && node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs --model-dir <artifact-dir> [options]',
    '',
    'Options:',
    '  --model-dir <dir>       Local official or legacy artifact directory (required; no download)',
    '  --audio <wav>           WAV fixture (default: tests/fixtures/jfk2.en.wav)',
    '  --backend <id>          wasm or webgpu (default: wasm)',
    '  --encoder <variant>     dynamic or static-t1100 (default: dynamic)',
    '  --dtype <dtype>         fp16 or fp32 (default: fp16)',
    '  --warmup <n>            Warmup runs (default: 1)',
    '  --runs <n>              Measured runs (default: 3)',
    '  --language <name|code>  Optional forced language, e.g. Turkish or tr',
    '  --max-new-tokens <n>    Generation cap (default: 256)',
    '  --window-seconds <n>    Force model-safe windows of this length for long-audio measurement',
    '  --overlap-seconds <n>   Optional overlap for forced windows (default: model policy)',
    '  --reference <json|txt>  Optional fixture/reference JSON or plain-text label; adjacent .json is used when present',
    '  --output <json>         Optional path for the structured benchmark result',
    '',
  ].join('\n');
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

async function findExisting(candidates) {
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Optional companion data may be absent for self-contained graphs.
    }
  }
  return undefined;
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

function referenceText(candidate) {
  const fields = ['normalized', 'transcription', 'text'];
  const field = fields.find(
    (name) => typeof candidate?.[name] === 'string' && candidate[name].trim(),
  );
  if (!field) return undefined;
  return { field, text: candidate[field] };
}

export async function loadFixtureReference(referencePath, audioSha256) {
  const raw = await readFile(referencePath, 'utf8');
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    const text = raw.trim();
    if (text && /\.txt$/i.test(referencePath)) {
      return {
        path: referencePath,
        field: 'text',
        text,
        kind: 'fixture-text-label',
      };
    }
    throw new Error(
      `Could not read benchmark reference ${referencePath}: ${error?.message ?? error}`,
    );
  }

  if (Array.isArray(parsed?.samples)) {
    const sample = parsed.samples.find((candidate) => candidate?.audio_sha256 === audioSha256);
    if (!sample) {
      const hashes = parsed.samples
        .map((candidate) => candidate?.audio_sha256)
        .filter((candidate) => typeof candidate === 'string');
      throw new Error(
        `Benchmark reference ${referencePath} has no sample matching audio SHA-256 ${audioSha256}. ` +
          (hashes.length
            ? `Available hashes: ${hashes.join(', ')}`
            : 'The reference has no audio hashes.'),
      );
    }
    const resolved = referenceText(sample);
    if (!resolved) {
      throw new Error(
        `Benchmark reference ${referencePath} sample ${sample.sample_id ?? '<unknown>'} has no non-empty text field.`,
      );
    }
    return {
      path: referencePath,
      field: `samples[${parsed.samples.indexOf(sample)}].${resolved.field}`,
      text: resolved.text,
      kind: parsed.reference_kind ?? 'structured-reference',
      sampleId: sample.sample_id,
    };
  }

  const resolved = referenceText(parsed);
  if (!resolved) {
    throw new Error(
      `Benchmark reference ${referencePath} has no non-empty normalized, transcription, or text field.`,
    );
  }
  return {
    path: referencePath,
    field: resolved.field,
    text: resolved.text,
    kind: parsed?.reference_kind ?? 'fixture-sidecar-dataset-label',
  };
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
  if (!['dynamic', 'static-t1100'].includes(args.encoder))
    throw new Error('--encoder must be dynamic or static-t1100.');
  if (!['fp16', 'fp32'].includes(args.dtype)) throw new Error('--dtype must be fp16 or fp32.');
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
  if (
    args.windowSeconds !== undefined &&
    (!Number.isFinite(args.windowSeconds) || args.windowSeconds <= 0)
  )
    throw new Error('--window-seconds must be positive.');
  if (
    args.overlapSeconds !== undefined &&
    (!Number.isFinite(args.overlapSeconds) || args.overlapSeconds < 0)
  )
    throw new Error('--overlap-seconds must be non-negative.');
  if (args.overlapSeconds !== undefined && args.windowSeconds === undefined)
    throw new Error('--overlap-seconds requires --window-seconds.');
  if (
    args.windowSeconds !== undefined &&
    args.overlapSeconds !== undefined &&
    args.overlapSeconds >= args.windowSeconds
  )
    throw new Error('--overlap-seconds must be smaller than --window-seconds.');

  const modelDir = path.resolve(args.modelDir);
  const officialEncoder =
    args.encoder === 'static-t1100'
      ? ['audio-encoder-static-t1100.onnx', 'audio-encoder-static-t1100-fp16.onnx']
      : ['audio-encoder-dynamic.onnx'];
  const encoder = await firstExisting(
    [
      ...officialEncoder.flatMap((filename) => [
        path.join(modelDir, filename),
        path.join(modelDir, 'onnx', filename),
      ]),
      ...(args.encoder === 'dynamic'
        ? [
            path.join(modelDir, 'onnx', 'audio_encoder_fp16.onnx'),
            path.join(modelDir, 'audio_encoder_fp16.onnx'),
          ]
        : []),
    ],
    'Qwen encoder graph',
  );
  const decoderSuffix = args.dtype === 'fp16' ? '-fp16' : '';
  const decoder = await firstExisting(
    [
      path.join(modelDir, `decoder-prefill${decoderSuffix}.onnx`),
      path.join(modelDir, 'onnx', `decoder-prefill${decoderSuffix}.onnx`),
      path.join(modelDir, `decoder_with_past${decoderSuffix}.onnx`),
      path.join(modelDir, 'onnx', `decoder_with_past${decoderSuffix}.onnx`),
    ],
    'Qwen decoder prefill graph',
  );
  const decoderStep = await findExisting([
    path.join(modelDir, `decoder-step${decoderSuffix}.onnx`),
    path.join(modelDir, 'onnx', `decoder-step${decoderSuffix}.onnx`),
  ]);
  const tokenizer = await firstExisting(
    [
      path.join(modelDir, 'tokenizer', 'tokenizer.json'),
      path.join(modelDir, 'processor', 'tokenizer.json'),
      path.join(modelDir, 'tokenizer.json'),
    ],
    'Qwen tokenizer',
  );
  const encoderData = await findExisting([
    `${encoder}.data`,
    path.join(modelDir, 'onnx', 'audio_encoder_fp16.onnx_data'),
    path.join(modelDir, 'audio_encoder_fp16.onnx_data'),
  ]);
  const decoderData = decoderStep
    ? undefined
    : await firstExisting(
        [
          `${decoder}.data`,
          path.join(modelDir, `decoder${decoderSuffix}.onnx.data`),
          path.join(modelDir, 'onnx', `decoder${decoderSuffix}.onnx.data`),
          path.join(modelDir, `decoder_with_past${decoderSuffix}.onnx_data`),
          path.join(modelDir, 'onnx', `decoder_with_past${decoderSuffix}.onnx_data`),
        ],
        'Qwen decoder external data',
      );
  const decoderPrefillData = decoderStep
    ? await firstExisting(
        [
          path.join(modelDir, `decoder${decoderSuffix}.onnx.data`),
          `${decoder}.data`,
          path.join(modelDir, `decoder-prefill${decoderSuffix}.onnx.data`),
          path.join(modelDir, 'onnx', `decoder-prefill${decoderSuffix}.onnx.data`),
        ],
        'Qwen decoder prefill external data',
      )
    : undefined;
  const decoderStepData = decoderStep
    ? await firstExisting(
        [
          path.join(modelDir, `decoder${decoderSuffix}.onnx.data`),
          `${decoderStep}.data`,
          path.join(modelDir, `decoder-step${decoderSuffix}.onnx.data`),
          path.join(modelDir, 'onnx', `decoder-step${decoderSuffix}.onnx.data`),
        ],
        'Qwen decoder step external data',
      )
    : undefined;
  const audioPath = path.resolve(args.audio);
  await access(audioPath);
  const audioBuffer = await readFile(audioPath);
  const audio = decodeWav(audioBuffer);
  const audioSha256 = sha256(audioBuffer);
  const referencePath = args.reference
    ? path.resolve(args.reference)
    : audioPath.replace(/\.wav$/i, '.json');
  let reference;
  if (args.reference) {
    reference = await loadFixtureReference(referencePath, audioSha256);
  } else if (await findExisting([referencePath])) {
    reference = await loadFixtureReference(referencePath, audioSha256);
  }

  const { loadSpeechModel } = await import('../../dist/index.js');
  const { characterErrorRate, normalizeBenchmarkTranscript, wordErrorRate } =
    await import('../../dist/bench.js');
  const directArtifacts = {
    encoderUrl: pathToFileURL(encoder).href,
    decoderUrl: pathToFileURL(decoder).href,
    tokenizerUrl: pathToFileURL(tokenizer).href,
    ...(encoderData
      ? {
          encoderDataUrl: pathToFileURL(encoderData).href,
          encoderDataPath: path.basename(encoderData),
        }
      : {}),
    ...(decoderStep
      ? {
          decoderStepUrl: pathToFileURL(decoderStep).href,
          decoderPrefillDataUrl: pathToFileURL(decoderPrefillData).href,
          decoderPrefillDataPath: path.basename(decoderPrefillData),
          decoderStepDataUrl: pathToFileURL(decoderStepData).href,
          decoderStepDataPath: path.basename(decoderStepData),
        }
      : {
          decoderDataUrl: pathToFileURL(decoderData).href,
          decoderDataPath: path.basename(decoderData),
        }),
  };
  const loaded = await loadSpeechModel({
    family: 'qwen-asr',
    modelId: 'Qwen/Qwen3-ASR-0.6B',
    backend: args.backend,
    options: {
      source: {
        kind: 'direct',
        artifacts: directArtifacts,
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
    ...(args.windowSeconds !== undefined
      ? {
          windowing: 'force',
          windowDurationSeconds: args.windowSeconds,
          ...(args.overlapSeconds !== undefined ? { overlapSeconds: args.overlapSeconds } : {}),
        }
      : {}),
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
    const qualityRuns = reference
      ? results.map((result) => ({
          wordErrorRate: wordErrorRate(reference.text, result.text),
          characterErrorRate: characterErrorRate(reference.text, result.text),
          normalizedExactMatch:
            normalizeBenchmarkTranscript(reference.text) ===
            normalizeBenchmarkTranscript(result.text),
        }))
      : undefined;
    const payload = {
      model: 'Qwen3-ASR-0.6B',
      backend: args.backend,
      encoder: args.encoder,
      dtype: args.dtype,
      audio: audioPath,
      audioSha256,
      durationSeconds: audio.durationSeconds,
      warmup: args.warmup,
      runs: args.runs,
      elapsedMs: {
        mean: meanMs,
        p50: percentile(elapsed, 0.5),
        p95: percentile(elapsed, 0.95),
      },
      rtfx: audio.durationSeconds > 0 ? audio.durationSeconds / (meanMs / 1000) : null,
      windowing:
        args.windowSeconds === undefined
          ? null
          : {
              windowDurationSeconds: args.windowSeconds,
              overlapSeconds: args.overlapSeconds ?? null,
            },
      text: last?.text ?? '',
      language: last?.meta?.language,
      quality: reference
        ? {
            referenceKind: reference.kind,
            referencePath: reference.path,
            referenceField: reference.field,
            ...(reference.sampleId ? { referenceSampleId: reference.sampleId } : {}),
            runs: qualityRuns,
          }
        : null,
      metrics: last?.meta?.metrics,
    };
    const serialized = JSON.stringify(payload, null, 2);
    if (args.output) {
      const outputPath = path.resolve(args.output);
      await mkdir(path.dirname(outputPath), { recursive: true });
      await writeFile(outputPath, `${serialized}\n`, 'utf8');
    }
    console.log(serialized);
  } finally {
    await loaded.dispose();
  }
}

const invokedAsScript =
  process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;

if (invokedAsScript) {
  main().catch((error) => {
    console.error(error?.stack ?? error);
    process.exit(1);
  });
}
