import fs from 'fs';
import path from 'path';
import process from 'process';
import { pathToFileURL } from 'url';

function readUInt32LE(buf, off) {
  return buf.readUInt32LE(off);
}

function readUInt16LE(buf, off) {
  return buf.readUInt16LE(off);
}

function loadWavAsFloat32(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error(`Unsupported WAV container: ${filePath}`);
  }

  let offset = 12;
  let audioFormat = null;
  let numChannels = null;
  let sampleRate = null;
  let bitsPerSample = null;
  let dataOffset = null;
  let dataSize = null;

  while (offset + 8 <= buf.length) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = readUInt32LE(buf, offset + 4);
    const chunkDataStart = offset + 8;
    const next = chunkDataStart + chunkSize + (chunkSize % 2);

    if (chunkId === 'fmt ') {
      audioFormat = readUInt16LE(buf, chunkDataStart + 0);
      numChannels = readUInt16LE(buf, chunkDataStart + 2);
      sampleRate = readUInt32LE(buf, chunkDataStart + 4);
      bitsPerSample = readUInt16LE(buf, chunkDataStart + 14);
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
  let idx = dataOffset;
  for (let i = 0; i < totalFrames; i++) {
    let sum = 0;
    for (let c = 0; c < numChannels; c++) {
      let v = 0;
      if (audioFormat === 1 && bitsPerSample === 16) {
        v = buf.readInt16LE(idx) / 32768;
      } else if (audioFormat === 1 && bitsPerSample === 32) {
        v = buf.readInt32LE(idx) / 2147483648;
      } else if (audioFormat === 3 && bitsPerSample === 32) {
        v = buf.readFloatLE(idx);
      } else {
        throw new Error(`Unsupported WAV format. format=${audioFormat}, bits=${bitsPerSample}`);
      }
      sum += v;
      idx += bytesPerSample;
    }
    mono[i] = sum / numChannels;
  }

  return { audio: mono, sampleRate };
}

function resampleLinear(audio, fromRate, toRate) {
  if (fromRate === toRate) return audio;
  if (!Number.isFinite(fromRate) || !Number.isFinite(toRate) || fromRate <= 0 || toRate <= 0) {
    throw new Error(`Invalid resample rates: from=${fromRate}, to=${toRate}`);
  }

  const ratio = toRate / fromRate;
  const outLength = Math.max(1, Math.round(audio.length * ratio));
  const out = new Float32Array(outLength);
  const scale = fromRate / toRate;
  for (let i = 0; i < outLength; i++) {
    const pos = i * scale;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, audio.length - 1);
    const frac = pos - i0;
    out[i] = audio[i0] * (1 - frac) + audio[i1] * frac;
  }
  return out;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    model: 'Xenova/whisper-tiny.en',
    audio: null,
    mode: 'segments',
    chunkLengthS: 30,
    strideLengthS: 5,
    local: process.env.TRANSFORMERS_LOCAL !== 'false',
    localModule: path.resolve(process.cwd(), '../reference-js-runtime/dist/reference.node.mjs'),
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--model') out.model = args[++i];
    else if (a === '--audio') out.audio = path.resolve(process.cwd(), args[++i]);
    else if (a === '--mode') out.mode = args[++i];
    else if (a === '--chunk-length-s') out.chunkLengthS = Number(args[++i]);
    else if (a === '--stride-length-s') out.strideLengthS = Number(args[++i]);
    else if (a === '--local') out.local = true;
    else if (a === '--npm') out.local = false;
    else if (a === '--local-module') out.localModule = path.resolve(process.cwd(), args[++i]);
  }

  if (!out.audio) {
    throw new Error('Missing --audio <path-to-wav-file>.');
  }
  if (!['none', 'segments', 'words'].includes(out.mode)) {
    throw new Error(`Invalid --mode ${out.mode}. Use one of: none, segments, words.`);
  }
  return out;
}

async function main() {
  const opts = parseArgs();
  const modulePath = opts.local ? opts.localModule : '@huggingface/transformers';

  if (!fs.existsSync(opts.audio)) {
    throw new Error(`Audio file not found: ${opts.audio}`);
  }
  if (path.extname(opts.audio).toLowerCase() !== '.wav') {
    throw new Error(`WAV input required. Got: ${opts.audio}`);
  }
  if (opts.local && !fs.existsSync(modulePath)) {
    throw new Error(
      `Local reference module not found at: ${modulePath}\n` +
      'Build your local JS reference runtime first and point --local-module at it.',
    );
  }

  const importTarget = opts.local ? pathToFileURL(modulePath).href : modulePath;
  const mod = await import(importTarget);
  const { env, pipeline } = mod;

  env.allowLocalModels = true;
  env.allowRemoteModels = true;

  const transcriber = await pipeline('automatic-speech-recognition', opts.model, { device: 'cpu' });
  const { audio, sampleRate } = loadWavAsFloat32(opts.audio);
  const targetRate = transcriber?.processor?.feature_extractor?.config?.sampling_rate ?? sampleRate;
  const audioForModel = resampleLinear(audio, sampleRate, targetRate);

  const runOpts = {};
  if (opts.mode === 'segments') {
    runOpts.return_timestamps = true;
  } else if (opts.mode === 'words') {
    runOpts.return_timestamps = 'word';
  }
  if (audioForModel.length / targetRate > 30) {
    runOpts.chunk_length_s = opts.chunkLengthS;
    runOpts.stride_length_s = opts.strideLengthS;
  }

  const output = await transcriber(audioForModel, runOpts);

  console.log(JSON.stringify({
    model: opts.model,
    mode: opts.mode,
    audio: {
      file: opts.audio,
      source_sample_rate: sampleRate,
      model_sample_rate: targetRate,
      source_seconds: +(audio.length / sampleRate).toFixed(3),
      model_seconds: +(audioForModel.length / targetRate).toFixed(3),
    },
    options: runOpts,
    output,
  }, null, 2));

  await transcriber.dispose?.();
}

main().catch((err) => {
  console.error('[node-whisper-inspect] failed:', err);
  process.exitCode = 1;
});
