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

function defaultAudioPath() {
  const candidate = 'N:\\JFK.wav';
  return fs.existsSync(candidate) ? candidate : null;
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

function toFiniteNumber(value, label) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    throw new Error(`Expected a finite number for ${label}. Got: ${value}`);
  }
  return num;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    model: 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4',
    audio: defaultAudioPath(),
    api: 'pipeline',
    timestamps: 'none',
    chunkLengthS: null,
    strideLengthS: null,
    encoderDevice: 'cpu',
    decoderDevice: 'cpu',
    encoderDtype: null,
    decoderDtype: null,
    local: process.env.TRANSFORMERS_LOCAL !== 'false',
    localModule: path.resolve(process.cwd(), '../reference-js-runtime/dist/reference.node.mjs'),
    returnWords: false,
    returnTokens: false,
    returnMetrics: false,
    returnFrameConfidences: false,
    returnFrameIndices: false,
    returnLogProbs: false,
    returnTdtSteps: false,
    timeOffset: 0,
    output: null,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--model') out.model = args[++i];
    else if (a === '--audio') out.audio = path.resolve(process.cwd(), args[++i]);
    else if (a === '--api') out.api = args[++i];
    else if (a === '--timestamps') out.timestamps = args[++i];
    else if (a === '--chunk-length-s') out.chunkLengthS = toFiniteNumber(args[++i], '--chunk-length-s');
    else if (a === '--stride-length-s') out.strideLengthS = toFiniteNumber(args[++i], '--stride-length-s');
    else if (a === '--encoder-device') out.encoderDevice = args[++i];
    else if (a === '--decoder-device') out.decoderDevice = args[++i];
    else if (a === '--encoder-dtype') out.encoderDtype = args[++i];
    else if (a === '--decoder-dtype') out.decoderDtype = args[++i];
    else if (a === '--return-words') out.returnWords = true;
    else if (a === '--return-tokens') out.returnTokens = true;
    else if (a === '--return-metrics') out.returnMetrics = true;
    else if (a === '--return-frame-confidences') out.returnFrameConfidences = true;
    else if (a === '--return-frame-indices') out.returnFrameIndices = true;
    else if (a === '--return-log-probs') out.returnLogProbs = true;
    else if (a === '--return-tdt-steps') out.returnTdtSteps = true;
    else if (a === '--time-offset') out.timeOffset = toFiniteNumber(args[++i], '--time-offset');
    else if (a === '--output') out.output = path.resolve(process.cwd(), args[++i]);
    else if (a === '--local') out.local = true;
    else if (a === '--npm') out.local = false;
    else if (a === '--local-module') out.localModule = path.resolve(process.cwd(), args[++i]);
  }

  if (!out.audio) {
    throw new Error('Missing --audio <path-to-wav-file>. No default WAV was found at N:\\JFK.wav.');
  }
  if (!['pipeline', 'direct'].includes(out.api)) {
    throw new Error(`Invalid --api ${out.api}. Use one of: pipeline, direct.`);
  }
  if (!['none', 'segments', 'words'].includes(out.timestamps)) {
    throw new Error(`Invalid --timestamps ${out.timestamps}. Use one of: none, segments, words.`);
  }
  return out;
}

function makePipelineOptions(opts) {
  const runOpts = {};
  if (opts.timestamps === 'segments') {
    runOpts.return_timestamps = true;
  } else if (opts.timestamps === 'words') {
    runOpts.return_timestamps = 'word';
  }
  if (opts.chunkLengthS != null) {
    runOpts.chunk_length_s = opts.chunkLengthS;
  }
  if (opts.strideLengthS != null) {
    runOpts.stride_length_s = opts.strideLengthS;
  }
  return runOpts;
}

function makeDirectOptions(tokenizer, opts) {
  return {
    tokenizer,
    returnTimestamps: opts.timestamps !== 'none',
    returnWords: opts.returnWords || opts.timestamps === 'words',
    returnTokens: opts.returnTokens,
    returnMetrics: opts.returnMetrics,
    returnFrameConfidences: opts.returnFrameConfidences,
    returnFrameIndices: opts.returnFrameIndices,
    returnLogProbs: opts.returnLogProbs,
    returnTdtSteps: opts.returnTdtSteps,
    timeOffset: opts.timeOffset,
  };
}

async function disposeInputs(inputs) {
  const seen = new Set();
  for (const value of Object.values(inputs ?? {})) {
    if (value && typeof value.dispose === 'function' && !seen.has(value)) {
      value.dispose();
      seen.add(value);
    }
  }
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

  const pipelineOptions = {
    device: {
      encoder_model: opts.encoderDevice,
      decoder_model_merged: opts.decoderDevice,
    },
  };
  if (opts.encoderDtype || opts.decoderDtype) {
    pipelineOptions.dtype = {};
    if (opts.encoderDtype) pipelineOptions.dtype.encoder_model = opts.encoderDtype;
    if (opts.decoderDtype) pipelineOptions.dtype.decoder_model_merged = opts.decoderDtype;
  }

  const transcriber = await pipeline('automatic-speech-recognition', opts.model, pipelineOptions);

  const { audio, sampleRate } = loadWavAsFloat32(opts.audio);
  const targetRate = transcriber?.processor?.feature_extractor?.config?.sampling_rate ?? sampleRate;
  const audioForModel = resampleLinear(audio, sampleRate, targetRate);

  const started = Date.now();
  let output;
  let effectiveOptions;
  if (opts.api === 'pipeline') {
    effectiveOptions = makePipelineOptions(opts);
    output = await transcriber(audioForModel, effectiveOptions);
  } else {
    effectiveOptions = makeDirectOptions(transcriber.tokenizer, opts);
    const inputs = await transcriber.processor(audioForModel);
    try {
      output = await transcriber.model.transcribe(inputs, effectiveOptions);
    } finally {
      await disposeInputs(inputs);
    }
  }
  const elapsedS = Number(((Date.now() - started) / 1000).toFixed(2));

  const payload = {
    model: opts.model,
    api: opts.api,
    audio: {
      file: opts.audio,
      source_sample_rate: sampleRate,
      model_sample_rate: targetRate,
      source_seconds: +(audio.length / sampleRate).toFixed(3),
      model_seconds: +(audioForModel.length / targetRate).toFixed(3),
    },
    load: {
      transformers_source: opts.local ? modulePath : 'npm:@huggingface/transformers',
      device: pipelineOptions.device,
      dtype: pipelineOptions.dtype ?? null,
      model_type: transcriber?.model?.config?.model_type ?? null,
    },
    options: effectiveOptions,
    elapsed_s: elapsedS,
    output,
  };

  const encoded = JSON.stringify(payload, null, 2);
  if (opts.output) {
    fs.mkdirSync(path.dirname(opts.output), { recursive: true });
    fs.writeFileSync(opts.output, encoded, 'utf8');
  } else {
    console.log(encoded);
  }

  await transcriber.dispose?.();
}

main().catch((err) => {
  console.error('[node-nemo-inspect] failed:', err);
  process.exitCode = 1;
});
