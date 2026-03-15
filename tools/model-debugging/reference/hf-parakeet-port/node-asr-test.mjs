import fs from 'fs';
import path from 'path';
import process from 'process';
import { performance } from 'perf_hooks';
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
  const defaultWav = path.resolve(process.cwd(), 'public/assets/life_Jim.wav');
  const out = {
    model: 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4',
    audio: defaultWav,
    encoderDevice: 'webgpu',
    encoderDtype: 'fp16',
    decoderDtype: 'int8',
    timestamps: false,
    local: process.env.TRANSFORMERS_LOCAL !== 'false',
    localModule: path.resolve(process.cwd(), '../reference-js-runtime/dist/reference.node.mjs'),
    loop: 1,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--model') out.model = args[++i];
    else if (a === '--audio') out.audio = path.resolve(process.cwd(), args[++i]);
    else if (a === '--encoder-device') out.encoderDevice = args[++i];
    else if (a === '--encoder-dtype') out.encoderDtype = args[++i];
    else if (a === '--decoder-dtype') out.decoderDtype = args[++i];
    else if (a === '--timestamps') out.timestamps = true;
    else if (a === '--local') out.local = true;
    else if (a === '--npm') out.local = false;
    else if (a === '--local-module') out.localModule = path.resolve(process.cwd(), args[++i]);
    else if (a === '--loop') out.loop = Math.max(1, Number(args[++i]) || 1);
  }
  return out;
}

async function main() {
  const opts = parseArgs();
  const modulePath = opts.local ? opts.localModule : '@huggingface/transformers';

  if (!fs.existsSync(opts.audio)) {
    throw new Error(
      `Audio file not found: ${opts.audio}\n` +
      `Node CLI supports WAV input only. Pass --audio <path-to-wav-file>.`
    );
  }

  const audioExt = path.extname(opts.audio).toLowerCase();

  if (audioExt !== '.wav') {
    throw new Error(
      `Node CLI currently supports WAV input only. Got: ${opts.audio}\n` +
      `Use --audio <path-to-wav-file> or run the browser demo sample (OGG is supported in browser).`
    );
  }

  if (opts.local && !fs.existsSync(modulePath)) {
    throw new Error(
      `Local reference module not found at: ${modulePath}\n` +
      `Build your local JS reference runtime first and point --local-module at it.`,
    );
  }

  const importTarget = opts.local ? pathToFileURL(modulePath).href : modulePath;
  const mod = await import(importTarget);
  const { env, pipeline } = mod;

  env.allowLocalModels = true;
  env.allowRemoteModels = true;

  const decoderDevice = 'cpu'; // Node runtime uses CPU for decoder (WASM is browser-only).

  console.log('[node-asr] opts:', {
    ...opts,
    decoderDevice,
    transformersSource: opts.local ? modulePath : 'npm:@huggingface/transformers',
    importTarget,
  });

  const t0 = performance.now();
  const transcriber = await pipeline('automatic-speech-recognition', opts.model, {
    device: {
      encoder_model: opts.encoderDevice,
      decoder_model_merged: decoderDevice,
    },
    dtype: {
      encoder_model: opts.encoderDtype,
      decoder_model_merged: opts.decoderDtype,
    },
    session_options: {
      logId: `node-asr-${Date.now()}`,
      logSeverityLevel: 2,
      graphOptimizationLevel: 'disabled',
      enableMemPattern: false,
      enableCpuMemArena: false,
      freeDimensionOverrides: { batch_size: 1 },
    },
  });
  const tLoad = performance.now();

  const { audio, sampleRate } = loadWavAsFloat32(opts.audio);
  const expectedSampleRate = transcriber?.processor?.feature_extractor?.config?.sampling_rate ?? sampleRate;
  const audioForModel = resampleLinear(audio, sampleRate, expectedSampleRate);
  console.log('[node-asr] wav:', {
    file: opts.audio,
    sampleRate,
    modelSampleRate: expectedSampleRate,
    samples: audio.length,
    seconds: +(audio.length / sampleRate).toFixed(3),
    modelSamples: audioForModel.length,
    modelSeconds: +(audioForModel.length / expectedSampleRate).toFixed(3),
  });

  for (let run = 1; run <= opts.loop; run++) {
    const runStart = performance.now();
    const output = await transcriber(audioForModel, opts.timestamps ? { return_timestamps: true } : {});
    const runDone = performance.now();
    console.log(`[node-asr] run ${run}/${opts.loop}:`, {
      ms: Math.round(runDone - runStart),
      text: output?.text ?? '',
      chunks: Array.isArray(output?.chunks) ? output.chunks.length : 0,
    });
  }
  const tDone = performance.now();

  console.log('[node-asr] model_type:', transcriber?.model?.config?.model_type);
  console.log('[node-asr] sessions:', Object.keys(transcriber?.model?.sessions ?? {}));
  console.log('[node-asr] timing_ms:', {
    load: Math.round(tLoad - t0),
    run: Math.round(tDone - tLoad),
    total: Math.round(tDone - t0),
  });

  await transcriber.dispose?.();
}

main().catch((err) => {
  console.error('[node-asr] failed:', err);
  process.exitCode = 1;
});
