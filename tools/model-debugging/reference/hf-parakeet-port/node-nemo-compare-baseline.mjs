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

function normalizeText(text) {
  return String(text ?? '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1')
    .trim();
}

function joinChunkTexts(chunks) {
  return normalizeText((chunks ?? []).map((chunk) => chunk?.text ?? '').join(' '));
}

function toWordListFromChunks(chunks) {
  return (chunks ?? []).map((chunk) => ({
    text: chunk?.text ?? '',
    startTime: chunk?.timestamp?.[0] ?? null,
    endTime: chunk?.timestamp?.[1] ?? null,
  }));
}

function getWordStart(word) {
  return word?.startTime ?? word?.start_time ?? null;
}

function getWordEnd(word) {
  return word?.endTime ?? word?.end_time ?? null;
}

function compareWordTimings(referenceWords, actualWords) {
  const length = Math.min(referenceWords.length, actualWords.length);
  let maxStartDiff = 0;
  let maxEndDiff = 0;
  let over40ms = 0;
  let over80ms = 0;

  for (let i = 0; i < length; i++) {
    const ref = referenceWords[i];
    const act = actualWords[i];
    const startDiff = Math.abs((getWordStart(act) ?? 0) - (getWordStart(ref) ?? 0));
    const endDiff = Math.abs((getWordEnd(act) ?? 0) - (getWordEnd(ref) ?? 0));
    maxStartDiff = Math.max(maxStartDiff, startDiff);
    maxEndDiff = Math.max(maxEndDiff, endDiff);
    if (startDiff > 0.04 || endDiff > 0.04) over40ms += 1;
    if (startDiff > 0.08 || endDiff > 0.08) over80ms += 1;
  }

  return {
    compared_words: length,
    max_start_diff_s: +maxStartDiff.toFixed(3),
    max_end_diff_s: +maxEndDiff.toFixed(3),
    words_over_40ms: over40ms,
    words_over_80ms: over80ms,
  };
}

function findFirstWordMismatch(referenceWords, actualWords) {
  const maxLength = Math.max(referenceWords.length, actualWords.length);
  for (let i = 0; i < maxLength; i++) {
    const ref = referenceWords[i]?.text ?? null;
    const act = actualWords[i]?.text ?? null;
    if (ref !== act) {
      return {
        index: i,
        reference: ref,
        actual: act,
      };
    }
  }
  return null;
}

function findFirstTextMismatch(referenceText, actualText) {
  const length = Math.min(referenceText.length, actualText.length);
  for (let i = 0; i < length; i++) {
    if (referenceText[i] !== actualText[i]) {
      return {
        index: i,
        reference_snippet: referenceText.slice(Math.max(0, i - 40), i + 80),
        actual_snippet: actualText.slice(Math.max(0, i - 40), i + 80),
      };
    }
  }
  if (referenceText.length !== actualText.length) {
    const i = length;
    return {
      index: i,
      reference_snippet: referenceText.slice(Math.max(0, i - 40), i + 80),
      actual_snippet: actualText.slice(Math.max(0, i - 40), i + 80),
    };
  }
  return null;
}

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    model: 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4',
    audio: defaultAudioPath(),
    fixture: path.resolve(process.cwd(), 'docs/fixtures/jfk-nemo-single-pass.json'),
    local: process.env.TRANSFORMERS_LOCAL !== 'false',
    localModule: path.resolve(process.cwd(), '../reference-js-runtime/src/reference-runtime.js'),
    encoderDevice: 'cpu',
    decoderDevice: 'cpu',
    chunkLengthS: 90,
    strideLengthS: 10,
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--model') out.model = args[++i];
    else if (a === '--audio') out.audio = path.resolve(process.cwd(), args[++i]);
    else if (a === '--fixture') out.fixture = path.resolve(process.cwd(), args[++i]);
    else if (a === '--local') out.local = true;
    else if (a === '--npm') out.local = false;
    else if (a === '--local-module') out.localModule = path.resolve(process.cwd(), args[++i]);
    else if (a === '--encoder-device') out.encoderDevice = args[++i];
    else if (a === '--decoder-device') out.decoderDevice = args[++i];
    else if (a === '--chunk-length-s') out.chunkLengthS = Number(args[++i]);
    else if (a === '--stride-length-s') out.strideLengthS = Number(args[++i]);
  }

  if (!out.audio) {
    throw new Error('Missing --audio <path-to-wav-file>. No default WAV was found at N:\\JFK.wav.');
  }
  return out;
}

async function main() {
  const opts = parseArgs();
  if (!fs.existsSync(opts.audio)) {
    throw new Error(`Audio file not found: ${opts.audio}`);
  }
  if (!fs.existsSync(opts.fixture)) {
    throw new Error(`Fixture file not found: ${opts.fixture}`);
  }

  const modulePath = opts.local ? opts.localModule : '@huggingface/transformers';
  if (opts.local && !fs.existsSync(modulePath)) {
    throw new Error(`Local reference module not found at: ${modulePath}`);
  }

  const importTarget = opts.local ? pathToFileURL(modulePath).href : modulePath;
  const { env, pipeline } = await import(importTarget);

  env.allowLocalModels = true;
  env.allowRemoteModels = true;

  const reference = JSON.parse(fs.readFileSync(opts.fixture, 'utf8'));
  const referenceText = normalizeText(reference.output?.text ?? '');
  const referenceWords = reference.output?.words ?? [];

  const transcriber = await pipeline('automatic-speech-recognition', opts.model, {
    device: {
      encoder_model: opts.encoderDevice,
      decoder_model_merged: opts.decoderDevice,
    },
  });

  const { audio, sampleRate } = loadWavAsFloat32(opts.audio);
  const targetRate = transcriber?.processor?.feature_extractor?.config?.sampling_rate ?? sampleRate;
  const audioForModel = resampleLinear(audio, sampleRate, targetRate);

  const runs = [
    { name: 'pipeline_plain', options: {} },
    { name: 'pipeline_segments', options: { return_timestamps: true } },
    { name: 'pipeline_words', options: { return_timestamps: 'word' } },
    {
      name: 'pipeline_words_windowed',
      options: {
        return_timestamps: 'word',
        chunk_length_s: opts.chunkLengthS,
        stride_length_s: opts.strideLengthS,
      },
    },
  ];

  const results = [];
  for (const run of runs) {
    const started = Date.now();
    const output = await transcriber(audioForModel, run.options);
    const elapsedS = Number(((Date.now() - started) / 1000).toFixed(2));
    const text = normalizeText(output?.text ?? '');
    const chunks = Array.isArray(output?.chunks) ? output.chunks : [];
    const chunkText = joinChunkTexts(chunks);
    const actualWords = run.options.return_timestamps === 'word' ? toWordListFromChunks(chunks) : [];

    results.push({
      name: run.name,
      elapsed_s: elapsedS,
      text_matches_reference: text === referenceText,
      first_text_mismatch: text === referenceText ? null : findFirstTextMismatch(referenceText, text),
      chunk_text_matches_reference: chunks.length > 0 ? chunkText === referenceText : null,
      text_length: text.length,
      chunk_count: chunks.length,
      chunk_text_length: chunkText.length,
      word_count: actualWords.length || null,
      word_texts_match_reference:
        actualWords.length > 0
          ? actualWords.every((word, i) => word.text === referenceWords[i]?.text) && actualWords.length === referenceWords.length
          : null,
      first_word_mismatch:
        actualWords.length > 0 && !(actualWords.every((word, i) => word.text === referenceWords[i]?.text) && actualWords.length === referenceWords.length)
          ? findFirstWordMismatch(referenceWords, actualWords)
          : null,
      timing_diff: actualWords.length > 0 ? compareWordTimings(referenceWords, actualWords) : null,
      first_chunks: chunks.slice(0, 3),
      last_chunks: chunks.slice(-3),
    });
  }

  await transcriber.dispose?.();

  console.log(JSON.stringify({
    model: opts.model,
    audio: {
      file: opts.audio,
      source_sample_rate: sampleRate,
      model_sample_rate: targetRate,
      source_seconds: +(audio.length / sampleRate).toFixed(3),
      model_seconds: +(audioForModel.length / targetRate).toFixed(3),
    },
    reference: {
      fixture: opts.fixture,
      text_length: referenceText.length,
      word_count: referenceWords.length,
      utteranceTimestamp: reference.output?.utteranceTimestamp ?? reference.output?.utterance_timestamp ?? null,
    },
    runs: results,
  }, null, 2));
}

main().catch((err) => {
  console.error('[node-nemo-compare-baseline] failed:', err);
  process.exitCode = 1;
});
