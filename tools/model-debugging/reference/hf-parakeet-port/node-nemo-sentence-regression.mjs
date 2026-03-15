import fs from 'fs';
import os from 'os';
import path from 'path';
import process from 'process';
import { spawnSync } from 'child_process';
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

function toFiniteNumber(value, label) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    throw new Error(`Expected a finite number for ${label}. Got: ${value}`);
  }
  return num;
}

function normalizeText(text) {
  return String(text ?? '')
    .normalize('NFKC')
    .replace(/\r\n/g, '\n')
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[_*]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

function tokenizeWords(text) {
  return normalizeText(text)
    .split(/[^a-z0-9]+/i)
    .map((x) => x.trim())
    .filter(Boolean);
}

function computeWordOverlap(referenceText, hypothesisText) {
  const ref = tokenizeWords(referenceText);
  const hyp = tokenizeWords(hypothesisText);
  let matchedPrefix = 0;
  const limit = Math.min(ref.length, hyp.length);
  while (matchedPrefix < limit && ref[matchedPrefix] === hyp[matchedPrefix]) {
    matchedPrefix += 1;
  }
  return {
    reference_words: ref.length,
    hypothesis_words: hyp.length,
    matched_prefix_words: matchedPrefix,
    matched_prefix_ratio: ref.length > 0 ? Number((matchedPrefix / ref.length).toFixed(4)) : 0,
  };
}

function normalizeSentenceList(items) {
  if (!Array.isArray(items)) {
    return [];
  }
  return items
    .map((item) => normalizeText(item))
    .filter(Boolean);
}

function computeSentenceAlignment(expectedSentences, actualChunks) {
  const expected = normalizeSentenceList(expectedSentences);
  const actual = normalizeSentenceList(
    Array.isArray(actualChunks)
      ? actualChunks.map((chunk) => chunk?.text ?? '')
      : [],
  );

  let matchedPrefix = 0;
  const limit = Math.min(expected.length, actual.length);
  while (matchedPrefix < limit && expected[matchedPrefix] === actual[matchedPrefix]) {
    matchedPrefix += 1;
  }

  let exactPositionMatches = 0;
  for (let i = 0; i < limit; i++) {
    if (expected[i] === actual[i]) {
      exactPositionMatches += 1;
    }
  }

  return {
    expected_sentences: expected.length,
    actual_sentences: actual.length,
    exact_position_matches: exactPositionMatches,
    exact_position_match_ratio: expected.length > 0 ? Number((exactPositionMatches / expected.length).toFixed(4)) : 0,
    matched_prefix_sentences: matchedPrefix,
    matched_prefix_ratio: expected.length > 0 ? Number((matchedPrefix / expected.length).toFixed(4)) : 0,
  };
}

function joinedChunkText(output) {
  return Array.isArray(output?.chunks)
    ? output.chunks.map((chunk) => String(chunk?.text ?? '').trim()).filter(Boolean).join(' ').replace(/\s+/g, ' ').trim()
    : '';
}

function hasMonotonicTimestamps(chunks, maxOverlapS = 0) {
  if (!Array.isArray(chunks)) return true;
  let lastEnd = -Infinity;
  for (const chunk of chunks) {
    const timestamp = chunk?.timestamp;
    if (!Array.isArray(timestamp) || timestamp.length !== 2) {
      return false;
    }
    const [start, end] = timestamp;
    if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start < lastEnd - maxOverlapS) {
      return false;
    }
    lastEnd = Math.max(lastEnd, end);
  }
  return true;
}

function assertOutputShape(label, output, maxOverlapS = 0) {
  if (!output || typeof output.text !== 'string' || !output.text.trim()) {
    throw new Error(`${label}: expected a non-empty text output.`);
  }
  if (!Array.isArray(output.chunks) || output.chunks.length === 0) {
    throw new Error(`${label}: expected non-empty chunks.`);
  }
  if (!hasMonotonicTimestamps(output.chunks, maxOverlapS)) {
    throw new Error(`${label}: chunk timestamps are not monotonic.`);
  }
  const joined = normalizeText(joinedChunkText(output));
  const whole = normalizeText(output.text);
  if (joined !== whole) {
    throw new Error(`${label}: joined chunk text does not match full text.`);
  }
}

function collectOutputChecks(output, referenceText, maxOverlapS = 0) {
  const joinedMatches = normalizeText(joinedChunkText(output)) === normalizeText(output.text);
  const monotonic = hasMonotonicTimestamps(output?.chunks, maxOverlapS);
  return {
    chunks: Array.isArray(output?.chunks) ? output.chunks.length : 0,
    joined_chunk_text_matches_text: joinedMatches,
    monotonic_timestamps: monotonic,
    overlap_vs_reference: computeWordOverlap(referenceText, output?.text ?? ''),
  };
}

function convertAudioToTempWav(inputPath) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nemo-sentence-regression-'));
  const wavPath = path.join(tmpDir, `${path.parse(inputPath).name}.wav`);
  const ffmpeg = spawnSync('ffmpeg', [
    '-y',
    '-i', inputPath,
    '-ac', '1',
    wavPath,
  ], { encoding: 'utf8' });

  if (ffmpeg.status !== 0) {
    throw new Error(`ffmpeg failed for ${inputPath}\n${ffmpeg.stderr || ffmpeg.stdout}`);
  }

  return { tmpDir, wavPath };
}

function cleanupTempDir(tmpDir) {
  if (tmpDir && fs.existsSync(tmpDir)) {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

function parseArgs() {
  const fixtureBase = path.resolve(
    process.cwd(),
    'tools/data/fixtures/audio/scientistsdoscienceinspaceedreadsshortscifivolvii_01_01',
  );
  const args = process.argv.slice(2);
  const out = {
    model: 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4',
    audio: `${fixtureBase}.mp3`,
    transcript: `${fixtureBase}.txt`,
    sentences: fs.existsSync(`${fixtureBase}.json`) ? `${fixtureBase}.json` : null,
    output: path.resolve(process.cwd(), 'tools/data/results/nemo-tdt/scientists-nemo-sentence-regression.json'),
    chunkLengthS: 90,
    encoderDevice: 'cpu',
    decoderDevice: 'cpu',
    encoderDtype: null,
    decoderDtype: null,
    local: process.env.TRANSFORMERS_LOCAL !== 'false',
    localModule: path.resolve(process.cwd(), '../reference-js-runtime/dist/reference.node.mjs'),
  };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--model') out.model = args[++i];
    else if (a === '--audio') out.audio = path.resolve(process.cwd(), args[++i]);
    else if (a === '--transcript') out.transcript = path.resolve(process.cwd(), args[++i]);
    else if (a === '--sentences') out.sentences = path.resolve(process.cwd(), args[++i]);
    else if (a === '--output') out.output = path.resolve(process.cwd(), args[++i]);
    else if (a === '--chunk-length-s') out.chunkLengthS = toFiniteNumber(args[++i], '--chunk-length-s');
    else if (a === '--encoder-device') out.encoderDevice = args[++i];
    else if (a === '--decoder-device') out.decoderDevice = args[++i];
    else if (a === '--encoder-dtype') out.encoderDtype = args[++i];
    else if (a === '--decoder-dtype') out.decoderDtype = args[++i];
    else if (a === '--local') out.local = true;
    else if (a === '--npm') out.local = false;
    else if (a === '--local-module') out.localModule = path.resolve(process.cwd(), args[++i]);
  }

  return out;
}

async function main() {
  const opts = parseArgs();
  const modulePath = opts.local ? opts.localModule : '@huggingface/transformers';

  if (!fs.existsSync(opts.audio)) {
    throw new Error(`Audio file not found: ${opts.audio}`);
  }
    if (!fs.existsSync(opts.transcript)) {
      throw new Error(`Transcript file not found: ${opts.transcript}`);
    }
    if (opts.sentences && !fs.existsSync(opts.sentences)) {
      throw new Error(`Sentence reference file not found: ${opts.sentences}`);
    }
  if (opts.local && !fs.existsSync(modulePath)) {
    throw new Error(
      `Local reference module not found at: ${modulePath}\n` +
      'Build your local JS reference runtime first and point --local-module at it.',
    );
  }

  const { tmpDir, wavPath } = convertAudioToTempWav(opts.audio);
  try {
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
    const { audio, sampleRate } = loadWavAsFloat32(wavPath);
    const targetRate = transcriber?.processor?.feature_extractor?.config?.sampling_rate ?? sampleRate;
    const audioForModel = resampleLinear(audio, sampleRate, targetRate);

    const segmentOutput = await transcriber(audioForModel, {
      return_timestamps: true,
      chunk_length_s: opts.chunkLengthS,
    });
    const wordOutput = await transcriber(audioForModel, {
      return_timestamps: 'word',
      chunk_length_s: opts.chunkLengthS,
    });

    const referenceText = fs.readFileSync(opts.transcript, 'utf8');
    const referenceSentences = opts.sentences ? JSON.parse(fs.readFileSync(opts.sentences, 'utf8')) : null;
    const segmentChecks = collectOutputChecks(segmentOutput, referenceText, 0);
    const wordChecks = collectOutputChecks(wordOutput, referenceText, 0.1);
    const report = {
      model: opts.model,
      generated_at: new Date().toISOString(),
      audio: {
        source_file: opts.audio,
        wav_file: wavPath,
        source_sample_rate: sampleRate,
        model_sample_rate: targetRate,
        seconds: Number((audioForModel.length / targetRate).toFixed(3)),
      },
      transcript: {
        file: opts.transcript,
        characters: referenceText.length,
        words: tokenizeWords(referenceText).length,
      },
      sentences: {
        file: opts.sentences,
        count: Array.isArray(referenceSentences) ? referenceSentences.length : 0,
      },
      options: {
        chunk_length_s: opts.chunkLengthS,
        device: pipelineOptions.device,
        dtype: pipelineOptions.dtype ?? null,
        transformers_source: opts.local ? modulePath : 'npm:@huggingface/transformers',
      },
      checks: {
        segments: {
          ...segmentChecks,
          sentence_alignment: referenceSentences
            ? computeSentenceAlignment(referenceSentences, segmentOutput.chunks)
            : null,
        },
        words: wordChecks,
      },
      outputs: {
        segments: segmentOutput,
        words: wordOutput,
      },
    };

    fs.mkdirSync(path.dirname(opts.output), { recursive: true });
    fs.writeFileSync(opts.output, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({
      ok: true,
      output: opts.output,
      segment_chunks: report.checks.segments.chunks,
      word_chunks: report.checks.words.chunks,
      segment_prefix_ratio: report.checks.segments.overlap_vs_reference.matched_prefix_ratio,
      segment_sentence_prefix_ratio: report.checks.segments.sentence_alignment?.matched_prefix_ratio ?? null,
      segment_sentence_exact_ratio: report.checks.segments.sentence_alignment?.exact_position_match_ratio ?? null,
      word_prefix_ratio: report.checks.words.overlap_vs_reference.matched_prefix_ratio,
    }, null, 2));

    assertOutputShape('segments', segmentOutput, 0);
    assertOutputShape('words', wordOutput, 0.1);

    await transcriber.dispose?.();
  } finally {
    cleanupTempDir(tmpDir);
  }
}

main().catch((err) => {
  console.error('[node-nemo-sentence-regression] failed:', err);
  process.exitCode = 1;
});
