#!/usr/bin/env node
/**
 * WhisperX-compatible transcription runner with full quality gates.
 *
 * Accepts the same CLI parameters as WhisperX:
 *   --model, --language, --vad_onset, --beam_size, --temperature,
 *   --compression_ratio_threshold, --logprob_threshold,
 *   --no_speech_threshold, --entropy_threshold, etc.
 *
 * Exports runAsrPipeline() for programmatic use (smoke tests).
 */

import path from 'node:path';
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');

// ──────────────────────────────────────────────────────────
// CLI argument parsing
// ──────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const usedIndices = new Set();

const BOOLEAN_FLAGS = new Set([
  'verbose', 'noalign', 'suppressnumerals', 'conditiononprevioustext',
  'diarize', 'speakerembeddings', 'highlightwords', 'returncharalignments',
  'printprogress', 'modelcacheonly', 'fp16',
]);

const DEFAULT_OPTS = {
  model: process.env.WHISPER_MODEL_DIR || '/tmp/whisper-base-4graph/fp32',
  device: 'cpu', batchSize: 1,
  vadBackendType: 'ten-vad', vadOnset: 0.5, vadOffset: 0.363, chunkSize: 30,
  temperature: 0, temperatureIncrement: 0.2,
  bestOf: 1, beamSize: 1, patience: 1.0, lengthPenalty: 1.0,
  suppressNumerals: false,
  compressionRatioThreshold: 2.4, logprobThreshold: -1.0,
  noSpeechThreshold: 0.6, entropyThreshold: 2.4,
  initialPrompt: null, hotwords: null, conditionOnPreviousText: false,
  task: 'transcribe', language: null,
  noAlign: false,
  outputFormat: 'vtt',
  verbose: true,
};

function parseArgs(customArgs) {
  const opts = { ...DEFAULT_OPTS };
  const parsed = customArgs ?? args;
  const localUsed = new Set();

  for (let i = 0; i < parsed.length; i++) {
    const a = parsed[i];
    if (a.startsWith('--') && !localUsed.has(i)) {
      localUsed.add(i);
      const key = a.slice(2).replace(/-/g, '');
      const isBool = BOOLEAN_FLAGS.has(key);
      const val = !isBool && i + 1 < parsed.length && !parsed[i + 1].startsWith('--')
        ? parsed[++i] : true;
      localUsed.add(i);

      const mapping = {
        model: 'model', 'model-dir': 'modelDir',
        device: 'device', vadmethod: 'vadBackendType',
        vadonset: 'vadOnset', vadoffset: 'vadOffset', chunksize: 'chunkSize',
        temperature: 'temperature', temperatureincrementonfallback: 'temperatureIncrement',
        bestof: 'bestOf', beamsize: 'beamSize',
        patience: 'patience', lengthpenalty: 'lengthPenalty',
        compressionratiothreshold: 'compressionRatioThreshold',
        logprobthreshold: 'logprobThreshold', nospeechthreshold: 'noSpeechThreshold',
        entropythreshold: 'entropyThreshold',
        initialprompt: 'initialPrompt', hotwords: 'hotwords',
        task: 'task', language: 'language',
        noalign: 'noAlign', outputformat: 'outputFormat',
        verbose: 'verbose', batchsize: 'batchSize',
      };

      const optKey = mapping[key];
      if (optKey) {
        if (['noAlign', 'suppressNumerals', 'conditionOnPreviousText', 'verbose'].includes(optKey)) {
          opts[optKey] = val === true || val === 'true' || val === '1';
        } else if (typeof opts[optKey] === 'number') {
          opts[optKey] = Number(val);
        } else {
          opts[optKey] = String(val);
        }
      }
    }
  }

  const audioFiles = parsed.filter((a, i) => !a.startsWith('--') && !localUsed.has(i));
  return { opts, audioPath: audioFiles[0] };
}

// ──────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────
function log(msg, verbose) {
  if (verbose) console.log(msg);
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(7, '0')}`;
}

function toVtt(segments) {
  const lines = ['WEBVTT', 'Kind: captions', 'Language: tr', ''];
  for (const seg of segments) {
    lines.push(`${formatTime(seg.start)} --> ${formatTime(seg.end)}`);
    lines.push(seg.text);
    lines.push('');
  }
  return lines.join('\n');
}

// ──────────────────────────────────────────────────────────
// Temperature sampling
// ──────────────────────────────────────────────────────────

/** Greedy argmax for temperature=0. */
function argmax(arr) {
  let idx = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[idx]) idx = i;
  }
  return idx;
}

/** Temperature-scaled sampling for temperature > 0. */
function sample(logits, temperature) {
  // Apply temperature scaling
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }
  // Softmax
  const maxVal = Math.max(...scaled);
  const exps = new Float32Array(scaled.length);
  let sumExp = 0;
  for (let i = 0; i < scaled.length; i++) {
    const e = Math.exp(scaled[i] - maxVal);
    exps[i] = e;
    sumExp += e;
  }
  // Sample from distribution
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < exps.length; i++) {
    cum += exps[i] / sumExp;
    if (r < cum) return i;
  }
  return exps.length - 1;
}

function nextToken(logits, temperature) {
  return temperature <= 0 ? argmax(logits) : sample(logits, temperature);
}

// ──────────────────────────────────────────────────────────
// Core ASR pipeline
// ──────────────────────────────────────────────────────────

/**
 * Run the full ASR pipeline: audio → VAD → decode (with quality gates + fallback).
 *
 * @param {object} opts
 * @param {string} opts.model           Path to ONNX model directory
 * @param {string} [opts.language]      Language code or null for auto-detect
 * @param {number} [opts.vadOnset]      VAD speech onset threshold
 * @param {number} [opts.vadOffset]     VAD speech offset threshold
 * @param {number} [opts.compressionRatioThreshold]  Default 2.4
 * @param {number} [opts.logprobThreshold]           Default -1.0
 * @param {number} [opts.noSpeechThreshold]          Default 0.6
 * @param {number} [opts.entropyThreshold]           Default 2.4
 * @param {number} [opts.temperature]                Starting temperature (default 0)
 * @param {number} [opts.temperatureIncrement]       Fallback increment (default 0.2)
 * @param {number} [opts.bestOf]        Best-of N sampling (not implemented in temp fallback yet)
 * @param {number} [opts.beamSize]      Beam search size (not implemented yet)
 * @param {boolean} [opts.conditionOnPreviousText]  Context conditioning
 * @param {string} [opts.initialPrompt] Initial prompt text
 * @param {number} [opts.chunkSize]     Max segment duration in seconds
 * @param {boolean} [opts.verbose]      Print progress
 * @param {string} opts.audioPath       Path to audio file
 * @returns {Promise<object>} { segments, fullText, wordCount, asrTime, vttContent,
 *                              temperaturesUsed, fallbackCount, gateResults }
 */
export async function runAsrPipeline(_opts) {
  // Merge with defaults
  const opts = { ...DEFAULT_OPTS, ..._opts };
  const { audioPath } = opts;
  const verbose = opts.verbose !== false;

  log(`Audio: ${audioPath}`, verbose);
  log(`Model: ${opts.model}`, verbose);
  log(`Language: ${opts.language || 'auto-detect'}`, verbose);
  log(`VAD: ${opts.vadBackendType} (onset=${opts.vadOnset}, offset=${opts.vadOffset})`, verbose);
  log(`Beam: ${opts.beamSize}, BestOf: ${opts.bestOf}`, verbose);
  log('', verbose);

  // ── 1. Decode audio to 16kHz mono WAV ──
  log('Decoding audio...', verbose);
  const tmpWav = path.join(REPO_ROOT, 'tmp/_whisperx_input_16k.wav');
  fs.mkdirSync(path.join(REPO_ROOT, 'tmp'), { recursive: true });
  const ext = path.extname(audioPath).toLowerCase();
  if (ext === '.wav') {
    fs.copyFileSync(audioPath, tmpWav);
  } else {
    execSync(`ffmpeg -y -i "${audioPath}" -ar 16000 -ac 1 -sample_fmt s16 -f wav "${tmpWav}" 2>/dev/null`, { stdio: 'ignore' });
  }

  const buf = fs.readFileSync(tmpWav);
  const frameCount = Math.floor((buf.length - 44) / 2);
  const pcm = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) pcm[i] = buf.readInt16LE(44 + i * 2) / 32768;
  const sampleRate = 16000;
  const audioDuration = frameCount / sampleRate;
  log(`  ${frameCount} frames @ ${sampleRate}Hz = ${audioDuration.toFixed(1)}s`, verbose);

  // ── 2. Load model ──
  log('Loading ONNX model...', verbose);
  const tLoad = performance.now();
  const ort = await import('onnxruntime-node');
  const dist = REPO_ROOT + '/dist';

  const { WhisperTokenizer, fetchText } = await import(
    path.join(dist, 'models/whisper-seq2seq/index.js')
  );
  const { WhisperMelProcessor } = await import(
    path.join(dist, 'audio/whisper-mel.js')
  );
  const { WhisperTimestampLogitProcessor } = await import(
    path.join(dist, 'models/whisper-seq2seq/processors.js')
  );
  const { parseWhisperGenerationConfig, parseWhisperModelConfig } = await import(
    path.join(dist, 'models/whisper-seq2seq/generation-config.js')
  );

  const tokenizer = await WhisperTokenizer.fromUrl(path.join(opts.model, 'tokenizer.json'));
  const language = opts.language || 'en';
  const langId = tokenizer.getTokenId(`<|${language}|>`) ?? tokenizer.getTokenId('<|en|>') ?? 50259;

  const genConfig = parseWhisperGenerationConfig(
    JSON.parse(await fetchText(path.join(opts.model, 'generation_config.json')))
  );
  const configRaw = JSON.parse(await fetchText(path.join(opts.model, 'config.json')));
  const modelConfig = parseWhisperModelConfig(configRaw);
  const melBins = modelConfig.numMelBins ?? 80;
  const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;

  const encSess = await ort.InferenceSession.create(path.join(opts.model, 'encoder_model.onnx'));
  const initSess = await ort.InferenceSession.create(path.join(opts.model, 'decoder_init.onnx'));
  const stepSess = await ort.InferenceSession.create(path.join(opts.model, 'decoder_step.onnx'));
  log(`  Loaded in ${((performance.now() - tLoad) / 1000).toFixed(1)}s`, verbose);

  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const timestampProc = new WhisperTimestampLogitProcessor(tokenizer, genConfig);

  // ── 3. VAD segmentation ──
  log('Running VAD...', verbose);
  const tVad = performance.now();
  const { TenVadBackend } = await import(path.join(dist, 'chunking/backends/ten-vad.js'));
  const { segmentAudio } = await import(path.join(dist, 'chunking/vad-segmenter.js'));

  const tenVad = await TenVadBackend.create({
    threshold: opts.vadOnset, hopSize: 512,
    minSpeechDurationMs: 250, minSilenceDurationMs: 100,
  });

  const segments = await segmentAudio(pcm, {
    vad: tenVad, sampleRate, threshold: opts.vadOnset, noiseGate: false,
    merge: {
      minSilenceDurationMs: 100, speechPadMs: 400,
      maxSegmentDurationMs: opts.chunkSize * 1000,
      minSpeechDurationMs: 250,
    },
  });
  log(`  ${segments.length} VAD segments in ${((performance.now() - tVad) / 1000).toFixed(2)}s`, verbose);
  for (const seg of segments.slice(0, 5)) {
    log(`    [${seg.startSeconds.toFixed(1)}-${seg.endSeconds.toFixed(1)}] ${seg.durationSeconds.toFixed(1)}s`, verbose);
  }
  if (segments.length > 5) log(`    ... and ${segments.length - 5} more`, verbose);

  // ── 4. Quality gates ──
  const { compressionRatioGate, logProbGate, noSpeechGate, entropyGate } = await import(
    path.join(dist, 'quality/index.js')
  );
  const { withTemperatureFallback } = await import(
    path.join(dist, 'quality/temperature-fallback.js')
  );

  const gates = [
    compressionRatioGate(opts.compressionRatioThreshold ?? 2.4),
    logProbGate(opts.logprobThreshold ?? -1.0),
    entropyGate(opts.entropyThreshold ?? 2.4),
    noSpeechGate(opts.noSpeechThreshold ?? 0.6, opts.logprobThreshold ?? -1.0),
  ];

  // Build temperature progression
  const inc = opts.temperatureIncrement ?? 0.2;
  const startTemp = opts.temperature ?? 0;
  const temperatures = [startTemp];
  if (inc > 0) {
    for (let t = startTemp + inc; t <= 1.0 + 1e-6; t += inc) {
      temperatures.push(Math.round(t * 1000) / 1000);
    }
  }

  // ── 5. Transcribe each segment with quality gates + fallback ──
  log('Transcribing with quality gates...', verbose);
  const tAsr = performance.now();
  const allSegments = [];
  const gateResultsAll = [];
  const temperaturesUsed = new Set();
  let fallbackCount = 0;

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    const startSample = Math.max(0, Math.floor(seg.startSeconds * sampleRate));
    const endSample = Math.min(pcm.length, Math.ceil(seg.endSeconds * sampleRate));
    const chunk = pcm.slice(startSample, endSample);
    if (chunk.length < 1600) continue;

    // Mel features (shared across fallback attempts)
    const mel = WhisperMelProcessor.padToFrames(melProc.process(chunk), 3000);

    // Encoder run (shared — no need to re-encode on fallback)
    const encOut = await encSess.run({
      input_features: new ort.Tensor('float32', mel, [1, melBins, 3000]),
    });
    const encKey = Object.keys(encOut).find(k =>
      k === 'last_hidden_state' || k.includes('hidden') || k.includes('output')
    ) ?? Object.keys(encOut)[0];
    const encHs = new Float32Array(encOut[encKey].data);
    const dModel = modelConfig.dModel ?? 384;
    const encDims = [1, encHs.length / dModel, dModel];
    const encTensor = new ort.Tensor('float32', encHs, encDims);

    // Transcribe function (returns TranscribeAttempt)
    const transcribeFn = async (temperature) => {
      // Prompt tokens
      const promptTokens = [
        tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
        langId,
        tokenizer.getTokenId('<|transcribe|>') ?? 50359,
        tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
      ];
      if (opts.initialPrompt) {
        const encoded = tokenizer.encode(opts.initialPrompt);
        if (encoded) promptTokens.push(...encoded);
      }
      const promptLen = promptTokens.length;

      // Decoder init
      const initFeeds = {
        input_ids: new ort.Tensor('int64', BigInt64Array.from(promptTokens.map(BigInt)), [1, promptTokens.length]),
        encoder_hidden_states: encTensor,
      };
      const initOut = await initSess.run(initFeeds);

      // Logits output name
      const logitsKey = Object.keys(initOut).find(k => k.startsWith('logits')) ?? Object.keys(initOut)[0];
      const initLogitsData = initOut[logitsKey];
      const vSize = initLogitsData.dims[initLogitsData.dims.length - 1];

      // First token
      const lastOffset = initLogitsData.data.length - vSize;
      const firstLogits = initLogitsData.data.subarray(lastOffset, lastOffset + vSize);
      timestampProc.process(firstLogits, promptTokens, promptLen);
      const firstToken = nextToken(firstLogits, temperature);
      const tokens = [firstToken];
      const stepLogits = [new Float32Array(firstLogits)];

      // Build KV cache from init outputs
      const kvKeys = Object.keys(initOut).filter(k => k.startsWith('present'));
      let pastKv = {};
      const kvDims = {};
      for (const k of kvKeys) {
        const d = initOut[k].dims;
        if (d) {
          kvDims[k] = d;
          kvDims[k.replace(/^present\./, 'past_key_values.')] = d;
        }
        pastKv[k] = new Float32Array(initOut[k].data);
      }

      // Decode loop (greedy or temperature-sampled)
      // maxLength from config is TOTAL length (prompt + generated), so
      // max new tokens = maxLength - promptLen - 1 (for the init output)
      const maxNewTokens = (genConfig.maxLength ?? 448) - promptTokens.length - 1;
      for (let step = 1; step < maxNewTokens; step++) {
        const feeds = {
          input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(tokens[tokens.length - 1])]), [1, 1]),
        };
        for (const [k, v] of Object.entries(pastKv)) {
          const stepKey = k.replace(/^present\./, 'past_key_values.');
          const dims = kvDims[k] ?? kvDims[stepKey] ?? kvDims[k.replace(/^past_key_values\./, 'present.')];
          if (dims) {
            feeds[stepKey] = new ort.Tensor('float32', v, dims);
          }
        }
        const stepOut = await stepSess.run(feeds);
        const stepLogitsKey = Object.keys(stepOut).find(k => k.startsWith('logits'));
        const sl = new Float32Array(stepOut[stepLogitsKey].data);
        const stepVSize = stepOut[stepLogitsKey].dims[stepOut[stepLogitsKey].dims.length - 1];
        const slLast = sl.subarray(sl.length - stepVSize, sl.length);
        timestampProc.process(slLast, [...promptTokens, ...tokens], promptLen);
        const nextTok = nextToken(slLast, temperature);
        tokens.push(nextTok);
        stepLogits.push(new Float32Array(slLast));

        // Update KV
        const stepKvKeys = Object.keys(stepOut).filter(k => k.startsWith('present'));
        const newKv = {};
        for (const [k, v] of Object.entries(pastKv)) {
          if (k.includes('.encoder.')) newKv[k] = v;
        }
        for (const k of stepKvKeys) {
          newKv[k] = new Float32Array(stepOut[k].data);
          kvDims[k] = stepOut[k].dims;
          kvDims[k.replace(/^present\./, 'past_key_values.')] = stepOut[k].dims;
        }
        pastKv = newKv;

        if (nextTok === eosId) break;
      }

      // Decode tokens to text
      const text = tokens
        .map(t => {
          try { return tokenizer.decode([t]); } catch { return ''; }
        })
        .filter(t => t && !t.startsWith('<|') && !t.startsWith('['))
        .join('')
        .trim();

      return {
        result: { text, start: seg.startSeconds, end: seg.endSeconds },
        text,
        tokens,
        logits: stepLogits,
        vocabSize: vSize,
      };
    };

    // Run with temperature fallback
    const fbResult = await withTemperatureFallback(transcribeFn, gates, temperatures);
    temperaturesUsed.add(fbResult.temperature);
    if (fbResult.temperature > temperatures[0]) fallbackCount++;
    gateResultsAll.push(fbResult.gateResults);

    const segResult = fbResult.result;
    if (segResult.text) {
      allSegments.push(segResult);
      if (verbose) {
        const fb = fbResult.attempts > 1 ? ` [fallback t=${fbResult.temperature}]` : '';
        log(`  [${si + 1}/${segments.length}] ${segResult.text.slice(0, 60)}${fb}`, verbose);
      }
    }
  }

  // ── 6. Summary ──
  const asrTime = (performance.now() - tAsr) / 1000;
  const fullText = allSegments.map(s => s.text).join(' ');
  const wordCount = fullText.split(/\s+/).filter(Boolean).length;
  log(`\nASR: ${asrTime.toFixed(1)}s for ${segments.length} segments, ${wordCount} words`, verbose);
  log(`Fallbacks triggered: ${fallbackCount}/${segments.length} segments`, verbose);
  log(`Temperatures used: ${[...temperaturesUsed].sort((a, b) => a - b).join(', ')}`, verbose);

  const vttContent = toVtt(allSegments);

  // Cleanup
  try { fs.unlinkSync(tmpWav); } catch {}

  return {
    segments: allSegments,
    fullText,
    wordCount,
    asrTime,
    vttContent,
    temperaturesUsed: [...temperaturesUsed].sort((a, b) => a - b),
    fallbackCount,
    gateResults: gateResultsAll,
  };
}

// ──────────────────────────────────────────────────────────
// CLI entry point
// ──────────────────────────────────────────────────────────
async function main() {
  const { opts, audioPath } = parseArgs(args);

  if (!audioPath) {
    console.error('Usage: node tests/smoke/whisperx-runner.mjs [options] <audio-file>');
    process.exit(1);
  }

  const result = await runAsrPipeline({ ...opts, audioPath });

  // Output VTT
  const outPath = path.join(REPO_ROOT, 'tmp', '_whisperx_result.vtt');
  fs.writeFileSync(outPath, result.vttContent);

  console.log(`\n${'='.repeat(60)}`);
  console.log(`TRANSCRIPT (${result.segments.length} segments):`);
  console.log(`${'='.repeat(60)}`);
  console.log(result.fullText);
  console.log(`${'='.repeat(60)}`);
  console.log(`\nOutput saved: ${outPath}`);

  // Compare with reference VTT
  const refVtt = path.join(REPO_ROOT, 'tests/fixtures', '12_dans.tr.vtt');
  if (fs.existsSync(refVtt)) {
    console.log('\nComparing with reference VTT (WER estimation)...');
    const refContent = fs.readFileSync(refVtt, 'utf-8');
    const refWords = refContent.split('\n')
      .filter(l => l && !l.startsWith('WEBVTT') && !l.startsWith('Kind:') && !l.startsWith('Language:') && !l.includes('-->'))
      .map(l => l.trim()).filter(Boolean).join(' ').split(/\s+/);
    const hypWords = result.fullText.split(/\s+/).filter(Boolean);

    console.log(`  Reference words: ${refWords.length}`);
    console.log(`  Hypothesis words: ${hypWords.length}`);

    const refSet = new Set(refWords.map(w => w.toLowerCase()));
    const hypSet = new Set(hypWords.map(w => w.toLowerCase()));
    const correct = [...hypSet].filter(w => refSet.has(w)).length;
    const ins = hypSet.size - correct;
    const del = [...refSet].filter(w => !hypSet.has(w)).length;
    const wer = refSet.size > 0 ? (ins + del) / refSet.size : 1;

    console.log(`  Correct unique: ${correct}, Ins: ${ins}, Del: ${del}`);
    console.log(`  Estimated WER: ${(wer * 100).toFixed(1)}%`);
  }

  console.log('\nDONE');
}

// Run as CLI (only when executed directly, not imported)
const isMain = process.argv[1] && (
  process.argv[1] === fileURLToPath(import.meta.url)
  || process.argv[1].endsWith('whisperx-runner.mjs')
);
if (isMain) {
  main().catch(e => { console.error(e.stack); process.exit(1); });
}
