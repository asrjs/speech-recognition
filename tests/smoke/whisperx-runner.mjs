#!/usr/bin/env node
/**
 * WhisperX-compatible transcription runner with full quality gates + word timestamps.
 *
 * Accepts WhisperX CLI parameters:
 *   --model, --language, --vad_onset, --word_timestamps, --beam_size,
 *   --compression_ratio_threshold, --logprob_threshold,
 *   --no_speech_threshold, --entropy_threshold, --temperature, etc.
 *
 * Exports runAsrPipeline() for programmatic use.
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
  'printprogress', 'modelcacheonly', 'fp16', 'wordtimestamps',
  'nowordtimestamps',
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
  verbose: false,
  wordTimestamps: true,
  w2vModel: null,
};

function parseArgs(customArgs) {
  const opts = { ...DEFAULT_OPTS };
  const parsed = customArgs ?? args;
  const localUsed = new Set();

  for (let i = 0; i < parsed.length; i++) {
    const a = parsed[i];
    if (a.startsWith('--') && !localUsed.has(i)) {
      localUsed.add(i);
      const key = a.slice(2).replace(/[-_]/g, '');
      const isBool = BOOLEAN_FLAGS.has(key);
      const val = !isBool && i + 1 < parsed.length && !parsed[i + 1].startsWith('--')
        ? parsed[++i] : true;
      localUsed.add(i);

      const mapping = {
        model: 'model', modeldir: 'model',
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
        'w2v-model': 'w2vModel', w2vmodel: 'w2vModel',
        wordtimestamps: 'wordTimestamps',
        nowordtimestamps: 'noWordTimestamps',
      };

      const optKey = mapping[key];
      if (optKey === 'noWordTimestamps') {
        opts.wordTimestamps = false;
      } else if (optKey) {
        if (['noAlign', 'suppressNumerals', 'conditionOnPreviousText', 'verbose', 'wordTimestamps'].includes(optKey)) {
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

export { parseArgs };

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

function toVtt(segments, language = 'en') {
  const lines = ['WEBVTT', 'Kind: captions', `Language: ${language || 'en'}`, ''];
  for (const seg of segments) {
    lines.push(`${formatTime(seg.start)} --> ${formatTime(seg.end)}`);
    lines.push(seg.text);
    lines.push('');
  }
  return lines.join('\n');
}

function toSrt(segments) {
  return segments.map((seg, i) => {
    const start = formatTime(seg.start).replace('.', ',');
    const end = formatTime(seg.end).replace('.', ',');
    return `${i + 1}\n${start} --> ${end}\n${seg.text}\n`;
  }).join('\n');
}

function toTxt(segments) {
  return segments.map(s => s.text).join('\n');
}

function toJson(segments, meta) {
  return JSON.stringify({
    segments: segments.map(s => ({
      text: s.text,
      start: s.start,
      end: s.end,
      words: s.words?.map(w => ({
        text: w.text,
        start: w.start,
        end: w.end,
      })),
    })),
    ...(meta ? { metadata: meta } : {}),
  }, null, 2);
}

// ──────────────────────────────────────────────────────────
// Temperature sampling
// ──────────────────────────────────────────────────────────

function argmax(arr) {
  let idx = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > arr[idx]) idx = i;
  }
  return idx;
}

function sample(logits, temperature) {
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  let maxVal = -Infinity;
  for (let i = 0; i < scaled.length; i++) {
    if (scaled[i] > maxVal) maxVal = scaled[i];
  }
  const exps = new Float32Array(scaled.length);
  let sumExp = 0;
  for (let i = 0; i < scaled.length; i++) {
    const e = Math.exp(scaled[i] - maxVal);
    exps[i] = e;
    sumExp += e;
  }
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < exps.length; i++) {
    cum += exps[i] / sumExp;
    if (r < cum) return i;
  }
  return exps.length - 1;
}

export { sample };

function nextToken(logits, temperature) {
  return temperature <= 0 ? argmax(logits) : sample(logits, temperature);
}

/** Build word-level timestamps from DTW token timestamps. */
function buildWordsFromTimestamps(
  textTokenIds, decodedTokens, segStart, dtwTimestamps, tokenizer,
) {
  if (textTokenIds.length === 0 || dtwTimestamps.length < 2) return [];
  const words = [];
  let bufText = '';
  let bufStart = segStart + dtwTimestamps[0];

  for (let i = 0; i < textTokenIds.length; i++) {
    const t = decodedTokens[i] ?? '';
    // Whisper BPE: tokens starting with space are word-starts
    // Also handle punctuation boundaries
    const isWordStart = t.startsWith(' ') || (
      i > 0 && (decodedTokens[i - 1]?.endsWith('.') || decodedTokens[i - 1]?.endsWith('!') ||
                decodedTokens[i - 1]?.endsWith('?') || decodedTokens[i - 1]?.endsWith(':') ||
                decodedTokens[i - 1]?.endsWith(';'))
    );

    if (isWordStart && bufText) {
      words.push({ text: bufText.trim(), start: bufStart, end: segStart + dtwTimestamps[i] });
      bufText = t;
      bufStart = segStart + dtwTimestamps[i];
    } else {
      bufText += t;
    }
  }
  if (bufText.trim()) {
    words.push({ text: bufText.trim(), start: bufStart, end: segStart + dtwTimestamps[dtwTimestamps.length - 1] });
  }
  return words;
}

function nativeToRunnerWords(words, offsetSeconds = 0) {
  return words.map((word) => ({
    text: word.text,
    start: offsetSeconds + word.startTime,
    end: offsetSeconds + word.endTime,
  }));
}

// ──────────────────────────────────────────────────────────
// Core ASR pipeline
// ──────────────────────────────────────────────────────────

export async function runAsrPipeline(_opts) {
  const opts = { ...DEFAULT_OPTS, ..._opts };
  const { audioPath } = opts;
  const verbose = opts.verbose;

  log(`Audio: ${audioPath}`, verbose);
  log(`Model: ${opts.model}`, verbose);
  log(`Language: ${opts.language || 'auto-detect'}`, verbose);
  log(`Word timestamps: ${opts.wordTimestamps ? 'yes' : 'no'}`, verbose);
  log(`Output format: ${opts.outputFormat}`, verbose);
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
  const {
    collectSplitGraphTextTokenRows,
    processSplitGraphAlignment,
    processSplitGraphAlignmentByTimestampSpans,
  } = await import(
    path.join(dist, 'models/whisper-seq2seq/executor.js')
  );
  const {
    buildWhisperWordTimestampsFromDtwTokens,
    refineWhisperWordsWithForcedAlignment,
  } = await import(
    path.join(dist, 'models/whisper-seq2seq/word-timestamps.js')
  );
  const { whisperDecode } = await import(
    path.join(dist, 'models/whisper-seq2seq/core.js')
  );
  const { createWav2Vec2AlignerFromLogits } = await import(
    path.join(dist, 'alignment.js')
  );

  const tokenizer = await WhisperTokenizer.fromUrl(path.join(opts.model, 'tokenizer.json'));
  let language = 'en';
  let langId = tokenizer.getTokenId('<|en|>') ?? 50259;

  const genConfig = parseWhisperGenerationConfig(
    JSON.parse(await fetchText(path.join(opts.model, 'generation_config.json')))
  );
  const configRaw = JSON.parse(await fetchText(path.join(opts.model, 'config.json')));
  const modelConfig = parseWhisperModelConfig(configRaw);
  const dModel = modelConfig.dModel ?? 384;
  const melBins = modelConfig.numMelBins ?? 80;
  const eosId = tokenizer.getTokenId('<|endoftext|>') ?? 50257;
  const timestampBeginId = tokenizer.getTokenId('<|0.00|>') ?? 50364;
  const timestampEndId = tokenizer.getTokenId('<|30.00|>') ?? 51864;
  const noSpeechTokenId = genConfig.noSpeechTokenId
    ?? tokenizer.getTokenId('<|nospeech|>')
    ?? 50362;

  // Load split-graph sessions
  const encSess = await ort.InferenceSession.create(path.join(opts.model, 'encoder_model.onnx'));
  const initSess = await ort.InferenceSession.create(path.join(opts.model, 'decoder_init.onnx'));
  const stepSess = await ort.InferenceSession.create(path.join(opts.model, 'decoder_step.onnx'));

  // Load alignment session (for word timestamps)
  let alignSess = null;
  if (opts.wordTimestamps) {
    try {
      alignSess = await ort.InferenceSession.create(path.join(opts.model, 'decoder_align.onnx'));
    } catch {
      log('  Warning: decoder_align.onnx not found — word timestamps disabled', verbose);
    }
  }

  log(`  Loaded in ${((performance.now() - tLoad) / 1000).toFixed(1)}s`, verbose);

  const W2V_MODEL_BY_LANG = {
    en: {
      path: opts.w2vModel || '/tmp/wav2vec2-english-onnx/wav2vec2-base-960h.fp16.onnx',
      dataFile: '/tmp/wav2vec2-english-onnx/wav2vec2-base-960h.fp16.onnx.data',
      vocabUrl: '/tmp/wav2vec2-publish/vocab.json',
    },
    tr: {
      path: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.fp16.onnx',
      dataFile: '/tmp/wav2vec2-turkish-onnx/wav2vec2-large-xlsr-turkish.fp16.onnx.data',
      vocabUrl: '/tmp/wav2vec2-turkish-onnx/vocab.json',
    },
  };

  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const timestampProc = new WhisperTimestampLogitProcessor({
    eosTokenId: eosId,
    noTimestampsTokenId: genConfig.noTimestampsTokenId ?? tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
    timestampBegin: timestampBeginId,
    suppressTokens: genConfig.suppressTokens ?? [],
    beginSuppressTokens: genConfig.beginSuppressTokens ?? [],
  });

  // ── Language auto-detection (first 30s) ──
  {
    const langIdRaw = opts.language || null;
    if (!langIdRaw || langIdRaw === 'auto') {
      log('Detecting language...', verbose);
      const detectSamples = Math.min(pcm.length, 30 * sampleRate);
      try {
        const detectMel = WhisperMelProcessor.padToFrames(melProc.process(pcm.slice(0, detectSamples)), 3000);
        const detectEncOut = await encSess.run({
          input_features: new ort.Tensor('float32', detectMel, [1, melBins, 3000]),
        });
        const detectEncKey = Object.keys(detectEncOut).find(k =>
          k === 'last_hidden_state' || k.includes('hidden') || k.includes('output')
        ) ?? Object.keys(detectEncOut)[0];
        const detectEncHs = new Float32Array(detectEncOut[detectEncKey].data);
        const detectEncTensor = new ort.Tensor('float32', detectEncHs, [1, detectEncHs.length / dModel, dModel]);

        const sotId = tokenizer.getTokenId('<|startoftranscript|>') ?? 50258;
        const detectOut = await initSess.run({
          input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(sotId)]), [1, 1]),
          encoder_hidden_states: detectEncTensor,
        });
        const detectLogitsKey = Object.keys(detectOut).find(k => k.includes('logits')) ?? Object.keys(detectOut)[0];
        const detectLogits = new Float32Array(detectOut[detectLogitsKey].data);
        const vSize = detectOut[detectLogitsKey].dims[detectOut[detectLogitsKey].dims.length - 1];

        let maxLogit = -Infinity, maxLangToken = -1;
        for (let i = 50259; i <= 50357 && i < vSize; i++) {
          if (detectLogits[i] > maxLogit) { maxLogit = detectLogits[i]; maxLangToken = i; }
        }
        if (maxLangToken > 0) {
          const langToken = tokenizer.decode([maxLangToken]) || '';
          const match = langToken.match(/<\|(\w+)\|>/);
          if (match) { language = match[1]; langId = maxLangToken; }
        }
        log(`  Detected: ${language}`, verbose);
      } catch (detectErr) {
        log(`  Lang detect: ${detectErr.message}, using ${language}`, verbose);
      }
    } else {
      language = langIdRaw;
      langId = tokenizer.getTokenId(`<|${language}|>`) ?? tokenizer.getTokenId('<|en|>') ?? 50259;
    }
  }

  // Load Wav2Vec2 after language detection so Turkish uses the XLS-R model.
  let w2vSession = null;
  let w2vTokenizer = null;
  if (!opts.noAlign && opts.wordTimestamps) {
    const w2vCfg = W2V_MODEL_BY_LANG[language] || W2V_MODEL_BY_LANG.en;
    try {
      log(`Loading Wav2Vec2 aligner (${language})...`, verbose);
      const w2vOpts = {};
      if (fs.existsSync(w2vCfg.dataFile)) {
        const data = fs.readFileSync(w2vCfg.dataFile);
        w2vOpts.externalData = [{ path: path.basename(w2vCfg.path), data }];
      }
      w2vSession = await ort.InferenceSession.create(w2vCfg.path, w2vOpts);

      const { Wav2Vec2CharTokenizer } = await import(
        path.join(dist, 'models/wav2vec2/tokenizer.js')
      );
      w2vTokenizer = await Wav2Vec2CharTokenizer.fromUrl(w2vCfg.vocabUrl);
      log(`  Wav2Vec2 ready (${language})`, verbose);
    } catch (w2vErr) {
      log(`  Wav2Vec2 not loaded (${w2vErr.message}) — falling back to DTW alignment`, verbose);
    }
  }

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
  if (verbose) {
    for (const seg of segments.slice(0, 5)) {
      log(`    [${seg.startSeconds.toFixed(1)}-${seg.endSeconds.toFixed(1)}] ${seg.durationSeconds.toFixed(1)}s`);
    }
    if (segments.length > 5) log(`    ... and ${segments.length - 5} more`);
  }

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

  const inc = opts.temperatureIncrement ?? 0.2;
  const startTemp = opts.temperature ?? 0;
  const temperatures = [startTemp];
  if (inc > 0) {
    for (let t = startTemp + inc; t <= 1.0 + 1e-6; t += inc) {
      temperatures.push(Math.round(t * 1000) / 1000);
    }
  }

  // ── 5. Transcribe each segment ──
  log('Transcribing...', verbose);
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

    const mel = WhisperMelProcessor.padToFrames(melProc.process(chunk), 3000);

    // Encoder (shared across fallback attempts)
    const encOut = await encSess.run({
      input_features: new ort.Tensor('float32', mel, [1, melBins, 3000]),
    });
    const encKey = Object.keys(encOut).find(k =>
      k === 'last_hidden_state' || k.includes('hidden') || k.includes('output')
    ) ?? Object.keys(encOut)[0];
    const encHs = new Float32Array(encOut[encKey].data);
    const dModel = modelConfig.dModel ?? 384;
    const encoderFrameCount = encHs.length / dModel;
    const encDims = [1, encoderFrameCount, dModel];
    const encTensor = new ort.Tensor('float32', encHs, encDims);

    // Transcribe function for temperature fallback
    const transcribeFn = async (temperature) => {
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
      const maxNewTokens = (genConfig.maxLength ?? 448) - promptTokens.length - 1;

      // Session adapter for whisperDecode
      const kvDims = {};
      let decoderInitLogits;
      let decoderInitVocabSize = 51865;
      let decoderInitNoSpeechTokenId = noSpeechTokenId;
      const coreSession = {
        runInit: async (pt, _enc, _dims) => {
          const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(pt.map(BigInt)), [1, pt.length]),
            encoder_hidden_states: encTensor,
          };
          const out = await initSess.run(feeds);
          const lk = Object.keys(out).find(k => k.startsWith('logits')) ?? Object.keys(out)[0];
          const lt = out[lk];
          const vSize = lt.dims[lt.dims.length - 1];
          decoderInitVocabSize = vSize;
          const lastOff = lt.data.length - vSize;
          const logits = new Float32Array(lt.data.subarray(lastOff, lastOff + vSize));
          const pk = {};
          for (const k of Object.keys(out).filter(k => k.startsWith('present'))) {
            pk[k] = new Float32Array(out[k].data);
            kvDims[k] = out[k].dims;
            kvDims[k.replace(/^present\./, 'past_key_values.')] = out[k].dims;
          }
          return { logits, vocabSize: vSize, presentKv: pk };
        },
        runStep: async (tid, pastKv) => {
          const feeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(tid)]), [1, 1]),
          };
          for (const [k, v] of Object.entries(pastKv)) {
            const sk = k.replace(/^present\./, 'past_key_values.');
            const dims = kvDims[k] ?? kvDims[sk] ?? kvDims[k.replace(/^past_key_values\./, 'present.')];
            if (dims) feeds[sk] = new ort.Tensor('float32', v, dims);
          }
          const out = await stepSess.run(feeds);
          const lk = Object.keys(out).find(k => k.startsWith('logits'));
          const lt = out[lk];
          const vSize = lt.dims[lt.dims.length - 1];
          const sl = new Float32Array(lt.data);
          const slLast = sl.subarray(sl.length - vSize, sl.length);
          const pk = {};
          for (const k of Object.keys(out).filter(k => k.startsWith('present'))) {
            pk[k] = new Float32Array(out[k].data);
            kvDims[k] = out[k].dims;
            kvDims[k.replace(/^present\./, 'past_key_values.')] = out[k].dims;
          }
          // Preserve encoder KV
          for (const [k, v] of Object.entries(pastKv)) {
            if (k.includes('.encoder.') && !pk[k]) pk[k] = v;
          }
          return { logits: new Float32Array(slLast), vocabSize: vSize, presentKv: pk };
        },
      };

      const decodeResult = await whisperDecode(coreSession, {
        promptTokens,
        encoderOutput: encHs,
        encoderDims: encDims,
        eosTokenId: eosId,
        maxNewTokens,
        temperature,
        trackQuality: true,
        noSpeechTokenId,
        processLogits: (logits, genTokens, beginIdx) => {
          timestampProc.process(logits, genTokens, beginIdx);
        },
        onDecoderInitLogits: (rawLogits, initCtx) => {
          decoderInitLogits = new Float32Array(rawLogits);
          decoderInitVocabSize = initCtx.vocabSize;
          decoderInitNoSpeechTokenId = initCtx.noSpeechTokenId ?? noSpeechTokenId;
        },
        strategy: (opts.beamSize ?? 1) > 1 ? 'beam' : 'greedy',
        beamSize: opts.beamSize ?? 1,
        lengthPenalty: opts.lengthPenalty ?? 0,
        bestOf: opts.bestOf ?? 1,
      });

      const tokens = [...decodeResult.tokens];
      const qualityTokens = decodeResult.tokenTraces && decodeResult.tokenTraces.length > 0
        ? decodeResult.tokenTraces.map(trace => trace.tokenId)
        : tokens;

      // Build segment text (needed for Wav2Vec2 alignment)
      const text = tokens
        .map(t => { try { return tokenizer.decode([t]); } catch { return ''; } })
        .filter(t => t && !t.startsWith('<|') && !t.startsWith('['))
        .join('')
        .trim();

      // ── Word timestamps via decoder_align + DTW ──
      let segmentWords = [];
      let nativeWords = [];
      if (alignSess && opts.wordTimestamps) {
        const allTokenIds = [...promptTokens, ...tokens];
        const isTimestampToken = (id) => id >= timestampBeginId && id <= timestampEndId;
        const isTextToken = (id) => {
          const td = tokenizer.decode([id]) || '';
          return Boolean(td && !td.startsWith('<|') && !td.startsWith('[') && !td.startsWith('�'));
        };
        const { tokenIds: textIds, rowIndices } = collectSplitGraphTextTokenRows(
          allTokenIds,
          promptTokens.length,
          isTextToken,
        );
        const decodedTexts = textIds.map((id) => tokenizer.decode([id]) || '');
        const cropFrameCount = Math.max(1, Math.round((chunk.length / sampleRate) / 0.02));

        try {
          const alignFeeds = {
            input_ids: new ort.Tensor('int64', BigInt64Array.from(allTokenIds.map(BigInt)), [1, allTokenIds.length]),
            encoder_hidden_states: encTensor,
          };
          const alignOut = await alignSess.run(alignFeeds);
          const alignKey = Object.keys(alignOut)[0];
          const alignmentData = new Float32Array(alignOut[alignKey].data);

          const dtwTimestamps = processSplitGraphAlignmentByTimestampSpans({
            alignmentData,
            tokenIds: allTokenIds,
            promptLen: promptTokens.length,
            frameCount: encoderFrameCount,
            timePrecisionSeconds: 0.02,
            cropFrameCount,
            isTextToken,
            isTimestampToken,
            timestampTokenToSeconds: (id) => (id - timestampBeginId) * 0.02,
          }) ?? processSplitGraphAlignment({
            alignmentData,
            promptLen: promptTokens.length,
            textTokenCount: textIds.length,
            frameCount: encoderFrameCount,
            timePrecisionSeconds: 0.02,
            textTokenRowIndices: rowIndices,
            cropFrameCount,
          });

          nativeWords = buildWhisperWordTimestampsFromDtwTokens(
            textIds.map((id, index) => ({
              id,
              text: decodedTexts[index] ?? '',
              sourceIndex: index,
            })),
            dtwTimestamps,
            { language },
          );
          segmentWords = nativeToRunnerWords(nativeWords, seg.startSeconds);
        } catch (alignErr) {
          if (verbose) log(`    align fail on seg ${si}: ${alignErr.message}`);
        }
      }

      // ── Wav2Vec2 forced alignment (refines DTW when available) ──
      if (w2vSession && w2vTokenizer && opts.wordTimestamps && !opts.noAlign) {
        try {
          const inputTensor = new ort.Tensor('float32', chunk, [1, chunk.length]);
          const w2vOut = await w2vSession.run({ input_values: inputTensor });
          const w2vKey = Object.keys(w2vOut)[0];
          const w2vTensor = w2vOut[w2vKey];
          const logitsData = new Float32Array(w2vTensor.data);
          const w2vDims = w2vTensor.dims;
          const wfc = w2vDims[1];
          const wvs = w2vDims[2];

          const reusableLogits = {
            logits: logitsData, frameCount: wfc, vocabSize: wvs,
            blankId: 0, tokenizer: w2vTokenizer,
            sampleRate: 16000, audioDurationSeconds: chunk.length / 16000,
            wordSeparator: ' ',
          };

          const aligner = createWav2Vec2AlignerFromLogits(reusableLogits);
          const alignment = aligner.align({
            transcript: text,
            audioDurationSeconds: chunk.length / 16000,
          });

          if (alignment.words && alignment.words.length > 0) {
            const aligned = alignment.words.map((word) => ({
              text: word.text,
              startTime: word.start,
              endTime: word.end,
              confidence: word.confidence,
            }));
            const baseWords = nativeWords.length > 0
              ? nativeWords
              : aligned.map((word, index) => ({
                  index,
                  text: word.text,
                  startTime: word.startTime,
                  endTime: word.endTime,
                }));
            nativeWords = refineWhisperWordsWithForcedAlignment(baseWords, aligned);
            segmentWords = nativeToRunnerWords(nativeWords, seg.startSeconds);
          }
        } catch (w2vErr) {
          if (verbose) log(`    w2v align fail on seg ${si}: ${w2vErr.message}`);
        }
      }

      return {
        result: { text, start: seg.startSeconds, end: seg.endSeconds, words: segmentWords },
        text,
        tokens: qualityTokens,
        logits: [],
        vocabSize: decoderInitVocabSize,
        qualityContext: {
          ...(decoderInitLogits ? { noSpeechLogits: decoderInitLogits } : {}),
          noSpeechTokenId: decoderInitNoSpeechTokenId,
          ...(decodeResult.tokenTraces && decodeResult.tokenTraces.length > 0
            ? { tokenTraces: decodeResult.tokenTraces }
            : {}),
        },
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
        const wordsInfo = segResult.words?.length ? ` (${segResult.words.length} words)` : '';
        log(`  [${si + 1}/${segments.length}] ${segResult.text.slice(0, 60)}${fb}${wordsInfo}`);
      }
    }
  }

  // ── 6. Summary ──
  const asrTime = (performance.now() - tAsr) / 1000;
  const fullText = allSegments.map(s => s.text).join(' ');
  const wordCount = fullText.split(/\s+/).filter(Boolean).length;
  log(`\nASR: ${asrTime.toFixed(1)}s for ${segments.length} segments, ${wordCount} words`, verbose);
  if (verbose) {
    log(`Fallbacks triggered: ${fallbackCount}/${segments.length} segments`);
    log(`Temperatures used: ${[...temperaturesUsed].sort((a, b) => a - b).join(', ')}`);
  }

  // Generate outputs
  const outputs = {};
  switch (opts.outputFormat) {
    case 'srt':
      outputs.srt = toSrt(allSegments);
      break;
    case 'txt':
      outputs.txt = toTxt(allSegments);
      break;
    case 'json':
      outputs.json = toJson(allSegments, {
        audioDuration,
        asrTime,
        language,
        segments: allSegments.length,
      });
      break;
    default:
      outputs.vtt = toVtt(allSegments, language);
      break;
  }

  // Cleanup
  try { fs.unlinkSync(tmpWav); } catch {}

  return {
    segments: allSegments,
    fullText,
    language,
    wordCount,
    asrTime,
    outputs,
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
    console.error('');
    console.error('Options:');
    console.error('  --model <path>                ONNX model directory');
    console.error('  --language <code>             Language (en, tr, auto)');
    console.error('  --vad_onset <float>           VAD onset threshold (0.5)');
    console.error('  --vad_offset <float>          VAD offset threshold (0.363)');
    console.error('  --chunk_size <sec>            Max segment duration (30)');
    console.error('  --word_timestamps             Enable word-level timestamps (default: on)');
    console.error('  --no-word_timestamps          Disable word-level timestamps');
    console.error('  --temperature <float>         Decoding temperature (0.0)');
    console.error('  --temperature_increment_on_fallback <float>  Fallback step (0.2)');
    console.error('  --beam_size <int>              Beam search width (1 = greedy)');
    console.error('  --best_of <int>                Independent decodings to pick best');
    console.error('  --patience <float>             Beam search patience');
    console.error('  --length_penalty <float>       Beam search length penalty (1.0)');
    console.error('  --compression_ratio_threshold <float>  2.4');
    console.error('  --logprob_threshold <float>           -1.0');
    console.error('  --no_speech_threshold <float>         0.6');
    console.error('  --entropy_threshold <float>           2.4');
    console.error('  --output_format <vtt|srt|txt|json>    Output format');
    console.error('  --no-align                     Skip forced alignment');
    console.error('  --w2v_model <path>             Wav2Vec2 model path');
    console.error('  --verbose                      Print progress');
    console.error('');
    console.error('Example:');
    console.error('  node tests/smoke/whisperx-runner.mjs \\');
    console.error('    --model /tmp/whisper-base-4graph/fp32 \\');
    console.error('    --language tr --word_timestamps \\');
    console.error('    tests/fixtures/12_dans.tr.m4a');
    process.exit(1);
  }

  const result = await runAsrPipeline({ ...opts, audioPath });

  // Write output
  const outBase = path.join(REPO_ROOT, 'tmp', '_whisperx_result');
  if (result.outputs.vtt) {
    fs.writeFileSync(outBase + '.vtt', result.outputs.vtt);
    console.log(`\n  VTT: ${outBase}.vtt`);
  }
  if (result.outputs.srt) {
    fs.writeFileSync(outBase + '.srt', result.outputs.srt);
    console.log(`  SRT: ${outBase}.srt`);
  }
  if (result.outputs.txt) {
    fs.writeFileSync(outBase + '.txt', result.outputs.txt);
    console.log(`  TXT: ${outBase}.txt`);
  }
  if (result.outputs.json) {
    fs.writeFileSync(outBase + '.json', result.outputs.json);
    console.log(`  JSON: ${outBase}.json`);
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(`TRANSCRIPT (${result.segments.length} segments):`);
  console.log(`${'='.repeat(60)}`);
  console.log(result.fullText);
  console.log(`${'='.repeat(60)}`);

  if (result.segments.length > 0 && result.segments[0].words?.length) {
    const firstSeg = result.segments[0];
    console.log(`\nSample words (segment 1, ${firstSeg.words.length} words):`);
    for (const w of firstSeg.words.slice(0, 10)) {
      console.log(`  ${formatTime(w.start)} --> ${formatTime(w.end)}  ${w.text}`);
    }
    if (firstSeg.words.length > 10) console.log(`  ... and ${firstSeg.words.length - 10} more`);
  }

  console.log(`\nDONE`);
}

const isMain = process.argv[1] && (
  process.argv[1] === fileURLToPath(import.meta.url)
  || process.argv[1].endsWith('whisperx-runner.mjs')
);
if (isMain) {
  main().catch(e => { console.error(e.stack); process.exit(1); });
}
