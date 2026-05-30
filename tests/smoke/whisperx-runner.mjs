#!/usr/bin/env node
/**
 * WhisperX-compatible transcription runner.
 *
 * Accepts the same CLI parameters as WhisperX and runs our ASR pipeline.
 */

import path from 'node:path';
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// ──────────────────────────────────────────────────────────
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../..');

const args = process.argv.slice(2);
const usedIndices = new Set();

const BOOLEAN_FLAGS = new Set([
  'verbose', 'noalign', 'suppressnumerals', 'conditiononprevioustext',
  'diarize', 'speakerembeddings', 'highlightwords', 'returncharalignments',
  'printprogress', 'modelcacheonly', 'fp16',
]);

const opts = {
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

for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a.startsWith('--') && !usedIndices.has(i)) {
    usedIndices.add(i);
    const key = a.slice(2).replace(/-/g, '');
    const isBool = BOOLEAN_FLAGS.has(key);
    const val = !isBool && i + 1 < args.length && !args[i + 1].startsWith('--')
      ? args[++i] : true;
    usedIndices.add(i);

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

const audioFiles = args.filter((a, i) => !a.startsWith('--') && !usedIndices.has(i));
const audioPath = audioFiles[0];
if (!audioPath) {
  console.error('Usage: node tests/smoke/whisperx-runner.mjs [options] <audio-file>');
  process.exit(1);
}

// ──────────────────────────────────────────────────────────
function log(msg) { if (opts.verbose) console.log(msg); }

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(7, '0')}`;
}

function toVtt(segments) {
  const lines = ['WEBVTT', 'Kind: captions', `Language: ${opts.language || 'tr'}`, ''];
  for (const seg of segments) {
    lines.push(`${formatTime(seg.start)} --> ${formatTime(seg.end)}`);
    lines.push(seg.text);
    lines.push('');
  }
  return lines.join('\n');
}

// ──────────────────────────────────────────────────────────
async function main() {
  log(`Audio: ${audioPath}`);
  log(`Model: ${opts.model}`);
  log(`Language: ${opts.language || 'auto-detect'}`);
  log(`VAD: ${opts.vadBackendType} (onset=${opts.vadOnset}, offset=${opts.vadOffset})`);
  log(`Beam: ${opts.beamSize}, BestOf: ${opts.bestOf}`);
  log(`Compression threshold: ${opts.compressionRatioThreshold}`);
  log(`Condition on prev text: ${opts.conditionOnPreviousText}`);
  log('');

  // ── 1. Decode audio to 16kHz mono WAV ──
  log('Decoding audio...');
  const tmpWav = path.join(REPO_ROOT, '/tmp/_whisperx_input_16k.wav');
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
  log(`  ${frameCount} frames @ ${sampleRate}Hz = ${audioDuration.toFixed(1)}s`);

  // ── 2. Load model ──
  log('Loading ONNX model...');
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
  log(`  Loaded in ${((performance.now()-tLoad)/1000).toFixed(1)}s`);

  const melProc = new WhisperMelProcessor({ nMels: melBins });
  const timestampProc = new WhisperTimestampLogitProcessor(tokenizer, genConfig);

  function argmax(arr) {
    let idx = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[idx]) idx = i;
    return idx;
  }

  // ── 3. VAD segmentation ──
  log('Running VAD...');
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
  log(`  ${segments.length} VAD segments in ${((performance.now()-tVad)/1000).toFixed(2)}s`);
  for (const seg of segments.slice(0, 5)) {
    log(`    [${seg.startSeconds.toFixed(1)}-${seg.endSeconds.toFixed(1)}] ${seg.durationSeconds.toFixed(1)}s`);
  }
  if (segments.length > 5) log(`    ... and ${segments.length - 5} more`);

  // ── 4. ASR per segment ──
  log('Transcribing...');
  const tAsr = performance.now();
  const allSegments = [];

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    const startSample = Math.max(0, Math.floor(seg.startSeconds * sampleRate));
    const endSample = Math.min(pcm.length, Math.ceil(seg.endSeconds * sampleRate));
    const chunk = pcm.slice(startSample, endSample);
    if (chunk.length < 1600) continue;

    // Mel features
    const mel = WhisperMelProcessor.padToFrames(melProc.process(chunk), 3000);
    const encOut = await encSess.run({
      input_features: new ort.Tensor('float32', mel, [1, melBins, 3000]),
    });
    const encKey = Object.keys(encOut).find(k => k === 'last_hidden_state' || k.includes('hidden') || k.includes('output')) ?? Object.keys(encOut)[0];
    const encHs = new Float32Array(encOut[encKey].data);
    const dModel = modelConfig.dModel ?? 384;
    const encDims = [1, encHs.length / dModel, dModel];

    // Prompt
    const promptTokens = [
      tokenizer.getTokenId('<|startoftranscript|>') ?? 50258,
      langId,
      tokenizer.getTokenId('<|transcribe|>') ?? 50359,
      tokenizer.getTokenId('<|notimestamps|>') ?? 50363,
    ];
    if (opts.initialPrompt) {
      promptTokens.push(...(tokenizer.encode(opts.initialPrompt) ?? []));
    }
    const promptLen = promptTokens.length;

    // Encoder dims for init
    const encTensor = new ort.Tensor('float32', encHs, encDims);
    const initFeeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(promptTokens.map(BigInt)), [1, promptTokens.length]),
      encoder_hidden_states: encTensor,
    };
    const initOut = await initSess.run(initFeeds);

    // Find logits output name
    const logitsKey = Object.keys(initOut).find(k => k.startsWith('logits')) ?? Object.keys(initOut)[0];
    const initLogitsData = initOut[logitsKey];
    const vocabSize = initLogitsData.dims[initLogitsData.dims.length - 1];

    // First token from init
    const lastOffset = initLogitsData.data.length - vocabSize;
    const firstLogits = initLogitsData.data.subarray(lastOffset, lastOffset + vocabSize);
    timestampProc.process(firstLogits, promptTokens, promptLen);
    const firstToken = argmax(firstLogits);
    const tokens = [firstToken];

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

    // Decode loop (greedy)
    const maxNewTokens = genConfig.maxLength ?? 448;
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
      const stepLogits = new Float32Array(stepOut[stepLogitsKey].data);
      timestampProc.process(stepLogits, [...promptTokens, ...tokens], promptLen);
      const nextToken = argmax(stepLogits);
      tokens.push(nextToken);

      // Track KV from step output — step only outputs decoder KV,
      // keep encoder KV from init across iterations
      const stepKvKeys = Object.keys(stepOut).filter(k => k.startsWith('present'));
      const newKv = {};
      // First, preserve encoder KV from previous iteration
      for (const [k, v] of Object.entries(pastKv)) {
        if (k.includes('.encoder.')) {
          newKv[k] = v; // encoder KV is unchanged across steps
        }
      }
      // Then, overlay decoder KV from step output
      for (const k of stepKvKeys) {
        newKv[k] = new Float32Array(stepOut[k].data);
        kvDims[k] = stepOut[k].dims;
        kvDims[k.replace(/^present\./, 'past_key_values.')] = stepOut[k].dims;
      }
      pastKv = newKv;

      if (nextToken === eosId) break;
    }

    // Decode tokens to text (skip prompt tokens, strip special tokens)
    const text = tokens
      .map(t => { try { const d = tokenizer.decode([t]); return d; } catch { return ''; } })
      .filter(t => t && !t.startsWith('<|') && !t.startsWith('['))
      .join('')
      .trim();

    if (text) {
      allSegments.push({ start: seg.startSeconds, end: seg.endSeconds, text });
      log(`  [${(si+1)}/${segments.length}] ${text.slice(0, 60)}`);
    }
  }

  // ── 5. Summary ──
  const tTotal = (performance.now() - tAsr) / 1000;
  const fullText = allSegments.map(s => s.text).join(' ');
  const wordCount = fullText.split(/\s+/).filter(Boolean).length;
  log(`\nASR: ${tTotal.toFixed(1)}s for ${segments.length} segments, ${wordCount} words`);

  const vttContent = toVtt(allSegments);
  const outPath = path.join(REPO_ROOT, 'tmp', '_whisperx_result.vtt');
  fs.writeFileSync(outPath, vttContent);

  log(`\n${'='.repeat(60)}`);
  log(`TRANSCRIPT (${allSegments.length} segments):`);
  log(`${'='.repeat(60)}`);
  log(fullText);
  log(`${'='.repeat(60)}`);
  log(`\nOutput saved: ${outPath}`);

  // ── 6. Compare with reference VTT ──
  const refVtt = path.join(REPO_ROOT, 'tests/fixtures', '12_dans.tr.vtt');
  if (fs.existsSync(refVtt)) {
    log('\nComparing with reference VTT (WER estimation)...');
    const refContent = fs.readFileSync(refVtt, 'utf-8');
    const refWords = refContent.split('\n').filter(l => l && !l.startsWith('WEBVTT') && !l.startsWith('Kind:') && !l.startsWith('Language:') && !l.includes('-->')).map(l => l.trim()).filter(Boolean).join(' ').split(/\s+/);
    const hypWords = fullText.split(/\s+/).filter(Boolean);

    log(`  Reference words: ${refWords.length}`);
    log(`  Hypothesis words: ${hypWords.length}`);

    // Unique-word-based WER estimate
    const refSet = new Set(refWords.map(w => w.toLowerCase()));
    const hypSet = new Set(hypWords.map(w => w.toLowerCase()));
    const correct = [...hypSet].filter(w => refSet.has(w)).length;
    const wer = hypSet.size > 0 ? (hypSet.size - correct + [...refSet].filter(w => !hypSet.has(w)).length) / refSet.size : 1;

    log(`  Correct unique: ${correct}, Estimated WER: ${(wer * 100).toFixed(1)}%`);
  }

  try { fs.unlinkSync(tmpWav); } catch {}
  console.log('\nDONE');
}

main().catch(e => { console.error(e.stack); process.exit(1); });
