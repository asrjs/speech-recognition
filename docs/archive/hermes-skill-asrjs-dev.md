---
name: asrjs-dev
description: Work on the asrjs/speech-recognition library. Use when fixing bugs, triaging issues, running tests, building demos, or working on any model family (Whisper, Wav2Vec2, NeMo, MedASR).
platforms: [linux]
---

## CRITICAL RULE: Library-Only Decode

**All decode logic, token processing, KV cache management, and mel spectrograms live in the library — never reimplement in test pages.**

Browser test pages (e.g. `webgpu-agent-test/index.html`) must be **UI shells** only: load models, compute mel via `WhisperMelProcessor` + `padToFrames`, call `whisperDecode()` from `core.ts` via `WhisperCoreSession` wrapper, display results. No custom mel, no custom decode loop, no custom logit processor, no custom KV bridge.

When the library can't be directly imported (ES module path resolution in browser), inline the exact library code with a clear header: `// Synced from src/path/to/file.ts — Last sync: YYYY-MM-DD`. The sync header means both copies must be updated together.

**Verification policy: Node ORT first, WebGPU last.** Every model variant (fp32, fp16, fp16io, q8, mixed) must pass all 5 verification steps on Node ORT (CPU EP) before WebGPU testing:

```
Step 1: Mel spectrogram     → MSE against regenerated reference
Step 2: Encoder output       → cosine similarity vs fp32 baseline (>0.999)
Step 3: Decoder init logits  → top-5 token IDs match baseline
Step 4: Full decode pipeline → coherent transcript, no early EOS
Step 5: Token-by-token       → first 5 tokens match fp32 baseline
```

Why: A session spent ~150 tool calls debugging a test page that reimplemented the decode loop with 7 bugs. All bugs were already handled correctly in the library. The Node runner worked throughout.

## Known bugs found in WebGPU test pages:

During a session debugging `fp16io` (fp16 internal + fp32 I/O encoder + fp32 decoder) on WebGPU, the test page's `index.html` at `/mnt/n/github/asrjs/webgpu-agent-test/` had **7 bugs** that broke the decode pipeline:

| #   | Bug                           | Wrong value             | Correct (large-v3-turbo)                                                       |
| --- | ----------------------------- | ----------------------- | ------------------------------------------------------------------------------ |
| 1   | Task token                    | `50359` (translate)     | `50360` (transcribe)                                                           |
| 2   | No_timestamps token           | `50363` (base model ID) | `50364` (v3-turbo no_timestamps_token_id)                                      |
| 3   | `suppress_tokens`             | **Missing entirely**    | ~80 special tokens from `generation_config.json`                               |
| 4   | `begin_suppress_tokens`       | **Missing entirely**    | `[220, 50257]` — EOS blocked only at step 0                                    |
| 5   | No_timestamps mode            | **Missing**             | Check `genTokens.includes(50364)`, suppress timestamps 50364+                  |
| 6   | Sequential timestamp rules    | **Missing**             | First-gen-suppress-text, two-timestamps-suppress, monotonicity                 |
| 7   | Encoder KV cache preservation | **Missing**             | Step output only has decoder KV; encoder KV must be merged from previous cache |

Bugs 1-6 were fixed with a proper logit processor. Bug #7 was fixed with oldPkv snapshot + encoder KV merge.

**Result (Entry 023, 2026-06-01):** First working WebGPU Whisper pipeline — fp16io encoder (2.13s) + fp32 decoder (3.32s) = 25.57s total, zero NaN, valid English output. Transcript quality is degraded (fp16 encoder precision) but pipeline is mechanically sound.

**Result (Entry 024, 2026-06-14):** Full 4-graph WebGPU pipeline works with fp16 decoder too when the demo resolves the custom repo `ysdede/whisper-large-v3-turbo-onnx-4graph` instead of `onnx-community/...`. Verified preset: `fp16io-fp16-webgpu` (`fp16_iofp32` encoder + `fp16` decoder), 29.9s JFK fixture, 50-token cap, correct transcript, transcribe `5.82s`, RTF `0.1944`, RTFx `5.1452`.

**Whisper mel performance note:** Whisper uses `n_fft=400`, so do not switch it to the NeMo/Parakeet 512-point radix-2 STFT for speed. Keep exact Whisper semantics and use the cached Bluestein FFT path in `src/audio/whisper-mel.ts`. The old direct DFT path measured ~9185ms for 30s audio; the optimized path measures ~204ms with direct-DFT parity (`maxDiff < 1e-4`). Use `npm run benchmark:whisper-mel`.

**Decoder performance note:** A working KV cache does not make Whisper decoding parallel. It avoids recomputing old decoder tokens, but `decoder_step.onnx` still runs once per generated token. In the 2026-06-14 Chrome WebGPU fp16 run, decode was `3979ms`; `decoderStepRunMs` was `3788ms` across 49 steps, while JS feed build + tensor bridge + output handling was under `4ms` total. If decoder feels slow, profile `decoderInitRunMs`, `decoderStepRunMs`, `decoderStepP50Ms`, and `decoderStepP95Ms` before changing KV glue. Beam search and `best_of` multiply decoder-step runs and are expected to slow inference.

### Token-sequence diagnostic

The buggy test page produced: `50360 → 50364 → 50257(EOS)` — only 3 generated tokens, all special, no text. This is the **hallmark of a missing-prompt-tokens problem**, NOT encoder precision:

- Step 0: decoder_init with prompt `[50258, 50259]` → highest logit is 50360 (transcribe) — generated as a token
- Step 1: decoder_step(50360) → highest logit is 50364 (notimestamps) — generated as a token
- Step 2: decoder_step(50364) → now no_timestamps mode active → timestamps suppressed → EOS vs text → EOS wins (7.73 vs 7.70)

Without bugs #1 and #2 (wrong tokens), no_timestamps mode would never activate and timestamps would remain available, changing the entire decode trajectory.

### Required logit processor (JavaScript)

For a stand-alone WebGPU test page, this function matches the runner's `WhisperTimestampLogitProcessor`:

```javascript
const st = MODEL_CFG.suppressTokens;
const bst = MODEL_CFG.beginSuppressTokens;
const eosId = MODEL_CFG.eosId;
const tsBegin = MODEL_CFG.timestampBegin;
const noTsId = MODEL_CFG.noTimestampsTokenId;
const promptLen = MODEL_CFG.promptIds.length;

function applyLogitProcessor(logits, genTokens, beginIdx) {
  const eosBefore = logits[eosId];
  // 1. suppress_tokens (always)
  for (const id of st) if (id < logits.length) logits[id] = -Infinity;
  // 2. begin_suppress_tokens (only first generated token step)
  if (genTokens.length === beginIdx)
    for (const id of bst) if (id < logits.length) logits[id] = -Infinity;
  // 3. no_timestamps mode
  const hasNoTs = genTokens.includes(noTsId);
  if (hasNoTs) {
    for (let ts = tsBegin; ts < logits.length; ts++) logits[ts] = -Infinity;
    return;
  }
  // 4. Sequential timestamp rules
  const sampled = genTokens.slice(beginIdx);
  if (sampled.length === 0) {
    for (let t = 0; t < tsBegin; t++) logits[t] = -Infinity;
    return;
  }
  const lastTs = sampled[sampled.length - 1] >= tsBegin;
  const prevTs = sampled.length < 2 || sampled[sampled.length - 2] >= tsBegin;
  if (lastTs && prevTs) {
    for (let ts = tsBegin; ts < logits.length; ts++) logits[ts] = -Infinity;
  } else if (lastTs) {
    for (let t = 0; t < eosId; t++) logits[t] = -Infinity;
  }
}
```

### Verifying policy is correct

After fixing the test page, log the EOS logit before and after suppression at every step for the first 5 steps:

```javascript
log(`Step ${n}: EOS_before=${eosBefore.toFixed(2)} EOS_after=${eosAfter} top5=[...]`);
```

Expected:

- Step 0: `EOS_before=7.73 EOS_after=-Inf` (begin_suppress fires)
- Step 1: `EOS_before=8.12 EOS_after=8.12` (EOS available, begin_suppress NOT firing)
- If step 2+ has EOS > text by >0.1 → encoder precision issue after all

### Simulation script

A Node.js script at `tests/smoke/decode-policy-check.mjs` reproduces the WhisperTimestampLogitProcessor behaviour step-by-step with the actual generation_config values. It demonstrates both Scenario A (correct prompt) and Scenario B (short prompt) and shows exactly where EOS fires.

## Repos (WSL2: ~/github/asrjs/)

| Project            | Path                 | Branch                            | Notes                            |
| ------------------ | -------------------- | --------------------------------- | -------------------------------- |
| speech-recognition | `speech-recognition` | `main`                            | Main lib (feature branch merged) |
| streaming-demo     | `streaming-demo`     | `feat/streaming-restart-baseline` | NOT main                         |
| browser-demo       | `browser-demo`       | `main`                            |                                  |
| benchmark-demo     | `benchmark-demo`     | `main`                            |                                  |
| playground         | `playground`         | `main`                            |                                  |

## Architecture

```
src/
  index.ts           — Public API
  types/             — Shared types (audio, io, transcript, etc.)
  io/                — Asset loading, cache, handles
  models/            — Model families
    whisper-seq2seq/ — Whisper (vanilla + enhanced, 4-graph splitgraph)
    wav2vec2/        — Wav2Vec2 CTC (raw waveform, single graph)
    nemo-tdt/        — Parakeet TDT
    nemo-rnnt/       — RNNT
    nemo-aed/        — Canary/AED
    lasr-ctc/        — MedASR CTC
  presets/           — Model presets + catalogs
  runtime/           — Browser runtime (capture, VAD, waveform, streaming)
  alignment/         — CTC Viterbi + Wav2Vec2 aligner
  quality/           — Quality gates (compression, logprob, entropy, no-speech)
  chunking/          — VAD segmenter, drift handler, context, noise gate, binarize
  post-processing/   — Segment merge, word dedup, SRT/VTT
  ctc/               — Shared CTC decode module
  pipeline/          — Production pipeline, windowing
```

## WhisperX Runner (`tests/smoke/whisperx-runner.mjs`)

Full WhisperX-compatible transcription pipeline. Accepts all major CLI flags:

```bash
node tests/smoke/whisperx-runner.mjs \
  --model /tmp/whisper-base-4graph/fp32 \
  --language auto \
  --word_timestamps \
  --beam_size 3 \
  --patience 2.0 \
  --length_penalty 0.0 \
  --compression_ratio_threshold 2.4 \
  --logprob_threshold -1.0 \
  --temperature 0.0 \
  --temperature_increment_on_fallback 0.2 \
  --output_format vtt \
  tests/fixtures/12_dans.tr.m4a
```

**Pipeline:**

```
Audio → ffmpeg 16k mono WAV → VAD (TenVAD) → segmentAudio()
  → per-chunk: mel → encoder → whisperDecode (greedy/beam/bestOf)
  → quality gates (compression/logprob/entropy/no-speech)
  → temperature fallback [0.0-1.0] on reject (greedy mode)
  → word timestamps via decoder_align DTW
  → Wav2Vec2 CTC forced alignment (--w2v_model, post-pass, overrides DTW)
  → VTT | SRT | TXT | JSON output
```

**CLI flags fully functional:**
| Category | Flags |
|----------|-------|
| Model | `--model`, `--device` |
| Language | `--language` (code or `auto`), `--task` |
| VAD | `--vad_onset`, `--vad_offset`, `--chunk_size` |
| Decode | `--beam_size`, `--best_of`, `--patience`, `--length_penalty` |
| Temperatures | `--temperature`, `--temperature_increment_on_fallback` |
| Quality gates | `--compression_ratio_threshold`, `--logprob_threshold`, `--no_speech_threshold`, `--entropy_threshold` |
| Context | `--initial_prompt`, `--condition_on_previous_text` |
| Alignment | `--word_timestamps`/`--no-word-timestamps`, `--no_align`, `--w2v_model` |
| Output | `--output_format` (vtt/srt/txt/json), `--verbose` |

**Batch encoder:** NOT wired. ONNX batch dim IS dynamic (batch=2+ works), but benchmarking shows 0.95-1.0x speedup on CPU — no benefit. Would help on CUDA GPU. See `references/batched-encoder-investigation.md`.

Programmatic API:

```javascript
import { runAsrPipeline } from './tests/smoke/whisperx-runner.mjs';
const result = await runAsrPipeline({ model, language: 'auto', audioPath /* ... */ });
```

Full implementation details: load `references/whisperx-runner-implementation.md`
Plan doc: `docs/plans/runner-productionization.md` (tracks remaining phases)

## Verification Gate

```bash
cd ~/github/asrjs/speech-recognition
npm run typecheck
npm run lint          # 5 pre-existing max-lines warnings
npm test              # ~600 tests, 1-2 flaky timeouts
npm run build
```

## Model Verification (fp16io / new variants)

```bash
cd ~/github/asrjs/speech-recognition
node tests/smoke/verify-step1-mel.mjs            # Mel: MSE=0
node tests/smoke/verify-step2-encoder.mjs        # Encoder: cosine > 0.999
node tests/smoke/verify-step3-5-decode.mjs       # Decode: token-by-token match
```

All 3 must pass on Node ORT before WebGPU promotion. See `references/model-verification-pipeline.md`.

## Smoke Commands

```bash
# Quality Gates (26 unit tests, fast)
node tests/smoke/quality-gates-smoke.mjs

# Quality Gates + ASR integration (requires model)
WHISPER_MODEL_DIR=/tmp/whisper-base-4graph/fp32 RUN_ASR=1 node tests/smoke/quality-gates-smoke.mjs

# Model Verification Pipeline (fp16io vs fp32)
node tests/smoke/verify-step1-mel.mjs            # Mel: MSE=0
node tests/smoke/verify-step2-encoder.mjs        # Encoder: cosine > 0.999
node tests/smoke/verify-step3-5-decode.mjs       # Decode: token-by-token match

# Wav2Vec2 English
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
node tests/smoke/wav2vec2-node-wasm-align-smoke.mjs           # forced alignment
node tests/smoke/wav2vec2-node-wasm-hf-smoke.mjs              # HF download

# Wav2Vec2 Turkish
node tests/smoke/wav2vec2-node-wasm-tr-smoke.mjs

# Whisper
node tests/smoke/whisper-minimal-smoke.mjs                    # splitgraph bridge
npm run validate:whisper-base                                 # variant validation

# Verification pipeline (fp16io vs fp32)
node tests/smoke/verify-step1-mel.mjs                         # Mel: MSE=0
node tests/smoke/verify-step2-encoder.mjs                     # Encoder: cosine > 0.999
node tests/smoke/verify-step3-5-decode.mjs                    # Decode: token-by-token match

# VAD Pipeline (WhisperX-style)
node tests/smoke/vad-integration-smoke.mjs                    # basic VAD segment detection
node tests/smoke/vad-pipeline-smoke.mjs                       # full pipeline (18 tests)
RUN_ASR=1 node tests/smoke/vad-pipeline-smoke.mjs             # + ASR with gates+fallback

# large-v3-turbo — all variants (test on P520, RTX 5060 Ti)
node tests/smoke/whisper-large-v3-turbo-native.mjs            # native ORT fp32 (persistent)
WHISPER_LARGE_DIR=/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8 node tests/smoke/whisper-large-v3-turbo-native.mjs  # native ORT q8
WHISPER_LARGE_DIR=/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8 node tests/smoke/whisper-large-v3-turbo-wasm.mjs --fp32  # WASM q8
WHISPER_LARGE_DIR=/tmp/whisper-mixed-q8-enc-fp32-dec node tests/smoke/whisper-large-v3-turbo-native.mjs  # mixed precision (fastest)
node tests/smoke/benchmark-variants.mjs                        # compare all variants

# WhisperX Runner (CLI-compatible)
# Full pipeline: VAD → whisperDecode(beam=3) → quality gates → word timestamps → Wav2Vec2 align
node tests/smoke/whisperx-runner.mjs \
  --model /tmp/whisper-base-4graph/fp32 \
  --language auto \
  --word_timestamps \
  --beam_size 3 \
  --length_penalty 0.0 \
  --compression_ratio_threshold 2.4 \
  --logprob_threshold -1.0 \
  --temperature 0.0 \
  --temperature_increment_on_fallback 0.2 \
  tests/fixtures/12_dans.tr.m4a
```

## VAD Pipeline (src/chunking/vad-segmenter.ts)

WhisperX-compatible preprocessing pipeline. Four exported functions:

| Function                           | Purpose                                                                |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `vadBinarize(probs, hopSec, opts)` | Probability→binary speech/silence with hysteresis                      |
| `noiseGate(audio, opts)`           | Energy-based noise gating (opt-in, smooth crossfade)                   |
| `mergeVadSegments(segs, ...)`      | Merge+pad+cap+split. New: `overlapDurationMs`, `vadOnset`, `vadOffset` |
| `segmentAudio(audio, {vad,...})`   | Full pipeline: noise gate → VAD → merge                                |

Key params: `VadMergeConfig` (minSilenceDurationMs, speechPadMs, maxSegmentDurationMs, overlapDurationMs, vadOnset, vadOffset), `NoiseGateOptions` (noiseFloorMultiplier=2.0, windowSize=512, attenuation=0.1, smoothEdges=true), `VadBinarizeOptions` (threshold=0.5, minSpeechHops=5, minSilenceHops=10, hangoverHops=5).

Noise gate is OPT-IN in `segmentAudio()` — defaults to disabled. Energy-based VAD (TenVAD) is sensitive to noise gate artifacts; use only with model-based VAD (FireRed) or noisy environments.

Full WhisperX comparison: load `references/whisperx-pipeline-architecture.md`

### OOM / Error Recovery — Already Fixed

**Large-model OOM** was fixed via sequential lifecycle (encoder→dispose→decoders) for WASM. Native ORT (primary backend) has no heap limit. fp16 1.2GB inline weights → external data. Verified in both large-v3-turbo smokes. Not a regression task.

## Wav2Vec2 Quick Reference

**Models published** (WhisperX alignment compatible):

| Language | Model                                   | Preset Aliases                                                     | Variants     |
| -------- | --------------------------------------- | ------------------------------------------------------------------ | ------------ |
| EN       | `facebook/wav2vec2-base-960h`           | `base-960h`, `base-960h-fp16`, `base-960h-q8`                      | fp32/fp16/q8 |
| TR       | `m3hrdadfi/wav2vec2-large-xlsr-turkish` | `wav2vec2-turkish`, `wav2vec2-turkish-fp16`, `wav2vec2-turkish-q8` | fp32/fp16/q8 |

**Quantization** (native ORT benchmarks): For **Wav2Vec2**, fp16 = optimal (same WER, 2x smaller). q8 degrades Wav2Vec2 CTC accuracy + slower. For **Whisper**, q8 produces identical output to fp32 (verified on large-v3-turbo).

**HF repos**: `ysdede/wav2vec2-base-960h-onnx`, `ysdede/wav2vec2-large-xlsr-turkish-onnx`

**Export**: fp16 via PyTorch `model.half()` + export. q8 via `optimize_model` → `quantize_dynamic` (optimizer pass required first).

**Architecture**: Raw 16kHz PCM → single ONNX session → CTC logits → argmax+collapse → text. No mel, no encoder/decoder split.

**Forced alignment**: `executor.extractLogits()` → `createWav2Vec2AlignerFromLogits()` → `.align({transcript})`. ~20ms word timestamps.

**Node.js HF download**: `materializeHuggingFaceArtifacts` downloads to `/tmp/asrjs-cache/` before ORT load.

Full details: load `references/wav2vec2-implementation.md`

## Whisper Quick Reference

**Models**: whisper-base (74M), whisper-large-v3-turbo (809M). **Variants** verified per model:

- **fp32** — full precision, baseline. Available for all sizes.
- **fp16** — half precision, 2x smaller weights. Works on WebGPU. **NOT usable on native ORT CPU provider** (ORT silently rejects float16 tensors on CPU). Use fp32 on Node.js.
- **q8 (int8 dynamic)** — smallest weight size (~616MB for large-v3-turbo). **Identical output to fp32** on both native ORT and WASM. KV cache stays float32 at runtime (only weights are quantized). ~25% faster total time on native, ~43% on WASM. Verified no accuracy degradation on large-v3-turbo. The previously suspected "KV cache tensor bug" does not reproduce.

**Mixed precision (recommended)** — q8 encoder + fp32 decoder is the **fastest configuration**:

- Encoder q8: 24% faster encode (smaller weights = less memory bandwidth)
- Decoder fp32: 2x faster per-step than q8 decoder (no dequant overhead on tiny autoregressive matmuls)
- **Combined: 1.46x total speedup** over pure fp32 (6644ms vs 9708ms on JFK 11s test)
- Pre-built at `/tmp/whisper-mixed-q8-enc-fp32-dec` (symlinks; decoder needs fp32 external data files)
- Run: `WHISPER_LARGE_DIR=/tmp/whisper-mixed-q8-enc-fp32-dec node tests/smoke/whisper-large-v3-turbo-native.mjs`
- Benchmark: `node tests/smoke/benchmark-variants.mjs`
- Full analysis: `docs/quantization-research.md`

**Why fp16 decoder doesn't work on native ORT**: `onnxruntime-node` CPU provider does NOT support float16 tensor type. Even if you create an `ort.Tensor('float16', ...)` correctly (Uint16Array with float16 bit patterns), ORT silently treats it as float32. Only WebGPU backend accepts float16 natively. CUDA provider (onnxruntime-gpu) also supports float16 when available.

**4-graph splitgraph**: encoder → decoder_init → decoder_step (loop) → decoder_align. KV cache bridge requires dim preservation + prefix conversion + encoder KV preservation.

**Backend**: Native ORT for local dev (>1.5GB models). WASM for browser (1.5GB heap limit). WebGPU for browser GPU accel.

**Beam search**: `numBeams`, `lengthPenalty`, `bestOf`, `patience` in `WhisperDecodeOptions`.

Full pitfalls: load `references/whisper-architecture.md`

## Key Pitfalls (condensed)

1. **Model HF README: only model-level features** — When writing a model's HuggingFace README, describe only the model's OWN inference features: graph structure, precision variants, beam search, word timestamps (DTW), language detection, token suppression, context conditioning. Do NOT list pipeline-level features (VAD, dual gate, quality gates, streaming window, long audio). Those belong in the library's README, not the model card. Pipeline features are asr.js library capabilities, not Whisper model capabilities.

1. **Splitgraph bridge**: tensor dims must be preserved, `present.`→`past_key_values.` prefix conversion needed, encoder KV must persist across steps.
1. **Mel dimensions**: `num_mel_bins` from `generation_config.json` (128 for large, 80 for base/tiny).
1. **VRAM/OOM**: skip merged decoder for splitgraph, defer alignment. Sequential lifecycle for large models on WASM (encoder→dispose→decoders). Native ORT handles all 3 sessions persistently. Verified: OOM already fixed, not a regression item.
1. **Node.js HF**: ORT WASM can't open HTTP URLs — download to temp first (Wav2Vec2: FIXED).
1. **ONNX external data**: uploaded filenames must match internal `external_data.location` references.
1. **q8 quantization**: Whisper q8 → identical output to fp32, verified on native ORT + WASM. Wav2Vec2 q8 → degrades CTC WER (use fp16 instead for Wav2Vec2). Requires `optimize_model` pass first (Conv weight-as-initializer error).
1. **fp16 input**: models exported with fp16 expect float16 input tensors. JS needs manual float32↔float16 conversion.
1. **Package.json exports trap**: `./models/*` wildcard matches flat files only. Add explicit subpath for directory barrels.
1. **Never round up partial success**: report exact blocker, not "done" without working smoke test.
1. **Noise gate + energy VAD**: noise gate with hard attenuation (0.0) creates window-boundary discontinuities that energy-based VAD (TenVAD) misinterprets as silence gaps. Use `smoothEdges: true` + `attenuation > 0`, or only pair noise gate with model-based VAD (FireRed).
1. **`write_file` tool on Hermes**: sometimes escapes newlines as literal `\n` in the file (single-line output). Workaround: use `terminal` with heredoc (`cat > file << 'EOF'`) instead. Always re-read the file after writing to verify multi-line integrity.
1. **Task status drift**: AGENT_TASKS.md may mark tasks DONE that the user considers incomplete. Always read the user's actual request over the handover doc — the user's word is authoritative.
1. **`--model` is a Node.js option**: Running `node script.mjs --model /path` causes Node to consume `--model` as a Node flag (not passed to script). Use a config object, env vars, or the `--` separator for scripts with `--model`. Safer: use `WHISPER_MODEL_DIR` env var for model path.
1. **Language token IDs are model-specific**: Never hardcode language token IDs. Always use `tokenizer.getTokenId('<|language|>')` from the actual tokenizer. Different models (whisper-tiny vs large) may assign different IDs.
1. **`Float32Array(buffer, byteOffset, length)` uses BYTE offset, not element offset**: To extract a subarray of a Float32Array at element index `i` with length `n`, use `.subarray(i, i+n)` not `new Float32Array(data, i, n)`. The constructor's second param is byte offset (bytes = elementIndex \* 4).
1. **KV cache dims must be updated EVERY step**: The decode loop uses `kvDims` from init to construct step session feeds. But step outputs have different dims (filled KV positions grow). After each step, update `kvDims` from the step output tensors: `kvDims[k] = stepOut[k].dims`.
1. **Encoder KV must be preserved across steps**: The step model outputs ONLY decoder self-attention KV (`present.{i}.decoder.{key,value}`). Encoder cross-attention KV must be merged from the previous iteration's cache. Without this, ORT reports `input 'past_key_values.0.encoder.key' is missing in 'feeds'`.
1. **`genConfig.maxLength` is TOTAL, not max_new**: The `maxLength` in whisper `generation_config.json` is the total sequence length (prompt + generated), not the number of new tokens. Using it directly as a decode-loop bound causes `Gather` position-embedding overflow at the 449th position: `Gather node '/decoder/embed_positions/Gather' — indices element out of data bounds, idx=448`. Fix: `const maxNewTokens = (genConfig.maxLength ?? 448) - promptTokens.length - 1`.
1. **Language detection needs melProc**: When wiring language auto-detection in a raw ORT runner, melProc must be initialized BEFORE the detection block. Trying to use melProc before creation causes `ReferenceError: Cannot access 'melProc' before initialization`. Detection uses encoder→decoder_init with single SOT token, then scans logits for max language token (IDs 50259-50357).
1. **Wav2Vec2 external data loading**: Wav2Vec2 fp16/opt models have separate `.data` files. When creating the ORT session, pass `externalData: [{ path: basename(modelPath), data: fs.readFileSync(dataFile) }]` in session options. Without this, ORT throws `file not found`.
1. **ONNX shape inspection**: `sess.inputs` and `sess.outputs` return `null` in `onnxruntime-node` (they're Map-like objects, not arrays). To inspect model input/output shapes, construct a dummy tensor with the expected shape and pass to `sess.run()` — ORT will either accept it (dynamic dim) or throw a shape mismatch error (fixed dim). `sess.run()` is the only reliable shape probe.
1. **Batched encoder**: ONNX whisper encoder batch dim IS dynamic (batch=2+ works), but **only helps on GPU/CUDA**. On CPU, ORT processes batch elements sequentially, yielding 0.95-1.0x (no benefit). See `references/batched-encoder-investigation.md`.
1. **q8 Whisper quantization**: q8 (int8 dynamic) Whisper models work identically to fp32 — no accuracy degradation, same output tokens. KV cache is float32 at runtime (only weights are quantized). ~25% faster on native ORT, ~43% faster on WASM. The previously suspected "KV cache tensor bug" does not reproduce. q8 is verified on both native ORT and WASM backends.
1. **fp16 decoder on CPU ORT**: `onnxruntime-node` CPU provider does NOT support float16 tensor type. Creating `new ort.Tensor('float16', uint16Data, dims)` produces the correct tensor type in JS, but ORT silently fails at runtime with "Unexpected input data type. Actual: (tensor(float)), expected: (tensor(float16))". Only WebGPU and CUDA providers accept float16. For Node.js inference, always use fp32 decoders. Mixed precision (q8 encoder + fp32 decoder) works without issues and is the fastest option.
1. **WebGPU decoder_step 3-token failure — RESOLVED**: If WebGPU inference produces only `<|transcribe|>`→EOS or `50360→50364→EOS`, the root cause is typically ONE of: (a) prompt too short (missing task/notimestamps tokens) → decoder generates them as regular tokens, leaving only 1-2 steps for actual text; (b) missing `begin_suppress_tokens [220, 50257]` → EOS never blocked at step 0; (c) missing encoder KV cache preservation → step 1+ crashes with missing input. All three fixed in Entry 023 (2026-06-01). Full diagnostic: check `references/decode-policy-verification.md`. Dump top-5 logits at each step; compare WASM vs WebGPU; try fp32 variant.
1. **Multi-agent cross-environment test pattern**: When testing browser inference across WSL and Windows host, create `webgpu-agent-test/` on `/mnt/n/` with index.html + INSTRUCTIONS.md + AGENT_CHAT.md shared log. The Windows agent runs the browser test, saves results to `_results/`, posts findings to AGENT_CHAT.md. See `references/webgpu-agent-test-pattern.md`.
1. **Multi-agent AGENT_CHAT.md format**: When maintaining a shared log across agents (e.g., `AGENT_CHAT.md`), use flat chronological order (newest at bottom, NOT top). Each entry: `## Entry NNN — AgentName (machine, OS, GPU)` with date, raw output, verdict, analysis, next-steps. Include an HTML comment block at bottom with format instructions for new agents. Template: see `references/webgpu-agent-test-pattern.md`.

1. **Hand-rolled mel spectrogram: magnitude outer-loop bug**: When implementing DFT-based mel in pure JS, magnitude computation must be INSIDE the frame loop, not outside. Writing magnitude to ALL frames using only the LAST frame's FFT produces identical columns -> encoder sees no temporal variation -> garbage logits -> 3-token EOS. See `references/whisper-webgpu-smoke-debug-pitfalls.md` for example code.
1. **WASM does NOT support fp16**: `onnxruntime-web` WASM execution provider cannot handle `float16` tensors. Always use fp32 for WASM tests. fp16 models silently crash WASM with `ERROR: undefined`. For browser fp16, use WebGPU backend. **fp16io on WASM**: encoder loads and runs (outputs float32) but produces incorrect results — garbage transcript "a, a," instead of JFK quote. The fp16 internal compute on WASM EP is broken. fp16io is **WebGPU-only**. Verified 2026-05-31.
1. **ORT Web float16 tensor .data is numeric (not raw Uint16 bits)**: When ORT Web returns a `float16` type tensor, the `.data` property is already readable as numeric values (Float32Array-like on element access), NOT raw Uint16 float16 bit patterns. Direct copy `f32[i] = tensor.data[i]` produces correct float32 values. Do NOT use bit-conversion functions (like `float16BitsToFloat32`) on ORT Web tensor data — they interpret the native numeric values as raw bit patterns and produce NaN. The bit-conversion approach is only needed when MANUALLY constructing float16 tensors for encoder input, where you convert Float32Array → Uint16Array raw bits via `float32ToFloat16Bits()`.
1. **WebGPU fp16 NaN root cause = fp16-specific, resolved 2026-05-30**: NaN is **fp16-specific** — the **fp32 decoder works on WebGPU**. Root cause: decoder_init's optional ONNX ops (Erf 4x, Where 2x, Tile 1x, Range 2x, LessOrEqual 1x) fail on WebGPU EP with fp16 precision. **Three WebGPU pipelines fully tested**: (a) **mixf32** (q8 enc + fp32 dec) — 74.7s, short transcript "G," due to q8 encoder quantization shift; (b) **mixf16f32** (fp16 enc + fp32 dec) — **fastest at 25.5s** (encoder 2.6s vs q8's 50s), zero NaN, but **empty transcript** due to cross-precision calibration mismatch; (c) **full q8** — encoder works (ConvInteger fixed), decoder_step has zero MatMulInteger ops (all 49 MatMuls are activation-activation), hence `quantize_dynamic` barely touched it → KV cache error propagation causes 1.5B logit overflow → 200 garbage tokens. **q8 decoder on WebGPU does not work**. Transformers.js avoids all issues by using merged decoder + fp32 default on WebGPU. Full reference: `references/webgpu-fp16-nan-investigation.md` and `mlops/onnx-webgpu-dtype-bridge` skill for Cast node / calibration techniques.\n33. **q8/Mixed models on WebGPU: ORT version-dependent (RESOLVED)**: Before ORT 1.26.0, WebGPU EP did NOT support `ConvInteger`. **ORT 1.26.0 fixes ConvInteger** — q8 encoder now runs on WebGPU EP. However: (a) **q8 full model** → encoder works, decoder_step fails with 1.5B logit overflow (MatMulInteger kernel bug on WebGPU EP). (b) **Mixed (q8 enc + fp16 dec)** → encoder works, decoder NaN from fp16 ops. (c) **mixf32** (q8 enc + fp32 dec) → both work cleanly (no ConvInteger in decoder, no fp16 ops). (d) **mixf16f32** (fp16 enc + fp32 dec) → both work cleanly, fastest at 25.5s, but empty transcript from distribution mismatch. For the original mixed model dtype fix, use the `onnx-webgpu-dtype-bridge` skill.
1. **decoder_init opset analysis**: To identify suspect WebGPU ops, load the ONNX model with `onnx` Python module and count op types. Use `pip install onnx` then iterate `m.graph.node` and bucket by `n.op_type`. Compare against the ORT WebGPU supported ops list. decoder_init and decoder_step share the same suspect ops (Erf, Where, Tile, Range, LessOrEqual). See `references/webgpu-fp16-nan-investigation.md`.
1. **fp16 model encoder expects float16 input tensor**: Models exported with fp16 precision require `float16` tensor type for `input_features`. In JavaScript, construct `new ort.Tensor('float16', uint16Data, dims)` using manually-converted float32→float16 bit patterns. Sending `float32` mel causes ORT to throw: `Unexpected input data type. Actual: (tensor(float)), expected: (tensor(float16))`. On native ORT CPU provider, float16 tensors are also rejected — only WebGPU and CUDA accept fp16 inputs.
1. **Headless browser fetch limit for ONNX external data**: Headless browsers (Browserbase, Puppeteer headless) have per-request fetch limits around 1.5-2GB. fp16 encoder external data (1.2GB) works. fp32 encoder external data (2.4GB) causes `Failed to fetch`. Use a non-headless browser (real Chrome window) for large model tests.
1. **npx serve for large ONNX files (>2GB)**: Python's `http.server` (SimpleHTTP/0.6, HTTP/1.0) fails to stream files >2GB. Use `npx serve` (HTTP/1.1 with Accept-Ranges: bytes, range requests) for serving ONNX external data files to browser tests. Verify with `curl -sI http://localhost:PORT/` — look for `HTTP/1.1` and `Accept-Ranges: bytes`.
1. **ORT Web version upgrade for WebGPU EP fixes**: WebGPU EP fp16 bugs may be fixed in newer ORT versions. Check with `npm view onnxruntime-web versions --json`. Current CDN: `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.all.min.js`. After upgrading, hard-refresh the browser (Ctrl+F5) to bypass cache.
1. **Model variant dropdown triple-point update**: When adding a new model variant to the cross-validation page, ALL three of these must be updated together or you get `Model dir: undefined`: (a) `MODEL_DIRS[variant] = 'path/'`, (b) `MODEL_SIZES[variant] = 'size'`, (c) `hasExt` logic and encoder ext-data conditions in the createSession section.
   .data.

1. **q8 quantization + splitgraph = limited benefit for decoder_step**: `onnxruntime.quantization.quantize_dynamic` only quantizes MatMuls where one input is a weight initializer. In Whisper's decoder_step (splitgraph), **all 49 MatMuls are activation-activation** (attention scores, gates) — no weight initializer inputs → zero MatMulInteger ops created. Result: the q8 decoder_step is essentially fp32 (415MB inline fp32 weights). The real quantization happens only in encoder (192 MatMulInteger) and decoder_init (41 of 57 MatMuls). **q8 decoder on WebGPU EP fails** because: (a) decoder_init's MatMulInteger produces slight quantization error in KV cache; (b) decoder_step amplifies these errors via attention → 1.5B logit overflow → garbage tokens. To check if a model can be meaningfully quantized: load with `onnx` Python, count `MatMul` vs `MatMulInteger` ops, and verify which MatMul inputs are initializers vs activations. Full reference: `references/webgpu-fp16-nan-investigation.md`.\n43. **Cross-precision calibration gap — RESOLVED by fp16io**: `fp16` encoder + fp32 decoder (mixf16f32) produced early EOS because the fp16 encoder's output distribution differed from the fp32 decoder's expectation. **fp16io** (`onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`) solves this — the encoder has fp16 internals but produces fp32 output, so the decoder sees the expected dtype. **Verified 2026-05-31**: fp16io encoder output is bit-identical to fp32 on Node ORT (cosine=0.999987, MSE=4.9e-6, 27/27 tokens match). The "degraded transcript quality" noted in Entry 023 was NOT from encoder precision — it was from WebGPU decode policy bugs (wrong prompt, missing suppress_tokens, missing begin_suppress_tokens, missing encoder KV preservation). All fixed. fp16io is production-ready. See `tests/smoke/verify-step2-encoder.mjs` and `verify-step3-5-decode.mjs`.\n47. **fp16io encoder on WASM EP = garbage output** (merged): The fp16io model (fp16 internal + fp32 I/O) does NOT work on WASM execution provider. The encoder runs (~109s) and claims float32 output, but the internal fp16 ops produce corrupt hidden states — decoder generates "a, a," (5 tokens, early EOS) instead of the JFK quote. Root cause: WASM EP does not properly support float16 tensor operations internally, even when I/O types are float32. fp16io is **WebGPU-only** (or CUDA). For WASM/browser without WebGPU, use q8 (no external data, identical to fp32) or fp32. Verified 2026-05-31 on onnxruntime-web@1.26.0.

1. **WASM sequential lifecycle for large models**: When loading large Whisper models on WASM (heap limit ~1.5GB), use sequential lifecycle: load encoder → run → dispose → load decoder_init → run → dispose → load decoder_step → run → dispose. The encoder output (Float32Array) must be saved before disposal. KV cache is passed through the session wrapper between steps. Pattern: `runInit` loads decoder_step lazily on first call, then disposes decoder_init. WebGPU has no heap limit — all sessions coexist persistently. See `webgpu-agent-test/index.html` `runSingleDecode()` for implementation.

1. **WebGPU verification workflow — ground truth files, not dual-encoder browser**: Never load two encoders simultaneously in the browser — VRAM won't support it (fp32 encoder is 2.5GB, fp16io is 1.3GB). Correct workflow: (1) Run verification on Node ORT, generate ground truth (encoder outputs, token sequences, transcripts) and save to JSON files; (2) Update `webgpu-agent-test/index.html` to load those reference files; (3) Bev agent (Windows, browser) runs WebGPU test and compares against references. Single encoder per browser session only. Cross-validation mode: hardcode fp32 baseline tokens as constants (from `verify-step3-5-decode.mjs`), run variant on WebGPU, compare transcript + token-by-token.

1. **Encoder verification produces bit-identical results (fp16io vs fp32)**: Verified 2026-05-31 on Node ORT — cosine=0.999987, MSE=4.9e-6, 27/27 tokens match. The fp16io encoder's `keep_io_types=True` ensures fp32 I/O, eliminating cross-precision calibration gap. Scripts: `tests/smoke/verify-step2-encoder.mjs` (encoder comparison), `tests/smoke/verify-step3-5-decode.mjs` (full decode comparison).

1. **Browser fetch limit + IndexedDB cache workaround**: Browser `fetch()` fails for `.onnx.data` files > ~1.5-2GB (fp32 encoder 2.4GB). Solutions: (a) Use q8 variant (no external data, all inline); (b) Use library's `IndexedDbAssetCache` + `resolveAssetHandle` from `src/io/` for HuggingFace download with streaming + IndexedDB caching; (c) Use `createSpeechPipeline({ cacheModels: true })` which handles this automatically for supported presets (`onnx-community/whisper-large-v3-turbo`, etc.). For custom HF repos (`ysdede/...`), use lower-level `IndexedDbAssetCache` directly.

1. **q8 variant — no external data, best for browser/WASM**: The q8 (int8 dynamic) variant has all weights inline in `.onnx` files — no `.onnx.data`. Sizes: encoder 616MB, decoder_init 228MB, decoder_step 415MB. Total ~1.3GB. Safest option for browsers (no fetch limit) and WASM (no fp16 issues). Verified identical output to fp32.

1. **Whisper mel `n_fft=400` is intentional**: Whisper's STFT size is 400, not the 512-point NeMo/Parakeet processor size. Do not zero-pad to 512 to make radix-2 FFT easy; that changes the frequency bins and model input contract. The optimized Whisper path uses Bluestein convolution over cached 1024-point FFTs, preserves the exact 400-point DFT, reuses buffers, and skips zero filterbank spans. Benchmark command: `npm run benchmark:whisper-mel`.

1. **Custom 4-graph preset must resolve `ysdede/...`, not `onnx-community/...`**: For the WebGPU fp16 decoder work, the built-in/custom preset must point to `ysdede/whisper-large-v3-turbo-onnx-4graph` with splitgraph artifacts. If logs say `Loading onnx-community/whisper-large-v3-turbo`, the app is exercising the wrong model source even if the transcript looks plausible.

1. **`splitGraphDecodeLoop` takes a single options object with callbacks**: The pre-computed mel reference (`jfk2-mel-128.json`) was committed, then `WhisperMelProcessor` was fixed in a later commit. The reference was stale — MSE 0.25 vs expected 0.0. Always check commit order: `git log --oneline tests/smoke/jfk2-mel-128.json` vs `git log --oneline src/audio/whisper-mel.ts`. Regenerate via `scripts/whisper-webgpu-smoke.sh` or the inline generation code. See `references/mel-reference-lifecycle.md`.\n46. **Multi-backend early-EOS debugging: policy first, precision second**: When fp16io encoder + fp32 decoder produces empty transcript on WebGPU but fp32 works on Node, do NOT assume encoder distribution shift. FIRST verify the generation policy is identical: (a) prompt must include `[SOT, lang, task, notimestamps]` — a shorter prompt causes the decoder to generate task/notimestamps tokens as regular tokens, burning 2-3 decode steps; (b) `begin_suppress_tokens [220, 50257]` must be applied at the first generated token step — log EOS logit before/after to confirm; (c) `suppress_tokens` list must match `generation_config.json`; (d) encoder KV cache must be preserved across decode steps (step model only outputs decoder KV). Token sequence `50360 → 50364 → EOS` is diagnostic of a missing-prompt-tokens bug, NOT encoder precision. After policy is confirmed identical, compare fp32 vs fp16 encoder outputs (cosine similarity, per-channel stats). Methodology: `references/decode-policy-verification.md`.
1. **`splitGraphDecodeLoop` takes a single options object with callbacks**: The function signature is `splitGraphDecodeLoop({ promptTokens, encoderHiddenStates, eosTokenId, maxNewTokens, modelConfig, runInit, runStep, processLogits, ... })`. It does NOT take `(encHs, promptTokens, decInit, decStep, opts)` as positional args. `runInit` and `runStep` are async callbacks that wrap the ORT session runs and return `{ logits, vocabSize, presentKv }`. See `tests/smoke/whisper-large-v3-turbo-native.mjs` lines 119-163 for the exact callback pattern. Calling with wrong signature gives `Cannot read properties of undefined (reading 'length')` at executor.js line 79.
1. **WebGPU verification workflow — ground truth files, not dual-encoder browser**: Never load two encoders simultaneously in the browser — VRAM won't support it (fp32 encoder is 2.5GB, fp16io is 1.3GB). Correct workflow: (1) Run verification on Node ORT, generate ground truth (encoder outputs, token sequences, transcripts) and save to JSON files; (2) Update `webgpu-agent-test/index.html` to load those reference files; (3) Bev agent (Windows, browser) runs WebGPU test and compares against references. Single encoder per browser session only.
1. **Viable WebGPU combinations (verified)**:
   - ✅ **fp32 full** — baseline, works on WebGPU
   - ✅ **fp16io + fp32 decoder** — fastest (encoder 2.13s), Entry 023 milestone
   - ✅ **fp16io + fp16 decoder** — verified 2026-06-14 with custom 4-graph preset, 29.9s fixture in 5.82s transcribe
   - ✅ **mixf32 (q8 enc + fp32 dec)** — works but slow encoder (50s on WebGPU)
   - ❌ **fp16 full** — plain fp16 encoder output may still need fp16io-style calibration for stable decode
   - ❌ **mixed (q8 enc + fp16 dec)** — fp16 decoder NaN
   - ❌ **q8 full** — decoder_step MatMulInteger overflow (1.5B logits)

   Prefer fp16io encoder output for WebGPU decoder experiments; it keeps the decoder input distribution stable while still using fp16 internally.

1. **Encoder verification produces bit-identical results (fp16io vs fp32)**: Verified 2026-05-31 on Node ORT — cosine=0.999987, MSE=4.9e-6, 27/27 tokens match. The fp16io encoder's `keep_io_types=True` ensures fp32 I/O, eliminating cross-precision calibration gap. Scripts: `tests/smoke/verify-step2-encoder.mjs` (encoder comparison), `tests/smoke/verify-step3-5-decode.mjs` (full decode comparison).

## Ground-Truth Verification Workflow

**CRITICAL RULE: All decode logic lives in the library — never reimplement in test pages.**

Browser test pages (e.g. `webgpu-agent-test/index.html`) must be UI shells only:

- Load models (encoder, decoder_init, decoder_step)
- Feed Mel input (computed by library or pre-computed)
- Call library's decode loop
- Display results

Test pages MUST NOT implement their own mel, logit processor, decode loop, or KV cache management.

**Verification pipeline (step by step, Node ORT first):**

```
Step 1: Mel spectrogram     → compare library output vs reference
Step 2: Encoder output       → compare fp16io vs fp32 (cosine sim, MSE)
Step 3: Decoder init logits  → compare top-5 vs baseline
Step 4: Full decode pipeline → compare transcript vs ground truth
Step 5: Token-by-token       → compare sequence vs baseline
```

Each step must pass on Node ORT (CPU EP) before moving to the next. WebGPU promotion only after all 5 steps pass.

**Rationale:** This session spent ~150 tool calls debugging a test page that reimplemented the decode loop with 7 bugs. All 7 bugs were already handled correctly in the library's `WhisperTimestampLogitProcessor` and `splitGraphDecodeLoop`. The Node runner worked perfectly — the bugs only existed in the separately-written test page.

Full methodology: `references/model-verification-pipeline.md`

## Multi-Backend Decoding Policy Debugging

When a model variant works on one backend but produces early EOS on another, **verify generation policy first, distribution second**. Full methodology: `references/decode-policy-verification.md`.

**Diagnostic**: token sequence `50360 → 50364 → EOS` means the prompt is too short (missing task/notimestamps tokens). The decoder generates them as regular tokens, wasting steps and creating an EOS opportunity.

**Quick verification script:** `scripts/decode-policy-check.mjs` — simulates the WhisperTimestampLogitProcessor step-by-step with actual `generation_config.json` values.

**Known bugs found in WebGPU test pages:**

1. Wrong task token `50359` (translate) instead of `50360` (transcribe)
2. Wrong no_timestamps token `50363` instead of `50364` (large-v3-turbo)
3. Missing `suppress_tokens` — the ~80 special tokens never blocked
4. Missing `begin_suppress_tokens` — EOS never suppressed at step 0
5. Missing no_timestamps mode — timestamps never blocked
6. Missing sequential timestamp rules

## Dev Conventions

1. Branch → local merge → push (no PR for small fixes). PR for big changes.
2. `git add` specific files, not `-A`. No private repo names in commits.
3. TDD: RED-GREEN-REFACTOR. Focused test → implement → run full suite.
4. Model-agnostic code in standalone modules (`src/ctc/`, `src/quality/`, etc.), never in `src/models/<family>/`.
5. Follow existing patterns: NeMo TDT / Whisper executor patterns for new model families.
6. `.serena/` — do not touch.
7. Use `delegate_task` for sub-agents, not opencode CLI (WSL path mangling).
8. Update skill + docs after complex tasks (5+ tool calls). Missed docs = failure mode.

## Persistent Docs

- Roadmap: `docs/plans/asr-pipeline-roadmap.md`
- Task coordination: `docs/AGENT_TASKS.md`
- Session handover: `docs/SESSION_HANDOVER.md`
- Runner productionization: `docs/plans/runner-productionization.md`
- Wav2Vec2 progress: `docs/handoffs/flexo-wav2vec2-progress.md`
- Whisper 4-graph handoff: `docs/handoffs/whisper-4graph-export-handoff.md`
- **Quantization research**: `docs/quantization-research.md` — comprehensive analysis of q8, fp16, mixed precision, KV cache quantization, Q4, GGUF paths, and recommended roadmap
- **Handover (current state)**: `docs/HANDOVER.md` — production state, rejected experiments, profiling fix, multi-token model, optimization roadmap
- **Profiling report**: `docs/PROFILING-REPORT-2026-06-19.md` — honest baseline with encoderGpuDrainMs
- **ORT flush investigation**: `docs/ORT-FLUSH-INVESTIGATION.md` — C++ command buffer audit, fp32 sync point found
- **Edge Hunt**: `docs/EDGE-HUNT-REPORT.md` — Edge A/B/B2/C/D concluded, root cause proven
- **Optimization sprint**: `docs/OPTIMIZATION-SPRINT-REPORT.md` — P1/P1-B/P1-C with ACCEPT/REJECT/DEFER

## Critical Knowledge (2026-06-19 update)

1. **decoderInitMs was a profiling lie — FIXED**: The ~196ms in decoderInitMs was the encoder's GPU async completion time (~178ms) appearing at the first synchronization point. ORT's `Submit()` is non-blocking. Both sessions share the same `device_queue_`. The encoder cost was billed to decoder_init. Added `encoderGpuDrainMs` (gated behind `encoderGpuDrain` flag, off by default) to measure the real GPU drain. After the fix: `encoderRunMs ~185ms`, `encoderGpuDrainMs ~193ms` (when enabled), `decoderInitMs ~15ms`. Do NOT optimize decoder_init — it's only 15ms. See `docs/EDGE-HUNT-REPORT.md` and `docs/ORT-FLUSH-INVESTIGATION.md`.

2. **fp32 path looked fast because of hidden GPU flush**: `maybeCastEncoderHiddenStates()` calls `getData(true)` when casting fp32→fp16, which forces GPU pipeline flush. The ~193ms was hidden in `encoderOutputCastMs`. fp16→fp16 path (no cast) is actually more efficient — no unnecessary CPU round-trip. The cost just appears in a different metric bucket.

3. **Multi-token decoder_step DEPLOYED**: `decoder_step.onnx` now supports dynamic sequence length (input_ids dim[1] changed from 1 to dynamic). Verified K=2,4,8 with token parity. Backward-compatible. Code infrastructure ready (`runDecoderStepMultiToken()`, `secondArgmax()`). Speedup requires draft model for speculative decoding — self-speculation breaks token parity. See `docs/OPTIMIZATION-SPRINT-REPORT.md`.

4. **Closed branches (do not revisit)**: decoder_init optimization (15ms, not bottleneck), fused encoder_decoder_init (rejected +19%), shared WebGPU device (rejected +22%), GPU ArgMax for decoder_init (15ms fine), Identity/Cast graph tricks (penalty was profiling attribution), CPU pass-through for encoder (fp16 GPU optimal).

5. **encoderGpuDrain is PROFILING ONLY**: Calls `getData(false)` which adds ~18ms staging buffer overhead. Gated behind `encoderGpuDrain` flag (off by default). In production, fp16 pass-through avoids readback. Total latency identical — only metric attribution differs. URL param: `&encoderGpuDrain=1`.

6. **WebGPU benchmark URL params**: `?auto=fp16io-fp16-webgpu&local=1&gpuKv=1` for production. Add `&encoderGpuDrain=1` for honest profiling. `browser_navigate` strips URL params — use `browser_console` to set `location.href` instead.

## Benchmarking (WhisperX comparison)

Benchmark system at `~/github/ysdede/asr_benchmark_tools/`:

- Configs: `configs/catalog/models.yaml` (model definitions), `configs/suites/` (benchmark suites)
- Runtimes: `runtimes/whisperx/`, `runtimes/onnx_asr/`, `runtimes/sherpa_onnx/`, etc.
- WhisperX smoke suite: `configs/suites/tr_whisperx_large_v3_smoke.yaml`
- WhisperX uses CT2 models (not raw openai/whisper) via faster-whisper under the hood
- For asrjs benchmarks: add a new runtime at `runtimes/asrjs/` with a runtime.toml + requirements

## Skill References (on-demand, use skill_view)

| File                                           | Content                                                                                                                                                                 |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/wav2vec2-implementation.md`        | Full Wav2Vec2 specs, architecture, benchmarks, export recipes                                                                                                           |
| `references/whisper-architecture.md`           | Whisper pitfalls, backend strategy, beam search, reproducibility                                                                                                        |
| `references/whisperx-pipeline-architecture.md` | WhisperX source code study: VAD, merge, batched inference, gating, CLI params                                                                                           |
| `references/whisper-onnx-integration.md`       | ONNX integration patterns                                                                                                                                               |
| `references/whisper-onnx-4graph-export.md`     | 4-graph KV-cache export                                                                                                                                                 |
| `references/whisper-node-wasm-validation.md`   | Node/WASM variant validation                                                                                                                                            |
| `references/webgpu-agent-test-pattern.md`      | Cross-environment WebGPU test page: HTML template, INSTRUCTIONS layout, agent-parseable Z Markdown output                                                               |
| `references/whisper-beam-search.md`            | Beam search implementation                                                                                                                                              |
| `references/whisperx-runner-implementation.md` | WhisperX runner implementation: KV bridge, pitfalls, CLI parsing                                                                                                        |
| `references/batched-encoder-investigation.md`  | Batched encoder benchmark results, batch dim is dynamic but only GPU benefit                                                                                            |
|| `references/ort-flush-fence.patch`               | Fix A reference patch (C++ OnSubmittedWorkDone fence in ORT Flush)                                                                                              |
|| `references/webgpu-fp16-nan-investigation.md`  | WebGPU fp16 NaN root cause analysis: 6-variant test matrix, decoder_init ops, ConvInteger fix in ORT 1.26.0, mixf32/mixf16f32 results, q8 decoder composability failure |
| `references/verification-scripts.md`           | Encoder & decode verification scripts (verify-step2, verify-step3-5)                                                                                                    |
| `references/browser-model-loading.md`          | Fetch limits, IndexedDB cache, pipeline API, WASM sequential lifecycle                                                                                                  |
