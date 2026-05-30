# Agent Task Coordination

Branch: `feat/large-v3-turbo-fp16-external-data`
Updated: 2026-06-01 (Flexo)

## Context Recovery

**Primary skill**: Load `asrjs-dev` skill first
**Progress file**: `docs/handoffs/flexo-wav2vec2-progress.md`
**HF models**: `ysdede/whisper-large-v3-turbo-onnx-4graph` (original, fixed fp16) + v2 backup
**Local models**: `/tmp/whisper-base-4graph/` (fast smoke), `/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/` (large smoke)

## Backend Strategy (FINAL)

| Priority | Backend | Model | Lifecycle | Time | Notes |
|----------|---------|-------|-----------|------|-------|
| **1st** | `onnxruntime-node` (native) | fp32 | Persistent | 13.3s | Streaming-ready, no heap limit |
| 2nd | WebGPU (browser) | fp16 | Persistent | 24.5s | Needs explicit externalData |
| Fallback | ORT Web/WASM | fp32 | Sequential | 62s | ~1.5GB heap → one session at a time |

**WASM heap limit (~1.5GB)**: Even with external data, WASM can only hold ONE large session. Use sequential lifecycle (encoder→dispose→decoders→dispose) for WASM. Native ORT and WebGPU load all sessions simultaneously.

**Browser externalData**: Must fetch `.data` files and pass `externalData: [{path, data: Uint8Array}]` in session options. Browser cannot auto-discover co-located files.

**fp16 dtype**: Encoder expects float16 input. Node.js lacks native Float16Array → fp32 for Node dev, fp16 for WebGPU.

## COMPLETED TASKS

### Core Pipeline ✅
- [x] Greedy decode + Beam search + bestOf + patience
- [x] Long audio windowing (30s windows, 84.8% overlap)
- [x] Timestamp tokens + token suppression
- [x] Context conditioning (extraPromptTokens)
- [x] Quality gates: compression, logprob, entropy, no-speech (src/quality/ + wired in runner)
- [x] Temperature fallback [0.0, 0.2, ..., 1.0] with temperature sampling
- [x] SRT/VTT subtitle export
- [x] Mel dimension auto-detect (numMelBins from manifest)
- [x] VRAM optimization (skip merged decoder, defer alignment)
- [x] URL/path unified (fetchText handles bare paths + file://)
- [x] Splitgraph KV cache bridge fix
- [x] CTC Viterbi alignment + WAV2VEC2 aligner
- [x] WAV2VEC2 model factory + presets + smoke
- [x] EnhancedWhisperExecutor: VAD+gates+fallback+drift+context+merge
- [x] ProductionWhisperPipeline: formatTranscript + SRT/VTT + metrics

### large-v3-turbo Validation ✅
- [x] Native ORT fp32 persistent smoke: 13.3s, perfect JFK
- [x] WebGPU fp16 persistent smoke: 24.5s, perfect JFK
- [x] WASM fp32 sequential smoke: 62s, perfect JFK
- [x] fp16 encoder fixed: inline→external data (0.4MB graph + 1.2GB data)
- [x] HF repos updated with fixed fp16

### Smoke Tests
- `tests/smoke/quality-gates-smoke.mjs` — quality gates (26 tests, RED_ASR=1 for integration)
- `tests/smoke/whisper-large-v3-turbo-native.mjs` — native ORT persistent
- `tests/smoke/whisper-large-v3-turbo-wasm.mjs` — WASM sequential
- `tests/smoke/whisper-e2e-pipeline-smoke.mjs` — full pipeline (encoder→gates→fallback)
- `tests/smoke/whisper-webgpu-smoke.html` — WebGPU with model selector
- `tests/smoke/whisper-bestof-smoke.mjs` — bestOf decodings
- `tests/smoke/wav2vec2-node-wasm-smoke.mjs` — WAV2VEC2 ASR
- `tests/smoke/wav2vec2-node-wasm-align-smoke.mjs` — WAV2VEC2 alignment
- `tests/smoke/vad-pipeline-smoke.mjs` — 18 VAD tests (VAD integration with WhisperX-style pipeline)

## REMAINING TASKS (priority order)

### 0. Error Recovery / OOM Handling ✅ DONE (Legacy)
**OOM already fixed**: The large-model OOM issue (ORT WASM heap limit) was fixed by the sequential lifecycle pattern:
- encoder → run → dispose, then decoders → run → dispose
- Peak memory ~max(encoder, decoders) instead of sum(encoder, decoders)
- Native ORT (primary backend) has no OOM issue — all 3 sessions load persistently
- WASM fallback uses sequential lifecycle (verified in large-v3-turbo-wasm smoke)
- fp16 fix: 1.2GB inline weights → external data (no more std::bad_alloc on WASM)
- Not a regression — working in both existing smoke tests

### 1. Language Auto-Detection ✅ DONE
Commit: `136ad2a`. Runs decoder_init with single `<|startoftranscript|>` token, reads language from first logit position. Wired in `transcribeWithSplitGraph` when `language='auto'`. Falls back to `config.languages[0] ?? 'en'`.

### 2. Word Timestamps via Cross-Attention DTW ✅ DONE
Already implemented in `computeAttentionWordTimestampsSplitGraph` + `runForcedAlignmentSplitGraph` + `processSplitGraphAlignment`. Wired through `transcribeWithSplitGraph` → `EnhancedWhisperExecutor` → `ProductionWhisperPipeline`. Enable with `returnWordTimestamps: true`.

### 3. Batched Encoder Processing ⏳ DEFERRED
2-3x faster for long audio. Requires encoder to accept batched input [N, mel, 3000].
Large effort — needs ONNX graph changes and executor refactor.

### 5. Beam Search in Runner ✅ DONE (2026-06-01, Flexo)
Commit: `03707b4`. Wired into `whisperx-runner.mjs`:
- `--beam_size N` (default 1 = greedy). Uses library's greedy decode at beam_size=1 for backward compat.
- `--best_of N` (default 1). Added `whisperDecode` import for best_of / decode_type branching.
- `--patience 2.0` — beam expansion patience
- `--length_penalty 0.0` — length penalty for beam / best_of
- `--decode_type greedy|beam|best_of` — explicit decode mode selection
- Fallback: when no beam search params given, uses existing greedy path (no behavior change)

### 5. Wav2Vec2 Forced Alignment in Runner ✅ DONE (2026-06-01, Flexo)
Commit: `9f62c42`. WhisperX-style CTC forced alignment:
- Loads `wav2vec2-base-960h-onnx` model from HF hub (fp16 default, falls back fp32)
- Preprocesses audio to wav2vec2 feature format (16kHz mono, padding/truncation)
- Runs Wav2Vec2 ONNX inference via ort-node → logits → argmax → CTC collapse
- Produces word-level timestamps using token alignment + Viterbi-like decoding
- Integrated as post-pass after whisper transcription (WhisperX semantic)
- Configurable via `--wav2vec2_model` flag
- Falls back gracefully if Wav2Vec2 model not available
  
Smoke verified: JFK English returns correct alignment for all 22 words.

### 6. VAD Integration — WhisperX-style Pipeline ✅ DONE
Enhanced 2026-05-30 (Flexo). The basic smoke (commit `77778e3`) detected segments.
Now expanded into full WhisperX-compatible VAD preprocessing pipeline:

**Added to `src/chunking/vad-segmenter.ts`:**
- `vadBinarize()` — probability→binary speech/silence with hysteresis (onset/offset confirm, hangover)
- `noiseGate()` — energy-based noise gating with smooth crossfade (opt-in)
- `mergeVadSegments()` — enhanced with overlap support (`overlapDurationMs`), `vadOnset`/`vadOffset` params
- `segmentAudio()` — full pipeline wrapper: optional noise gate → VAD → merge+pad+overlap

**Smoke test**: `tests/smoke/vad-pipeline-smoke.mjs` — 18 tests covering:
- Noise gate (silence attenuation, speech preservation, SNR improvement)
- VAD binarization (all-silence, all-speech, mixed speech→silence→speech)
- TenVAD energy-based segmentation
- mergeVadSegments with and without overlap
- segmentAudio() full pipeline (with noise gate opt-in)
- (Optional) Full ASR pipeline with quality gates + temperature fallback (RUN_ASR=1)

**WhisperX parity gaps closed:**
- Overlap between consecutive chunks ✓
- VAD probability binarization with hysteresis ✓
- Noise floor gating for noisy environments ✓
- vad_onset/vad_offset parameters exposed ✓

### 5b. Quality Gates Wired into Runner ✅ NEW 2026-06-01

The `whisperx-runner.mjs` now fully integrates all 4 quality gates with temperature fallback:

| Component | File | Details |
|-----------|------|---------|
| compressionRatioGate | `src/quality/compression-ratio.ts` | deflate-based, default threshold 2.4 |
| logProbGate | `src/quality/log-probability.ts` | per-token log prob avg, default -1.0 |
| entropyGate | `src/quality/entropy.ts` | avg distribution entropy, default 2.4 nats |
| noSpeechGate | `src/quality/no-speech.ts` | token 50362 prob + avgLogProb, default 0.6 |
| temperature-fallback | `src/quality/temperature-fallback.ts` | scaled loop [0.0, 0.2, ..., 1.0] |
| Runner | `tests/smoke/whisperx-runner.mjs` | exports `runAsrPipeline()`, all CLI flags |
| Smoke tests | `tests/smoke/quality-gates-smoke.mjs` | 26 unit + optional ASR integration |

**Accepted WhisperX CLI flags (now fully functional):**
`--compression_ratio_threshold`, `--logprob_threshold`, `--no_speech_threshold`, `--entropy_threshold`, `--temperature`, `--temperature_increment_on_fallback`

**Temperature sampling** for temp > 0: applies `logits / temperature` scaling before softmax, then samples from the distribution.

### 5. SRT/VTT Export ✅ DONE
`ProductionWhisperPipeline` generates SRT/VTT via `generateSubtitles()`. 7 unit tests pass.

## Quantization Roadmap (deferred)

- **q8 (1.4GB)**: KV cache tensor bug on large-v3-turbo (ORT-level defect). Validated on whisper-base.
- **q4/q4f16**: Experimental. Needs opset research + WebGPU validation.
- **Mixed precision**: Deferred until dtype boundary + KV cache validated.
- **Priority**: fp32/fp16 correctness first. Quantization is deployment optimization.

## Stale Items Cleanup

The following items from earlier docs were **already fixed** as of this session:
- **loadSpeechModel fix** — URL/path wiring issue fixed in commits `87e5e6a` + `0ceb405`. `fetchText` now handles bare file paths, `file://` URLs, and HTTP URLs. Verified: `loadSpeechModel({modelId:'base', preset:'whisper'})` works with and without `useManifestSources`.
- **OOM handling** — Fixed via sequential lifecycle (encoder→dispose→decoders) for WASM + external data for fp16. Native ORT has no heap limit. Verified both large-v3-turbo smokes pass.

## Verification

```bash
npm run typecheck && npm run lint && npm test          # 103 files, 601 tests
npm run build
node tests/smoke/whisper-e2e-pipeline-smoke.mjs        # Full pipeline
node tests/smoke/whisper-large-v3-turbo-native.mjs     # Native ORT persistent
node tests/smoke/whisper-large-v3-turbo-wasm.mjs --fp32 # WASM sequential
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
node tests/smoke/whisper-bestof-smoke.mjs
```

## Shared Files (coordinate before modifying)

- `src/models/whisper-seq2seq/core.ts` — decode loops (greedy, beam, bestOf, patience)
- `src/models/whisper-seq2seq/executor.ts` — ORT bridge, splitGraphDecodeLoop, transcribeWithSplitGraph
- `src/models/whisper-seq2seq/enhanced-executor.ts` — production pipeline
- `src/quality/` — quality gates (model-agnostic)
- `src/chunking/` — VAD, drift, context (model-agnostic)
- `src/post-processing/` — merge, format, subtitles
- `src/alignment/` — CTC Viterbi, WAV2VEC2 aligner
- `src/pipeline/` — ProductionWhisperPipeline

## WAV2VEC2 Quantization — DONE (Flexo-deepseek-v4-pro)

EN+TR models published in 3 variants each. Benchmark results (native ORT, P520):

| Model | fp32 | fp16 | q8 | Optimal |
|-------|------|------|-----|---------|
| EN base-960h (JFK 11s) | 362MB/704ms/4.5% | 182MB/769ms/4.5% | 91MB/2291ms/9.1% | fp16 |
| TR large-xlsr (18.6s) | 1207MB/5626ms/53.6% | 605MB/3856ms/53.6% | 302MB/5888ms/71.4% | fp16 |

Key finding: fp16 = identical WER to fp32, 2x smaller. q8 degrades accuracy AND is slower.
Use fp16 as default.

Export: fp16 via PyTorch model.half() + export. q8 via optimize_model → quantize_dynamic.
Pitfall: q8 requires optimizer pass first (Conv weight-as-initializer error otherwise).

Commits: `0d7d2fb` (q8), `08a5def` (fp16+benchmark), `7a329b9` (EN fp16/q8 presets)
HF: ysdede/wav2vec2-base-960h-onnx, ysdede/wav2vec2-large-xlsr-turkish-onnx
