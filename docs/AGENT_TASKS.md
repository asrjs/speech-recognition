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
- [x] Quality gates: compression, logprob, entropy, no-speech
- [x] Temperature fallback [0.0, 0.2, ..., 1.0]
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
- `tests/smoke/whisper-large-v3-turbo-native.mjs` — native ORT persistent
- `tests/smoke/whisper-large-v3-turbo-wasm.mjs` — WASM sequential
- `tests/smoke/whisper-e2e-pipeline-smoke.mjs` — full pipeline (encoder→gates→fallback)
- `tests/smoke/whisper-webgpu-smoke.html` — WebGPU with model selector
- `tests/smoke/whisper-bestof-smoke.mjs` — bestOf decodings
- `tests/smoke/wav2vec2-node-wasm-smoke.mjs` — WAV2VEC2 ASR
- `tests/smoke/wav2vec2-node-wasm-align-smoke.mjs` — WAV2VEC2 alignment

## REMAINING TASKS (priority order)

### 1. Language Auto-Detection ✅ DONE
Commit: `136ad2a`. Runs decoder_init with single `<|startoftranscript|>` token, reads language from first logit position. Wired in `transcribeWithSplitGraph` when `language='auto'`. Falls back to `config.languages[0] ?? 'en'`.

### 2. Word Timestamps via Cross-Attention DTW ✅ DONE
Already implemented in `computeAttentionWordTimestampsSplitGraph` + `runForcedAlignmentSplitGraph` + `processSplitGraphAlignment`. Wired through `transcribeWithSplitGraph` → `EnhancedWhisperExecutor` → `ProductionWhisperPipeline`. Enable with `returnWordTimestamps: true`.

### 3. Batched Encoder Processing ⏳ DEFERRED
2-3x faster for long audio. Requires encoder to accept batched input [N, mel, 3000].
Large effort — needs ONNX graph changes and executor refactor.

### 4. VAD Integration — WhisperX-style Pipeline ✅ DONE
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

### 5. SRT/VTT Export ✅ DONE
`ProductionWhisperPipeline` generates SRT/VTT via `generateSubtitles()`. 7 unit tests pass.

## Quantization Roadmap (deferred)

- **q8 (1.4GB)**: KV cache tensor bug on large-v3-turbo (ORT-level defect). Validated on whisper-base.
- **q4/q4f16**: Experimental. Needs opset research + WebGPU validation.
- **Mixed precision**: Deferred until dtype boundary + KV cache validated.
- **Priority**: fp32/fp16 correctness first. Quantization is deployment optimization.

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
