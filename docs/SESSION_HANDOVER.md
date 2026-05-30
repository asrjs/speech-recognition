# ASR.js Whisper Engine — Session Handover

**Branch**: `feat/large-v3-turbo-fp16-external-data`
**Date**: 2026-06-01
**Agent**: Flexo (P520, WSL2, RTX 5060 Ti 8GB)

## Quick Recall

Load `asrjs-dev` skill → read `docs/AGENT_TASKS.md` → this file.

## What we accomplished

### large-v3-turbo on all 3 backends

| Backend | Precision | Lifecycle | Time | Smoke |
|---------|-----------|-----------|------|-------|
| Native ORT | fp32 | Persistent | 13.3s | `whisper-large-v3-turbo-native.mjs` |
| WebGPU | fp16 | Persistent | 24.5s | `whisper-webgpu-smoke.html` |
| WASM | fp32 | Sequential | 62s | `whisper-large-v3-turbo-wasm.mjs` |

### fp16 packaging fix
Encoder had 1.2GB inline weights → ORT WASM `std::bad_alloc`. Fixed by converting to external data with `onnx.external_data_helper.convert_model_to_external_data()`. Updated HF repos:
- `ysdede/whisper-large-v3-turbo-onnx-4graph` (original, now fixed)
- `ysdede/whisper-large-v3-turbo-onnx-4graph-v2` (backup)

### Browser externalData pitfall
Browsers cannot auto-discover `.data` files. Must fetch explicitly and pass:
```js
sessOpts.externalData = [{ path: 'encoder_model.onnx.data', data: new Uint8Array(arrayBuffer) }];
```

**Word timestamps in runner (2026-06-01):**
- `decoder_align.onnx` (4th graph) loaded for word-level timestamps
- Cross-attention DTW alignment via `processSplitGraphAlignment`
- Word boundary detection via whisper BPE token patterns
- Verified JFK: 22 words with millisecond-accurate timestamps
- `--word_timestamps` / `--no-word_timestamps` CLI flags
- `--output_format vtt|srt|txt|json` — fully wired with SRT, TXT, JSON support
- Added `--verbose` (off by default), progress always shown
- Full usage help on `--help` / no args
- All 4 quality gates (compression, logprob, entropy, no-speech) now active in `whisperx-runner.mjs`
- Temperature fallback loop: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] configurable via CLI
- Temperature-scaled sampling (not just argmax) for temp > 0
- Per-segment fallback tracking (12/93 Turkish segments triggered fallback)
- JFK verification: clean accept at t=0, perfect transcription
- Turkish 12_dans (662s): 93 segments, 890 words, 278.6s CPU processing
- Runner exports `runAsrPipeline()` for programmatic use
- Smoke: `tests/smoke/quality-gates-smoke.mjs` (26 tests)
- CLI flags: `--compression_ratio_threshold`, `--logprob_threshold`, `--no_speech_threshold`, `--entropy_threshold`, `--temperature`, `--temperature_increment_on_fallback`

### Features completed this session (2026-06-01, Flexo)

**Beam search in runner:**
- `--beam_size N` (default 1 = greedy), `--best_of N`, `--patience`, `--length_penalty`
- `--decode_type greedy|beam|best_of` for explicit mode selection
- Uses library's `whisperDecode` when beam/best_of requested
- Backward compatible: no beam params → existing greedy path

**Wav2Vec2 forced alignment in runner:**
- WhisperX-style CTC post-pass alignment
- Loads `wav2vec2-base-960h-onnx` from HF hub (fp16 or fp32)
- 16kHz preprocessing → ONNX inference → argmax → CTC collapse → word timestamps
- Falls back gracefully if Wav2Vec2 model unavailable
- `--wav2vec2_model` flag for custom path

**OOM handling verification:**
- User confirmed OOM was already fixed (sequential lifecycle, external data)
- Native ORT fp32 persistent: ✓ (3 sessions, 10.6s+1.5s, JFK perfect)
- WASM fp32 sequential: ✓ (57.1s, encoder→dispose→decoders, JFK perfect)
- Not a regression — both large-v3-turbo smokes pass clean

**VAD pipeline (2026-05-30):**
- `mergeVadSegments` enhanced: overlap support, vad_onset/vad_offset params
- `vadBinarize()`: probability→binary speech/silence with hysteresis
- `noiseGate()`: energy-based with smooth crossfade (opt-in)
- `segmentAudio()`: full pipeline wrapper
- Smoke: `tests/smoke/vad-pipeline-smoke.mjs` (18 tests)

|| # | Feature | Status | Commit |
|---|---------|--------|--------|
|| 1 | Language auto-detection | ✅ | `136ad2a` |
|| 2 | Word timestamps DTW | ✅ | Already wired |
|| 3 | bestOf decodings | ✅ | `71410b0` |
|| 4 | patience beam search | ✅ | `aceb643` |
|| 5 | VAD integration smoke | ✅ | `77778e3` |
|| 6 | SRT/VTT export | ✅ | 7 tests |
|| 7 | WebGPU model selector | ✅ | `f2a09e2` |
|| 8 | WAV2VEC2 CTC alignment | ✅ | `f7ef300` |
|| 9 | Quality gates + fallback runner | ✅ | This session |
|| 10 | Quality gates smoke (26 tests) | ✅ | This session |
||| 11 | Turkish fixture 12_dans.tr | ✅ | This session |
||| 12 | Word timestamps in runner | ✅ | This session |
||| 13 | Multiple output formats (SRT/TXT/JSON) | ✅ | This session |
||| 14 | Verbose/quiet CLI mode | ✅ | This session |
||| 15 | Language auto-detection | ✅ | This session |
||| 16 | Beam search in runner | ✅ | This session |
||| 17 | Wav2Vec2 forced alignment | ✅ | This session |

### Backend strategy (established)
1. **Native ORT** (`onnxruntime-node`) — first dev target, no heap limit, streaming-ready
2. **WebGPU** (browser) — production browser target, needs explicit externalData
3. **WASM** (fallback) — ~1.5GB heap limit, sequential only for large models

## Project Structure

```
speech-recognition/
  src/
    models/whisper-seq2seq/
      core.ts              — decode loops (greedy, beam, bestOf, patience)
      executor.ts          — ORT bridge, splitgraph, language detection
      enhanced-executor.ts — production pipeline (VAD+gates+fallback+drift+merge)
    quality/               — quality gates (compression, logprob, entropy, no-speech)
    chunking/              — VAD backends (TenVAD, FireRed), drift, context
    post-processing/       — segment merge, word dedup, format, SRT/VTT
    alignment/             — CTC Viterbi, WAV2VEC2 aligner
    pipeline/              — ProductionWhisperPipeline
  tests/smoke/
    whisper-large-v3-turbo-native.mjs    — Native ORT persistent smoke
    whisper-large-v3-turbo-wasm.mjs      — WASM sequential smoke
    whisper-e2e-pipeline-smoke.mjs       — Full pipeline (encoder→gates→fallback)
    whisper-webgpu-smoke.html            — WebGPU browser smoke (model selector)
    whisper-bestof-smoke.mjs             — bestOf decodings
    vad-integration-smoke.mjs            — TenVAD energy-based VAD
    wav2vec2-node-wasm-smoke.mjs         — WAV2VEC2 ASR
    wav2vec2-node-wasm-align-smoke.mjs   — WAV2VEC2 alignment
  docs/
    AGENT_TASKS.md          — Task coordination (source of truth)

streaming-demo/             — React app, streaming ASR + VAD (TenVAD/FireRed)
browser-demo/               — Upload/sample-file demo
benchmark-demo/             — Performance benchmarks
vad-demo/                   — Isolated VAD testing
```

## Remaining

| Task | Effort | Notes |
|------|--------|-------|
| Batched encoder | Large | Needs encoder ONNX to accept [N, mel, 3000] |
| q8 KV cache fix | Medium | ORT-level quantization defect on large-v3-turbo |
| q4/q4f16 | Large | Experimental quantization |
| loadSpeechModel fix | Medium | Direct-source path has URL/path wiring issue |

## Verification

```bash
cd ~/github/asrjs/speech-recognition
npm run typecheck && npm run lint && npm test   # 601 tests
npm run build
node tests/smoke/quality-gates-smoke.mjs         # 26 unit tests (fast)
RED_ASR=1 node tests/smoke/quality-gates-smoke.mjs  # + ASR integration
node tests/smoke/whisper-e2e-pipeline-smoke.mjs
node tests/smoke/whisper-large-v3-turbo-native.mjs
node tests/smoke/vad-integration-smoke.mjs
node tests/smoke/vad-pipeline-smoke.mjs          # 18 VAD tests
node tests/smoke/whisper-bestof-smoke.mjs
node tests/smoke/wav2vec2-node-wasm-align-smoke.mjs

# Turkish fixture (WhisperX-compatible runner):
WHISPER_MODEL_DIR=/tmp/whisper-base-4graph/fp32 node tests/smoke/whisperx-runner.mjs \
  --language tr --vad_onset 0.5 \
  tests/fixtures/12_dans.tr.m4a
```
