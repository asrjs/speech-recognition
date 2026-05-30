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

### Features completed this session

| # | Feature | Status | Commit |
|---|---------|--------|--------|
| 1 | Language auto-detection | ✅ | `136ad2a` |
| 2 | Word timestamps DTW | ✅ | Already wired |
| 3 | bestOf decodings | ✅ | `71410b0` |
| 4 | patience beam search | ✅ | `aceb643` |
| 5 | VAD integration smoke | ✅ | `77778e3` |
| 6 | SRT/VTT export | ✅ | 7 tests |
| 7 | WebGPU model selector | ✅ | `f2a09e2` |
| 8 | WAV2VEC2 CTC alignment | ✅ | `f7ef300` |

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
node tests/smoke/whisper-e2e-pipeline-smoke.mjs
node tests/smoke/whisper-large-v3-turbo-native.mjs
node tests/smoke/vad-integration-smoke.mjs
node tests/smoke/whisper-bestof-smoke.mjs
node tests/smoke/wav2vec2-node-wasm-align-smoke.mjs
```
