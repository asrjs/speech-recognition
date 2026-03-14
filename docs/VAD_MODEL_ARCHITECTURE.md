# VAD (Voice Activity Detection) Model Architectures

Documentation of VAD model families for asr.js design reference. **Architecture docs only — no implementation.**

---

## Model Family Overview

| Family | Source | Frontend | Encoder | Output | Runtime |
|--------|--------|----------|---------|--------|---------|
| **TEN VAD** | [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) | — | Inference engine (static lib) | prob + flag per frame | WebAssembly |
| **FireRedVAD** | FireRedASR2S | Kaldi mel (80) | DFSMN | prob per frame | PyTorch / ONNX |
| **Silero-VAD** | Silero Team | Raw / mel | CNN/LSTM | prob per frame | PyTorch / ONNX |
| **WebRTC-VAD** | Chromium | Raw (8/16 kHz) | GMM | binary | C |
| **TEN-VAD (onnx-asr)** | onnx-asr | — | — | segments | ONNX |

---

## 1. TEN VAD

**Source:** [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad), [lib/Web](https://github.com/TEN-framework/ten-vad/tree/main/lib/Web)

**Topology:** Frame-based VAD with a private inference engine, compiled to WebAssembly for browser/worker use.

### Pipeline

```
Raw waveform (int16)
  │
  ▼ ten_vad_process(handle, audioPtr, audioLen, probPtr, flagPtr)
  │ hopSize samples per frame (e.g. 256 → ~16ms at 16kHz)
  │ threshold [0, 1] for binary decision
  ▼
  probability [0, 1], isVoice (0/1)
```

### API (C / Emscripten)

| Function | Purpose |
|----------|---------|
| `ten_vad_create(handlePtr, hopSize, threshold)` | Create instance |
| `ten_vad_process(handle, audioPtr, len, outProbPtr, outFlagPtr)` | Process one frame |
| `ten_vad_destroy(handlePtr)` | Release |
| `ten_vad_get_version()` | Version string |

### WebAssembly Build

- **Static library:** `libInferEngine.a` (inference engine)
- **WASM module:** `ten_vad.wasm` (~278KB) + `ten_vad.js` (glue)
- **Factory:** `createVADModule()` → `Promise<TenVADModule>`
- **Target:** `ENVIRONMENT='web,worker'`, `ALLOW_MEMORY_GROWTH=1`

### Notes

- No external mel frontend; model accepts raw waveform chunks.
- Frame-by-frame (streaming-friendly).
- Emscripten C API with `EMSCRIPTEN_KEEPALIVE`; no Embind.

---

## 2. FireRedVAD

**Source:** [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) — `fireredasr2s/fireredvad`

**Topology:** DFSMN (Deep Feed-Forward Sequential Memory Network) on Kaldi fbank. Non-streaming and streaming variants.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ AudioFeat (Kaldi fbank 80 bins, CMVN)
  ▼
  [B, T, 80]
  │
  ▼ DetectModel (DFSMN + Linear)
  │ DFSMN: Rx[H-P(N1,N2,S1,S2)]-MxH
  │   FSMN blocks with lookback (N1,S1) / lookahead (N2,S2)
  ▼
  ▼ sigmoid(logits) → probs [0,1]
  ▼
  ▼ VAD postprocessor (smooth, threshold, min/max frames)
  ▼
  timestamps (start, end) or frame-level probs
```

### DFSMN Config

- **R:** number of DFSMN blocks
- **M:** DNN layers
- **H:** hidden size
- **P:** projection size
- **N1, S1:** lookback order and stride
- **N2, S2:** lookahead order and stride (0 for causal streaming)

### Streaming

- `FireRedStreamVad` uses `caches` in DFSMN forward for incremental computation.
- No attention; efficient for real-time.

### mVAD (Multi-label)

- Multiple output heads: speech / singing / music.
- Separate thresholds per class.
- Lightweight Audio Event Detection (AED).

---

## 3. Silero-VAD

**Source:** [silero-vad](https://github.com/snakers4/silero-vad)

**Topology:** Small neural net on raw or mel-like features. Frame-based or segment-based.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ Model (CNN/LSTM or similar)
  │ ~16ms windows typical
  ▼
  probability [0, 1]
  ▼
  Optional: segment merging, min/max duration
  ▼
  segments (start, end)
```

### Notes

- Widely used; ONNX export available.
- Used in onnx-asr, FunASR, etc.
- 8 kHz and 16 kHz variants.

---

## 4. WebRTC-VAD

**Source:** Chromium / WebRTC

**Topology:** GMM-based (Gaussian Mixture Model), no neural network. Very lightweight.

### Pipeline

```
Raw waveform (8 or 16 kHz, 10/20/30 ms frames)
  │
  ▼ WebRTC VAD (GMM)
  │ Aggressiveness 0–3
  ▼
  binary (0: no voice, 1: voice)
```

### Notes

- No probabilities; binary only.
- Low latency, minimal compute.
- Often used as a pre-filter before ASR.

---

## Frontend Comparison (VAD)

| Family | Frontend | Input |
|--------|----------|-------|
| **TEN VAD** | — (raw in model) | int16, hopSize samples |
| **FireRedVAD** | Kaldi mel (80) | [B, T, 80] |
| **Silero-VAD** | Raw or internal | waveform |
| **WebRTC-VAD** | Raw | 8/16 kHz PCM |

---

## Output Modes

| Mode | Families | Use case |
|------|----------|----------|
| **Frame-level prob** | TEN VAD, FireRedVAD, Silero-VAD | Custom thresholding, streaming |
| **Frame-level binary** | TEN VAD, WebRTC-VAD | Simple on/off |
| **Segments** | FireRedVAD, Silero-VAD (postproc) | ASR input chunking |

---

## Cross-Reference

- **FireRedASR2S (VAD section):** [FIRERED_ASR2S_MODEL_ARCHITECTURE.md](./FIRERED_ASR2S_MODEL_ARCHITECTURE.md)
- **Master index:** [ASR_MODEL_FAMILY_INDEX.md](./ASR_MODEL_FAMILY_INDEX.md)
