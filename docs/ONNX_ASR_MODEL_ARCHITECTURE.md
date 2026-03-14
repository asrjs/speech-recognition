# ONNX ASR Model Families

Reference: `N:\github\ysdede\onnx-asr` — models supported for ONNX inference (Kaldi/Vosk, GigaAM, T-one, NeMo, Whisper).

This doc complements the NeMo, MedASR, and Whisper architecture docs by documenting **onnx-asr**–specific models and how they map to the shared layer taxonomy.

---

## Model Family Overview

| Family | Origin | Topology | Frontend | Encoder | Notes |
|--------|--------|----------|----------|---------|-------|
| **Kaldi / Vosk** | Alpha Cephei, k2-fsa | Stateless RNN-T | Kaldi mel | Zipformer | encoder + decoder + joiner (3 ONNX) |
| **GigaAM** | GigaChat (Sber) | CTC, RNN-T | GigaAM mel (64 bins) | Conformer | v2/v3, E2E variants |
| **T-one** | T-Tech | CTC | **Identity** (raw) | Chunk-based | 8 kHz, streaming state |
| **NeMo** | NVIDIA | CTC, RNNT, TDT, AED | NeMo mel (80/128 bins) | FastConformer | Parakeet, Canary |
| **Whisper** | OpenAI | AED | Whisper mel | Transformer | see WHISPER doc |

---

## 1. Kaldi / Vosk (Zipformer + Stateless RNN-T)

**Source:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [Alpha Cephei Vosk](https://alphacephei.com/vosk/)

**Topology:** Stateless RNN-T — decoder uses **context window** (last N tokens), not LSTM state.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ KaldiPreprocessor
  │ STFT, Kaldi mel (20–7600 Hz), Hann^0.85, snip_edges
  │ n_fft=512, win=400, hop=160, 80 bins
  ▼
[B, T, 80]  (Kaldi layout: time×mel; NeMo uses mel×time [B,80,T])
  │
  ▼ encoder.onnx
  │ Zipformer (Icefall)
  ▼
  ▼ decoder.onnx (context-based)
  │ context_size = 2
  ▼
  ▼ joiner.onnx
  │ joint(encoder_out, decoder_out) → logit
  ▼
  Greedy transducer decoding
```

### Key Differences from NeMo RNNT

- **Decoder:** Stateless — no LSTM; uses embedding of last `CONTEXT_SIZE` tokens. State = `dict[context_tuple, decoder_out]` (cache).
- **ONNX split:** 3 files (encoder, decoder, joiner) vs NeMo's decoder_joint fused.
- **Subsampling:** Typically 4× (configurable).

### Frontend (Kaldi)

- `win_length=400`, `hop_length=160`, `n_fft=512`
- Mel: Kaldi scale, `low_freq=20`, `high_freq=-400` (i.e. 7600 Hz)
- Window: Hann^0.85
- Preemphasis 0.97, remove_dc_offset, optional dither
- `snip_edges=True` or `False` (symmetric pad when False)

---

## 2. GigaAM (GigaChat / Sber)

**Source:** [GigaAM](https://github.com/salute-developers/GigaAM)

**Topology:** CTC or RNN-T (LSTM decoder). E2E variants include frontend in model.

### Pipeline

**CTC:**
```
Raw waveform [16 kHz]
  │
  ▼ GigaamPreprocessorV2/V3
  │ 64 mel bins, n_fft 400/320, hop 160
  │ v2: Slaney; v3: different win/mel
  ▼
[B, 64, T]
  │
  ▼ model.onnx (encoder + CTC head fused)
  ▼
  CTC greedy
```

**RNN-T:**
```
  ▼ encoder.onnx → decoder.onnx → joint.onnx
  │ LSTM decoder (state: h, c), pred_hidden=320
  │ max_tokens_per_step=3
  ▼
  Transducer greedy
```

### Frontend (GigaAM)

| Param | v2 | v3 |
|-------|----|----|
| n_fft | 400 | 320 |
| win_length | 400 | 320 |
| hop_length | 160 | 160 |
| n_mels | 64 | 64 |
| f_max | 8000 | 8000 |
| mel_scale | — | bfloat16 in export |

**Features layout:** `[B, 64, T]` (mel × time).

---

## 3. T-one (T-Tech)

**Source:** [T-one](https://github.com/voicekit-team/T-one)

**Topology:** CTC, **no separate frontend** — model accepts **raw waveform chunks** (2400 samples). 8 kHz. Streaming with recurrent state.

### Pipeline

```
Raw waveform [8 kHz]
  │
  ▼ Chunk into 2400-sample segments (+ padding)
  │ NO mel; signal × (2^15-1) as int32 input
  │ state: [B, 219729] float16 (recurrent)
  ▼
  ▼ model.onnx (encoder + CTC fused, with state in/out)
  ▼
  CTC greedy
```

**Subsampling:** `reduction_kernel_size` from config.

**Preprocessor:** `"identity"` — raw waveform, no STFT/mel.

---

## 4. NeMo (onnx-asr mapping)

Same as `NEMO_ASR_MODEL_ARCHITECTURE.md`. onnx-asr uses:

- **Preprocessor:** `NemoPreprocessor80` or `NemoPreprocessor128` (80/128 mel bins)
- **Models:** `NemoConformerCtc`, `NemoConformerRnnt`, `NemoConformerTdt`, `NemoConformerAED`
- **Subsampling:** 8× (FastConformer)
- **RNN-T:** decoder_joint fused (single ONNX for decoder+joint)

---

## 5. Whisper (onnx-asr mapping)

See `WHISPER_ASR_MODEL_ARCHITECTURE.md`. onnx-asr uses Whisper feature extractor (log10, clamp, scale) and ONNX export from onnxruntime or optimum.

---

## Frontend Comparison (onnx-asr preprocessors)

| Preprocessor | sample_rate | n_fft | win | hop | n_mels | mel_scale | norm |
|--------------|-------------|-------|-----|-----|--------|-----------|------|
| **Kaldi** | 16000 | 512 | 400 | 160 | 80 | Kaldi | — |
| **Nemo80** | 16000 | 512 | 400 | 160 | 80 | Slaney | per_feature |
| **Nemo128** | 16000 | 512 | 400 | 160 | 128 | Slaney | per_feature |
| **GigaAM v2** | 16000 | 400 | 400 | 160 | 64 | — | — |
| **GigaAM v3** | 16000 | 320 | 320 | 160 | 64 | — | — |
| **Whisper** | 16000 | 400 | — | 160 | 80 | Slaney | log10+clamp+scale |
| **T-one** | **8000** | — | — | — | — | **Identity** | — |

---

## ONNX Session Layout per Family

| Family | Files | Inputs |
|--------|-------|--------|
| Kaldi | encoder, decoder, joiner | encoder: x [B,T,80], x_lens |
| GigaAM CTC | model | features [B,64,T], feature_lengths |
| GigaAM RNN-T | encoder, decoder, joint | encoder: audio_signal [B,64,T], length |
| T-one | model | signal [B,2400,1], state [B,219729] |
| NeMo CTC | model | audio_signal [B,80,T], length |
| NeMo RNNT/TDT | encoder, decoder_joint | encoder: audio_signal, length |
| NeMo AED | encoder, decoder | encoder: audio_signal, length |
| Whisper | model | input_features [B,80,T] |

---

## Reusable Module Mapping for asr.js

When porting onnx-asr to JS/WebGPU:

| onnx-asr | asr.js module |
|----------|---------------|
| KaldiPreprocessor | processors/kaldi-mel.ts |
| NemoPreprocessor80/128 | processors/mel-spectrogram.ts (NeMo) |
| GigaamPreprocessorV2/V3 | processors/gigaam-mel.ts |
| T-one | no frontend; raw → model |
| Kaldi encoder/decoder/joiner | model-owned Zipformer + stateless transducer implementation |
| GigaAM encoder/decoder/joint | model-owned conformer-family + CTC or RNNT implementation |
| T-one model | model-owned chunk encoder implementation |

---

## Cross-Reference

- **NeMo:** [NEMO_ASR_MODEL_ARCHITECTURE.md](../NEMO_ASR_MODEL_ARCHITECTURE.md)
- **MedASR:** [GOOGLE-MEDASR_MODEL_ARCHITECTURE.md](../GOOGLE-MEDASR_MODEL_ARCHITECTURE.md)
- **Whisper:** [WHISPER_ASR_MODEL_ARCHITECTURE.md](../WHISPER_ASR_MODEL_ARCHITECTURE.md)
- **Master index:** [ASR_MODEL_FAMILY_INDEX.md](./ASR_MODEL_FAMILY_INDEX.md)
