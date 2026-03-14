# Hugging Face Transformers: ASR & Speech LLM Model Families

Reference: `N:\github\ysdede\transformers` — Hugging Face Transformers ASR and speech-to-text models.

This doc maps **transformers** model families to the shared layer taxonomy for designing `asr.js`.

---

## Model Family Overview

| Family | Topology | Frontend | Encoder | Decoder | Notes |
|--------|----------|----------|---------|---------|-------|
| **Wav2Vec2 / Conformer** | CTC | Raw (7-layer conv) | Transformer / Conformer | CTC head | No external mel |
| **Parakeet CTC** | CTC | NeMo-style mel | Conformer (rel pos) | CTC head | NVIDIA Parakeet |
| **Speech2Text** | AED | External mel (80) | Conv subsampler + Trans | Trans decoder | Fairseq S2T |
| **Whisper** | AED | Whisper mel | 2 conv + Trans | Trans decoder | See WHISPER doc |
| **Qwen2-Audio** | AED/Causal LM | Conv + mel in model | Trans encoder | Trans decoder | Alibaba |
| **Granite Speech** | AED | Projector (queries) | Trans encoder | LLM decoder | IBM |
| **Moonshine** | AED | Conv subsampler | Conformer (RoPE) | Trans decoder | Streaming |
| **Voxtral** | AED | — | — | LLM decoder | Mistral + audio |
| **GLMASR** | AED | — | — | GLM decoder | Zhipu AI |
| **VibeVoice ASR** | AED | — | — | Decoder | Conditional gen |
| **Kyutai Speech-to-Text** | AED | — | — | Decoder | Kyutai |
| **LASR CTC** | CTC | — | — | CTC head | — |
| **SeamlessM4T** | AED/CTC | — | — | — | Multimodal |

---

## 1. Wav2Vec2 / Wav2Vec2-Conformer (CTC)

**Source:** `wav2vec2`, `wav2vec2_conformer`, `wav2vec2_bert`, `hubert`, `unispeech`, `data2vec-audio`, `wavlm`, `sew`, `sew-d`

**Topology:** Encoder-only + CTC head. **No mel frontend** — model has 7-layer 1D conv on raw waveform.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ Wav2Vec2FeatureEncoder (inside model)
  │ 7 conv layers: (512,10,5), (512,3,2), (512,3,2), (512,3,2), (512,2,2), (512,2,2), (512,2,2)
  │ → ~320× temporal subsampling
  ▼
  [B, T/320, 512]
  │
  ▼ Transformer encoder (12 layers) or Conformer encoder
  ▼
  CTC head (linear → vocab)
```

**Subsampling:** `inputs_to_logits_ratio` (often 320).

---

## 2. Parakeet CTC (NVIDIA)

**Source:** `parakeet_ctc` — Hugging Face Parakeet (Conformer from NeMo lineage).

**Topology:** CTC. Frontend is **external** (NeMo-style mel); encoder is Conformer with relative positional encoding.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ External preprocessor (NeMo mel, 80 bins)
  │ STFT, mel, log, per-feature norm
  ▼
  [B, 80, T]
  │
  ▼ Parakeet encoder (Conformer blocks)
  │ RelPositionalEncoding, ConvolutionModule (from FastSpeech2)
  ▼
  CTC head
```

**Relation:** Same encoder style as NeMo FastConformer / Parakeet; CTC decoder.

---

## 3. Speech2Text (Fairseq S2T)

**Source:** `speech_to_text` — Facebook Fairseq S2T.

**Topology:** AED (encoder–decoder). Frontend is **external** — 80-dim log-mel filterbank. Encoder has Conv1d subsampler + Transformer.

### Pipeline

```
Raw waveform → External feature extractor
  │ 80-dim log-mel (input_feat_per_channel=80)
  ▼
  [B, T, 80]  (transposed to B x (C*D) x T internally)
  │
  ▼ Conv1dSubsampler
  │ num_conv_layers=2, kernel_sizes=(5,5), stride=2 each
  │ GLU activation
  ▼
  ▼ Transformer encoder (12 layers)
  ▼
  ▼ Transformer decoder (6 layers) with cross-attention
  ▼
  LM head
```

---

## 4. Whisper

See `WHISPER_ASR_MODEL_ARCHITECTURE.md`. Frontend: Whisper mel (log10, clamp, scale). Encoder: 2 conv + Transformer. Decoder: causal Transformer with cross-attention.

---

## 5. Qwen2-Audio

**Source:** `qwen2_audio` — Alibaba Qwen2-Audio (speech + text multimodal).

**Topology:** AED / conditional generation. Audio goes through **conv feature extractor** and then encoder; decoder is a language model.

### Pipeline

```
Raw waveform
  │
  ▼ Conv feature extractor (in model)
  │ Mel or raw convs similar to Wav2Vec2
  ▼
  ▼ Qwen2AudioEncoder (Transformer)
  ▼
  ▼ Qwen2AudioForConditionalGeneration
  │ Cross-attention from LM decoder to encoder
  ▼
  Causal LM decoder (next-token prediction)
```

---

## 6. Granite Speech (IBM)

**Source:** `granite_speech` — IBM Granite Speech.

**Topology:** AED. **Projector** maps encoder frames to fixed **queries** (window-based). Decoder is an LLM (e.g. Llama-style).

### Pipeline

```
Raw waveform
  │
  ▼ Encoder (e.g. Conv + Transformer)
  ▼
  ▼ GraniteSpeechEncoderProjector
  │ window_size, downsample_rate, num_queries = window_size // downsample_rate
  │ learnable query [1, num_queries, hidden]
  ▼
  ▼ Cross-attention: decoder attends to projected encoder output
  ▼
  LLM decoder (causal LM)
```

---

## 7. Moonshine

**Source:** `moonshine`, `moonshine_streaming` — Streaming ASR.

**Topology:** AED. Conformer encoder with **RoPE**; conv subsampler; Transformer decoder.

### Pipeline

```
Raw waveform
  │
  ▼ Conv subsampler (similar to Speech2Text)
  ▼
  ▼ Conformer encoder (MoonshineEncoder)
  │ MoonshineRotaryEmbedding, Conformer blocks
  ▼
  ▼ Transformer decoder with cross-attention
  ▼
  LM head
```

---

## 8. Voxtral (Mistral + Audio)

**Source:** `voxtral`, `voxtral_realtime` — Mistral with speech input.

**Topology:** Speech LLM. Audio encoder projects to token space; LLM decoder (Mistral-style) does autoregressive generation.

---

## 9. GLMASR (Zhipu AI)

**Source:** `glmasr` — GLM for ASR.

**Topology:** Encoder–decoder with GLM decoder (GLM-4 architecture).

---

## 10. VibeVoice ASR

**Source:** `vibevoice_asr` — Conditional generation for ASR.

---

## 11. Kyutai Speech-to-Text

**Source:** `kyutai_speech_to_text` — Kyutai ASR model.

---

## 12. LASR CTC

**Source:** `lasr_ctc` — LASR model with CTC head.

---

## 13. SeamlessM4T

**Source:** `seamless_m4t` — Multimodal (text, speech). Can do ASR via encoder–decoder or CTC depending on task.

---

## Frontend Summary (Transformers)

| Model | Frontend | Input |
|-------|----------|-------|
| Wav2Vec2 | 7-layer conv (in model) | Raw waveform |
| Wav2Vec2-Conformer | 7-layer conv (in model) | Raw waveform |
| Parakeet CTC | External mel (80) | Mel spectrogram |
| Speech2Text | External mel (80) | Log-mel filterbank |
| Whisper | Whisper mel (80) | WhisperFeatureExtractor |
| Qwen2-Audio | Conv (in model) | Raw waveform |
| Granite Speech | Projector + encoder | Raw → encoder → queries |
| Moonshine | Conv subsampler | Raw waveform |

---

## Decoder Topology Summary

| Topology | Models |
|----------|--------|
| **CTC** | Wav2Vec2, Wav2Vec2-Conformer, Parakeet CTC, LASR CTC |
| **AED (encoder–decoder)** | Speech2Text, Whisper, Qwen2-Audio, Granite Speech, Moonshine |
| **Speech LLM** | Voxtral, GLMASR, VibeVoice ASR, Kyutai S2T |

---

## Reusable Module Mapping for asr.js

| Transformers Model | asr.js module |
|-------------------|---------------|
| Wav2Vec2FeatureEncoder | processors/wav2vec2-conv.ts |
| WhisperFeatureExtractor | processors/whisper-mel.ts |
| Parakeet / NeMo mel | processors/mel-spectrogram.ts |
| Speech2Text Conv1dSubsampler | processors/conv-subsampler.ts |
| Conformer encoder | `src/inference/graph.ts` descriptor + model-owned implementation |
| Transformer encoder | `src/inference/graph.ts` descriptor + model-owned implementation |
| CTC head | `src/inference/graph.ts` descriptor + model-owned implementation |
| Transducer (RNNT) | `src/inference/graph.ts` descriptor + model-owned implementation |
| AED decoder | `src/inference/graph.ts` descriptor + model-owned implementation |

---

## Cross-Reference

- **NeMo:** [NEMO_ASR_MODEL_ARCHITECTURE.md](../NEMO_ASR_MODEL_ARCHITECTURE.md)
- **MedASR:** [GOOGLE-MEDASR_MODEL_ARCHITECTURE.md](../GOOGLE-MEDASR_MODEL_ARCHITECTURE.md)
- **Whisper:** [WHISPER_ASR_MODEL_ARCHITECTURE.md](../WHISPER_ASR_MODEL_ARCHITECTURE.md)
- **ONNX-ASR:** [ONNX_ASR_MODEL_ARCHITECTURE.md](./ONNX_ASR_MODEL_ARCHITECTURE.md)
- **Master index:** [ASR_MODEL_FAMILY_INDEX.md](./ASR_MODEL_FAMILY_INDEX.md)
