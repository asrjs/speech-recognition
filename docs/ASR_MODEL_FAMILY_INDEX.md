# ASR Model Family Index

Master reference for designing `asr.js` from scratch. Maps all documented model families to **frontend**, **encoder**, **decoder topology**, and **decoding strategy**.

---

## Quick Reference Table

| Family | Source | Frontend | Encoder | Topology | Decoding | Doc |
|--------|--------|----------|---------|----------|----------|-----|
| **NeMo (Parakeet, Canary, FastConformer)** | NeMo | Mel 80/128 | Conformer | CTC, RNNT, TDT, AED | Greedy, beam, TSD | [NEMO](../NEMO_ASR_MODEL_ARCHITECTURE.md) |
| **MedASR** | HF google/medasr | 7-layer conv (raw) | Wav2Vec2-Conformer | CTC | CTC greedy/beam | [MEDASR](../GOOGLE-MEDASR_MODEL_ARCHITECTURE.md) |
| **Whisper** | HF whisper | Whisper mel (log10) | 2 conv + Trans | AED | AR generation | [WHISPER](../WHISPER_ASR_MODEL_ARCHITECTURE.md) |
| **Kaldi / Vosk** | onnx-asr | Kaldi mel (80) | Zipformer | Stateless RNNT | Transducer | [ONNX](./ONNX_ASR_MODEL_ARCHITECTURE.md) |
| **GigaAM** | onnx-asr | GigaAM mel (64) | Conformer | CTC, RNNT | Greedy | [ONNX](./ONNX_ASR_MODEL_ARCHITECTURE.md) |
| **T-one** | onnx-asr | **Identity** (raw) | Chunk encoder | CTC | CTC greedy | [ONNX](./ONNX_ASR_MODEL_ARCHITECTURE.md) |
| **Wav2Vec2** | HF | 7-layer conv (raw) | Transformer | CTC | CTC | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Wav2Vec2-Conformer** | HF | 7-layer conv (raw) | Conformer | CTC | CTC | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Parakeet CTC** | HF parakeet_ctc | NeMo mel (80) | Conformer | CTC | CTC | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Speech2Text** | HF speech_to_text | External mel (80) | Conv + Trans | AED | AR generation | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Qwen2-Audio** | HF qwen2_audio | Conv (in model) | Trans | AED | AR generation | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Granite Speech** | HF granite_speech | Projector | Trans | AED | AR generation | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Moonshine** | HF moonshine | Conv subsampler | Conformer (RoPE) | AED | AR generation | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **Voxtral, GLMASR, etc.** | HF | Varies | Varies | Speech LLM | AR generation | [TRANSFORMERS](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) |
| **FireRedASR2-AED** | FireRedASR2S | Kaldi mel (80) | Conformer | AED | Beam search | [FIRERED](./FIRERED_ASR2S_MODEL_ARCHITECTURE.md) |
| **FireRedASR2-LLM** | FireRedASR2S | Kaldi mel (80) | Conformer + Adapter | Speech LLM | AR generation | [FIRERED](./FIRERED_ASR2S_MODEL_ARCHITECTURE.md) |
| **FireRedVAD** | FireRedASR2S | Kaldi mel (80) | DFSMN | Binary/multi-label | Sigmoid threshold | [FIRERED](./FIRERED_ASR2S_MODEL_ARCHITECTURE.md) |
| **FireRedLID** | FireRedASR2S | Kaldi mel (80) | Conformer | AED | Beam search | [FIRERED](./FIRERED_ASR2S_MODEL_ARCHITECTURE.md) |

---

## Frontend Taxonomy

| Type | Families | Params |
|------|----------|--------|
| **Mel (NeMo)** | NeMo, Parakeet CTC | 80/128 bins, n_fft=512, hop=160, Slaney, per_feature norm |
| **Mel (Kaldi)** | Kaldi/Vosk, FireRedASR2S | 80 bins, n_fft=512, hop=160, Kaldi mel, Hann^0.85 |
| **Mel (GigaAM)** | GigaAM | 64 bins, n_fft 400/320, hop=160 |
| **Mel (Whisper)** | Whisper | 80 bins, log10, clamp, scale |
| **Raw (7-layer conv)** | Wav2Vec2, MedASR, Wav2Vec2-Conformer | No mel; conv feature extractor |
| **Identity (raw chunks)** | T-one | No frontend; raw waveform chunks |
| **Conv subsampler** | Speech2Text, Moonshine | 1D conv on mel or raw |

---

## Encoder Taxonomy

| Type | Families |
|------|----------|
| **Conformer** | NeMo, GigaAM, Parakeet, Wav2Vec2-Conformer, Moonshine, FireRedASR2 |
| **Transformer** | Whisper, Speech2Text, Qwen2-Audio, Granite Speech |
| **Zipformer** | Kaldi/Vosk (Icefall) |
| **Chunk encoder** | T-one (stateful, 2400 samples) |
| **7-layer conv** | Wav2Vec2 (no separate encoder; feature extractor) |

---

## Decoder Topology Taxonomy

| Topology | Families | Notes |
|----------|----------|-------|
| **CTC** | NeMo, MedASR, GigaAM, T-one, Wav2Vec2, Parakeet CTC, LASR | Linear → vocab, blank handling |
| **RNNT** | NeMo, GigaAM | LSTM decoder + joint |
| **Stateless RNNT** | Kaldi/Vosk | Context-based, no LSTM |
| **TDT** | NeMo (Parakeet TDT) | RNNT + duration outputs |
| **AED** | Whisper, Speech2Text, Qwen2-Audio, Granite Speech, Moonshine, FireRedASR2-AED | Encoder–decoder, cross-attention |
| **Speech LLM** | Voxtral, GLMASR, VibeVoice, Kyutai S2T, FireRedASR2-LLM | LLM decoder + audio encoder |
| **DFSMN** | FireRedVAD | FSMN blocks, no attention (streaming) |

---

## asr.js Suggested Module Layout

```
asr.js/
├── processors/
│   ├── mel-spectrogram.ts      # NeMo, Parakeet, GigaAM variants
│   ├── kaldi-mel.ts            # Kaldi/Vosk
│   ├── whisper-mel.ts          # Whisper (log10, clamp, scale)
│   ├── wav2vec2-conv.ts        # 7-layer conv (raw)
│   └── identity.ts             # T-one passthrough
├── inference/
│   ├── graph.ts                # Shared encoder / decoder-head / decoding descriptors
│   ├── backends/               # Capability probes and execution helpers
│   └── streaming/              # Rolling windows, overlap merge, partial/final logic
├── tokenizers/
│   └── ...                     # SentencePiece, BPE, tiktoken
└── models/
    └── ...                     # Concrete implementation families own real encoder/decoder math
```

---

## Cross-References

| Document | Content |
|----------|---------|
| [NEMO_ASR_MODEL_ARCHITECTURE.md](../NEMO_ASR_MODEL_ARCHITECTURE.md) | NeMo pipelines, Conformer, CTC/RNNT/TDT/AED |
| [GOOGLE-MEDASR_MODEL_ARCHITECTURE.md](../GOOGLE-MEDASR_MODEL_ARCHITECTURE.md) | MedASR (Wav2Vec2-Conformer + CTC) |
| [WHISPER_ASR_MODEL_ARCHITECTURE.md](../WHISPER_ASR_MODEL_ARCHITECTURE.md) | Whisper encoder–decoder |
| [ONNX_ASR_MODEL_ARCHITECTURE.md](./ONNX_ASR_MODEL_ARCHITECTURE.md) | Kaldi, GigaAM, T-one, NeMo/Whisper ONNX |
| [TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md) | HF transformers ASR & speech LLM families |
| [FIRERED_ASR2S_MODEL_ARCHITECTURE.md](./FIRERED_ASR2S_MODEL_ARCHITECTURE.md) | FireRedASR2S (ASR-AED, ASR-LLM, VAD, LID) |
| [VAD_MODEL_ARCHITECTURE.md](./VAD_MODEL_ARCHITECTURE.md) | TEN VAD, FireRedVAD, Silero-VAD, WebRTC-VAD |
