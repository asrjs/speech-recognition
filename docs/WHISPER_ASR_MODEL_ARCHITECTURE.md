# Whisper ASR Model Architecture

Reference: Hugging Face Whisper implementation notes and upstream model code.

Whisper is an **encoder-decoder (sequence-to-sequence)** transformer for speech recognition. Unlike NeMo (CTC/RNNT/TDT) or MedASR (CTC), Whisper uses **autoregressive generation** with a decoder that cross-attends to the encoder output.

---

## The Full Inference Pipeline (6 layers)

```
Raw Waveform [16kHz mono]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. FRONTEND (Feature Extraction)                                     │
│    WhisperFeatureExtractor: STFT → mel filterbank → log10          │
│    Different from NeMo: log10 (not ln), clamp(max-8), (x+4)/4       │
│    hop_length=160 (10ms), n_fft=400, 80 mel bins                    │
└─────────────────────────────────────────────────────────────────────┘
               │ [B, 80, T_frames]
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. ENCODER SUBSAMPLING (inside encoder)                             │
│    conv1 (k=3, p=1) → conv2 (k=3, s=2, p=1) → 2× time reduction    │
│    GELU activation on both                                           │
└─────────────────────────────────────────────────────────────────────┘
               │ [B, d_model, T_enc]  (T_enc = T_frames / 2)
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. ENCODER (Acoustic Model)                                          │
│    WhisperEncoder: N × WhisperEncoderLayer (plain transformer)       │
│    Pre-norm: LayerNorm → Self-Attention → residual → LayerNorm → FF  │
│    Sinusoidal positional embeddings (fixed, not learned)            │
│    NO convolution module (unlike Conformer)                          │
└─────────────────────────────────────────────────────────────────────┘
               │ [B, T_enc, d_model]
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. DECODER (Topology) — Transformer decoder + LM head               │
│    WhisperDecoder: N × WhisperDecoderLayer                           │
│    Each layer: self-attn (causal) → cross-attn (to encoder) → FF     │
│    LM head: Linear(d_model, vocab_size), weight-tied with embed     │
│    Autoregressive: predicts next token given encoder + previous      │
└─────────────────────────────────────────────────────────────────────┘
               │ logits [B, seq_len, vocab_size]
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. DECODING — Autoregressive generation                             │
│    WhisperGenerationMixin / model.generate()                         │
│    Suppresses non-speech tokens, optional timestamp tokens           │
│    begin_suppress_tokens, suppress_tokens (NON_SPEECH_TOKENS)        │
│    decoder_start_token_id = 50257 (or 50256 for en-only)             │
└─────────────────────────────────────────────────────────────────────┘
               │ [token IDs]
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. TOKENIZER — WhisperTokenizer / WhisperTokenizerFast              │
│    BPE via tiktoken (multilingual.tiktoken or gpt2.tiktoken)         │
│    batch_decode(ids, skip_special_tokens=True) → text                │
│    Language/task tokens for prompting (e.g. <|en|>, <|transcribe|>)   │
└─────────────────────────────────────────────────────────────────────┘
               │
               ▼
           Text Output
```

---

## Layer 1: Frontend (WhisperFeatureExtractor)

**Reference module:** `models/whisper/feature_extraction_whisper.py`

| Param           | Value | Notes                            |
| --------------- | ----- | -------------------------------- |
| `feature_size`  | 80    | Mel bins (same as NeMo)          |
| `sampling_rate` | 16000 | 16 kHz                           |
| `hop_length`    | 160   | 10 ms per frame (160/16000)      |
| `n_fft`         | 400   | STFT window size                 |
| `chunk_length`  | 30    | Max seconds (padding/truncation) |
| `dither`        | 0.0   | Optional Gaussian noise          |

**Mel pipeline (differs from NeMo):**

1. STFT → power spectrogram (power=2.0)
2. Mel filterbank (Slaney norm, 0–8 kHz)
3. `log10` (not natural log)
4. Clamp: `max(log_spec, log_spec.max() - 8.0)`
5. Scale: `(log_spec + 4.0) / 4.0`

**Optional:** `do_normalize=True` → zero-mean unit-variance per sample.

**Output shape:** `[B, 80, T]` where `T = n_samples // hop_length` (minus last frame in impl).

**For your library:** Whisper mel frontend is **not** interchangeable with NeMo. Use a dedicated `WhisperMelFrontend` with the clamp + scale steps.

---

## Layer 2 + 3: Encoder (WhisperEncoder)

**Reference module:** `models/whisper/modeling_whisper.py`

### Subsampling (first part of encoder)

```python
conv1 = Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)   # no stride
conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
# inputs_embeds = gelu(conv2(gelu(conv1(input_features))))
```

- Total subsampling: **2×** (only conv2 has stride 2)
- NeMo Conformer: 4× or 8× — Whisper encoder is lighter on subsampling

### Encoder layers

- **Architecture:** Standard transformer (pre-norm): LayerNorm → Self-Attention → residual → LayerNorm → FF → residual
- **No convolution block** (unlike Conformer)
- **Position embeddings:** Sinusoidal, fixed (not learned)
- **Activation:** GELU

### Model sizes (siblings)

| Size     | Encoder | Decoder | d_model | encoder_ffn_dim | ~Params |
| -------- | ------- | ------- | ------- | --------------- | ------- |
| tiny     | 4       | 4       | 384     | 1536            | ~39M    |
| base     | 6       | 6       | 512     | 2048            | ~74M    |
| small    | 12      | 12      | 768     | 3072            | ~244M   |
| medium   | 24      | 24      | 1024    | 4096            | ~769M   |
| large    | 32      | 32      | 1280    | 5120            | ~1550M  |
| large-v2 | 32      | 32      | 1280    | 5120            | ~1550M  |
| large-v3 | 32      | 32      | 1280    | 5120            | ~1550M  |

English-only variants: `tiny.en`, `base.en`, `small.en`, `medium.en` — same architecture, smaller vocab (50257 vs 51865).

---

## Layer 4: Decoder (WhisperDecoder) + LM Head

**Reference module:** `models/whisper/modeling_whisper.py`

- **Topology:** Encoder-decoder (AED), not CTC or transducer
- **Decoder layer:** Self-attention (causal) → cross-attention (encoder) → FF
- **LM head:** `proj_out = Linear(d_model, vocab_size)`, weight-tied with `embed_tokens`
- **Decoding:** Autoregressive; each step conditions on encoder output + all previous tokens

**Special tokens:**

- `decoder_start_token_id`: 50257 (multilingual) or 50256 (en-only)
- `pad_token_id` = `eos_token_id` = 50256
- `begin_suppress_tokens`: [220, 50256] — space and EOS suppressed at start
- `suppress_tokens`: NON_SPEECH_TOKENS — long list of tokens to suppress during generation (prompt/language/task tokens, etc.)

---

## Layer 5: Decoding (WhisperGenerationMixin)

**Reference module:** `models/whisper/generation_whisper.py`

Whisper uses custom generation logic:

- **Logits processors:** `SuppressTokensLogitsProcessor`, `SuppressTokensAtBeginLogitsProcessor`, `WhisperNoSpeechDetection`, `WhisperTimeStampLogitsProcessor`
- **No CTC collapse** — pure autoregressive sampling/beam
- **Timestamp decoding:** Optional; uses alignment heads + median filter + DTW

---

## Layer 6: Tokenizer (WhisperTokenizer)

**Reference module:** `models/whisper/tokenization_whisper.py`

- **Type:** BPE via tiktoken (GPT-2 style mergeable ranks)
- **Vocab:** 51865 (multilingual) or 50257 (English-only)
- **Special tokens:** Language (`<|en|>`, `<|zh|>`, …), task (`<|transcribe|>`, `<|translate|>`), timestamps (`<|0.00|>`, …), etc.
- **Decoding:** `batch_decode(ids, skip_special_tokens=True)` plus optional `normalize` (English text normalizer)

---

## What Is Shared vs Not Shared with NeMo/MedASR

| Component            | NeMo                                | MedASR                | Whisper                            |
| -------------------- | ----------------------------------- | --------------------- | ---------------------------------- |
| **Frontend**         | Mel (STFT→mel→log→per_feature norm) | 128-bin kaldi-style mel | Mel (STFT→mel→log10→clamp→scale) |
| **Encoder**          | Conformer (conv block)              | Conformer             | Plain transformer (no conv block)  |
| **Subsampling**      | 4× or 8× inside encoder             | 10 ms frame hop path  | 2× in encoder (2 conv layers)      |
| **Decoder topology** | CTC / RNNT / TDT / AED              | CTC                   | AED (transformer decoder)          |
| **Decoding**         | CTC greedy/beam, RNNT, etc.         | CTC                   | Autoregressive + logits processors |
| **Tokenizer**        | SentencePiece / WPE                 | MedASR tokenizer      | Tiktoken BPE                       |

**Shared with NeMo:** Nothing at the layer level — different frontend, encoder, and topology.
**Shared with MedASR:** Nothing — MedASR is CTC; Whisper is encoder-decoder autoregressive.

---

## Recommended Module Map for @asrjs/speech-recognition

```
src/
├── audio/
│   ├── mel-spectrogram.ts        # NeMo-style (STFT->mel->ln->per_feature)
│   ├── whisper-mel.ts            # Whisper-style (log10, clamp, scale)
│   └── kaldi-mel.ts              # MedASR runtime frontend
├── inference/
│   ├── descriptors.ts            # Shared encoder / decoder-head / decoding descriptors
│   └── streaming/                # Shared overlap and partial/final orchestration
├── tokenizers/
│   ├── sentencepiece.ts          # NeMo
│   └── whisper-tiktoken.ts       # Whisper BPE (tiktoken format)
├── models/
│   ├── nemo-tdt/
│   ├── lasr-ctc/
│   └── whisper-seq2seq/
└── presets/
    ├── parakeet/
    ├── medasr/
    └── whisper/
```

---

## Key Dimensions / Wiring

| Config                    | Whisper                | Example (large)            |
| ------------------------- | ---------------------- | -------------------------- |
| `num_mel_bins`            | mel bins               | 80                         |
| `d_model`                 | encoder/decoder hidden | 1280                       |
| `encoder_layers`          | transformer layers     | 32                         |
| `decoder_layers`          | transformer layers     | 32                         |
| `encoder_attention_heads` | MHA heads              | 20                         |
| `decoder_attention_heads` | MHA heads              | 20                         |
| `encoder_ffn_dim`         | FF hidden              | 5120                       |
| `decoder_ffn_dim`         | FF hidden              | 5120                       |
| `vocab_size`              | tokenizer              | 51865 (multi) / 50257 (en) |
| `max_source_positions`    | max mel frames         | 1500                       |
| `max_target_positions`    | max decoder length     | 448                        |

**Input length:** Whisper expects `input_features` of length `max_source_positions * 2` (due to 2× conv subsampling) = 3000 frames. The feature extractor pads/truncates to `chunk_length * sample_rate` samples, then computes mel frames.

---

## Cross-Reference

- **NeMo:** `NEMO_ASR_MODEL_ARCHITECTURE.md` — CTC/RNNT/TDT, mel frontend, Conformer
- **MedASR:** `GOOGLE-MEDASR_MODEL_ARCHITECTURE.md` — LASR CTC family, current ONNX path uses mel + Conformer + CTC

The NEMO doc's "Other Vendors" section and recommended module map include Whisper. All three docs share a consistent 6-layer pipeline structure for comparison.
