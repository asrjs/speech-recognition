# NeMo ASR: Complete Component Breakdown for Inference Library Design

Reference: `N:\github\ysdede\NeMo` source code analysis.

---

## The Full Inference Pipeline (6 layers)

Every NeMo ASR model follows this pipeline at inference:

```
Audio Waveform
  │
  ▼
┌─────────────────────────────────┐
│ 1. PREPROCESSOR (Frontend)      │  AudioToMelSpectrogramPreprocessor
│    raw audio → mel spectrogram  │  (or AudioToMFCCPreprocessor)
└──────────────┬──────────────────┘
               │ [B, n_mels, T_frames]
               ▼
┌─────────────────────────────────┐
│ 2. SPEC AUGMENT (train only)    │  SpectrogramAugmentation
│    freq/time masking            │  (skip at inference)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 3. ENCODER (Acoustic Model)     │  ConformerEncoder / ConvASREncoder / ...
│    mel → encoded representation │  includes subsampling (striding/vgg/dw)
└──────────────┬──────────────────┘
               │ [B, T_enc, D_enc]
               ▼
┌─────────────────────────────────┐
│ 4. DECODER HEAD (Topology)      │  depends on model type:
│    encoded → logits/tokens      │    CTC: ConvASRDecoder (linear)
│                                 │    RNNT: RNNTDecoder + RNNTJoint
│                                 │    TDT:  RNNTDecoder + RNNTJoint(+durations)
│                                 │    AED:  TransformerDecoder + LM head
└──────────────┬──────────────────┘
               │ logits / token IDs
               ▼
┌─────────────────────────────────┐
│ 5. DECODING STRATEGY            │  CTCDecoding / RNNTDecoding / MultiTaskDecoding
│    logits → token ID sequence   │  (greedy, beam, TSD, ALSD, etc.)
│    (search algorithm)           │
└──────────────┬──────────────────┘
               │ [token IDs]
               ▼
┌─────────────────────────────────┐
│ 6. TOKENIZER                    │  SentencePieceTokenizer / AutoTokenizer /
│    token IDs → text string      │  AggregateTokenizer / CanaryTokenizer
└─────────────────────────────────┘
               │
               ▼
           Text Output
```

---

## Layer 1: Preprocessor / Frontend (Feature Extraction)

**Location:** `nemo/collections/asr/modules/audio_preprocessing.py`  
**Internal:** delegates to `nemo/collections/asr/parts/preprocessing/features.py` → `FilterbankFeatures`

| Class | What it does | Output |
|-------|-------------|--------|
| `AudioToMelSpectrogramPreprocessor` | STFT → mel filterbank → log → normalize | `[B, n_mels, T]` MelSpectrogram |
| `AudioToMFCCPreprocessor` | STFT → mel → DCT → MFCC | `[B, n_mfcc, T]` MFCC |

**Key params** (from config `preprocessor`):
- `sample_rate`: 16000 (universal for NeMo ASR)
- `window_size`: 0.025s (25ms)
- `window_stride`: 0.01s (10ms) — this is the frame rate
- `features`: 80 (number of mel bins — standard for Conformer/FastConformer)
- `n_fft`: 512
- `normalize`: `"per_feature"` (per-channel mean/std normalization)
- `dither`: 1e-5 (small noise added for numerical stability)
- `pad_to`: 0 (no padding)

**For your library:** This is the same across virtually all NeMo ASR models (Parakeet, Canary, Conformer, FastConformer, etc.). One `MelSpectrogramFrontend` class handles them all. The params may differ slightly but the math is identical.

---

## Layer 2: Spec Augment (training only — skip at inference)

**Location:** `nemo/collections/asr/modules/audio_preprocessing.py` → `SpectrogramAugmentation`

Applies SpecAugment (freq/time masks) and optionally SpecCutout. **Not used at inference.** You can ignore this for your library.

---

## Layer 3: Encoder (Acoustic Model)

**Location:** `nemo/collections/asr/modules/`

The encoder is the heavy neural network body. It includes an internal **subsampling front-end** that reduces the time dimension before the main transformer/conv blocks.

### Encoder classes

| Class | File | Description | Subsampling |
|-------|------|-------------|-------------|
| `ConformerEncoder` | `conformer_encoder.py` | Conformer & **FastConformer** | striding, vggnet, dw_striding, stacking |
| `ConvASREncoder` | `conv_asr.py` | Jasper, QuartzNet, CitriNet (conv blocks) | built into JasperBlock strides |
| `SqueezeformerEncoder` | `squeezeformer_encoder.py` | Squeezeformer | similar to Conformer |
| `RNNEncoder` | `rnn_encoder.py` | LSTM/GRU encoder | — |
| `ConvFeatureEncoder` | `wav2vec_modules.py` | Wav2Vec2 conv feature extractor | — |

### Conformer vs FastConformer

**They are the same class** (`ConformerEncoder`). The difference is in config:

| Param | Conformer | FastConformer |
|-------|-----------|---------------|
| `subsampling` | `striding` (factor 4) | `dw_striding` (factor 8) |
| `conv_kernel_size` | 31 | 9 |
| `subsampling_conv_channels` | -1 (= d_model) | 256 |

FastConformer achieves ~2x speedup via 8x subsampling + smaller kernels.

### Subsampling (inside encoder)

**Location:** `nemo/collections/asr/parts/submodules/subsampling.py`

| Class | Method | Factor |
|-------|--------|--------|
| `ConvSubsampling` | `"striding"` | 4 (2 conv layers, stride 2 each) |
| `ConvSubsampling` | `"vggnet"` | 4 |
| `ConvSubsampling` | `"dw_striding"` | 4 or 8 (depthwise separable) |
| `StackingSubsampling` | `"stacking"` | configurable |
| `SubsamplingReductionModule` | pooling / striding | additional mid-encoder reduction |

**Output:** `encoder._feat_out` = `d_model` (e.g. 512 for Large). Shape: `[B, T_enc, D_enc]` where `T_enc = T_frames / subsampling_factor`.

---

## Layer 4: Decoder Head (Topology)

This is what varies most between model families. Three fundamentally different topologies:

### 4A. CTC Decoder (non-autoregressive)

**Class:** `ConvASRDecoder` in `conv_asr.py`  
**What:** Single linear layer: `[B, T_enc, D_enc] → [B, T_enc, V+1]` (V = vocab size, +1 = blank)  
**Used by:** `EncDecCTCModel`, `EncDecCTCModelBPE`, auxiliary CTC head in hybrid models.

Config key: `decoder.feat_in` = encoder output dim, `decoder.num_classes` = vocab size.

### 4B. Transducer Decoder + Joiner (autoregressive)

Two sub-components that work together:

**Prediction Network (decoder):**

| Class | File | Description |
|-------|------|-------------|
| `RNNTDecoder` | `rnnt.py` | LSTM-based stateful prediction net (standard RNNT) |
| `StatelessTransducerDecoder` | `rnnt.py` | Embedding-based stateless (for TDT) |

**Abstract base:** `AbstractRNNTDecoder` in `rnnt_abstract.py`. Key interface: `predict(y, state, add_sos, batch_size) → (output, lengths, new_state)`.

Config key: `decoder.prednet.pred_hidden`, `decoder.prednet.pred_rnn_layers`, `decoder.vocab_size`.

**Joint Network (joiner):**

| Class | File | Description |
|-------|------|-------------|
| `RNNTJoint` | `rnnt.py` | Standard: `project_encoder(enc) + project_prednet(dec) → joint → logits [B,T,U,V+1]` |
| `SampledRNNTJoint` | `rnnt.py` | Same but with sampled softmax for large vocab |
| `HATJoint` | `hybrid_autoregressive_transducer.py` | HAT variant |

**Abstract base:** `AbstractRNNTJoint` in `rnnt_abstract.py`. Key interface: `joint(f, g)` calls `project_encoder(f)`, `project_prednet(g)`, then `joint_after_projection(f', g')`.

Config keys: `joint.jointnet.encoder_hidden` = enc_hidden, `joint.jointnet.pred_hidden` = pred_hidden, `joint.jointnet.joint_hidden`, `joint.num_classes` = vocab size.

**TDT (Token-and-Duration Transducer):** Same `RNNTJoint` class with `joint.num_extra_outputs` = number of duration bins (e.g. 5 for durations [0,1,2,3,4]). The joint output is `[B, T, U, V + 1 + num_durations]`.

### 4C. AED (Attention Encoder-Decoder) — Canary

**Used by:** `EncDecMultiTaskModel` (Canary, etc.)

Components:
- Optional `encoder_decoder_proj` (linear, if encoder dim ≠ decoder dim)
- Optional `transf_encoder` (additional transformer encoder layers)
- `transf_decoder` (transformer decoder with cross-attention to encoder)
- `log_softmax` / `head` (TokenClassifier — linear to vocab)

Config keys: `model_defaults.asr_enc_hidden`, `model_defaults.lm_enc_hidden`, `model_defaults.lm_dec_hidden`, `transf_decoder`, `head`.

---

## Layer 5: Decoding Strategy (Search Algorithm)

**Location:** `nemo/collections/asr/parts/submodules/`

This layer takes raw model outputs and runs a search algorithm to produce token sequences. **NOT a neural network** — it's pure algorithm, but model-type-specific.

| Topology | Decoding Class | File | Strategies |
|----------|---------------|------|------------|
| CTC | `CTCDecoding` / `CTCBPEDecoding` | `ctc_decoding.py` | greedy, beam (KenLM) |
| RNNT | `RNNTDecoding` / `RNNTBPEDecoding` | `rnnt_decoding.py` | greedy, greedy_batch, beam, TSD, ALSD, MAES |
| TDT | same `RNNTDecoding` with `model_type="tdt"` | `rnnt_decoding.py` | greedy (recommended), greedy_batch, beam, MAES |
| AED | `MultiTaskDecoding` | `multitask_decoding.py` | greedy, beam |

**Model type enum** (for transducer):
```
TransducerModelType: RNNT | TDT | MULTI_BLANK
```

**For TDT:** The decoding uses the duration predictions to skip frames. `greedy` (non-batch) is strongly recommended; `greedy_batch` gives poor results with `omega=0`.

**For your library:** Each topology needs its own decoding loop. CTC decoding is trivial (argmax + collapse). RNNT/TDT needs an autoregressive loop with the prediction net + joint. AED needs standard transformer autoregressive generation.

---

## Layer 6: Tokenizer

**Location:** `nemo/collections/common/tokenizers/`  
**Wiring:** via `ASRBPEMixin._setup_tokenizer()` in `nemo/collections/asr/parts/mixins/mixins.py`

| Class | File | Type | Used by |
|-------|------|------|---------|
| `SentencePieceTokenizer` | `sentencepiece_tokenizer.py` | BPE (SentencePiece) | Most models (Parakeet, Conformer-BPE, etc.) |
| `AutoTokenizer` | `huggingface/auto_tokenizer.py` | WPE (WordPiece via HF) | Some models |
| `AggregateTokenizer` | `aggregate_tokenizer.py` | Multi-lingual (multiple monolingual tokenizers with offset IDs) | Multilingual models |
| `CanaryTokenizer` | `canary_tokenizer.py` | Extends AggregateTokenizer with special prompt tokens | Canary |
| `CharTokenizer` | `char_tokenizer.py` | Character-level | Char models (non-BPE) |
| `ByteLevelTokenizer` | `bytelevel_tokenizers.py` | Byte-level | — |
| `TiktokenTokenizer` | `tiktoken_tokenizer.py` | Tiktoken (OpenAI-style) | — |

**Abstract base:** `TokenizerSpec` — defines the interface: `text_to_tokens`, `tokens_to_text`, `text_to_ids`, `ids_to_text`, `tokens_to_ids`, `ids_to_tokens`.

**Config:** `tokenizer.type` = `"bpe"` (SentencePiece) or `"wpe"` (WordPiece) or `"agg"` (aggregate). `tokenizer.dir` points to the directory with `tokenizer.model` / `vocab.txt`.

**For char-based models** (non-BPE): vocabulary is in `cfg.labels` as a list of characters; no tokenizer object is used — just index lookup.

---

## Complete Model Families (with all 6 layers)

### Parakeet-TDT (e.g. nvidia/parakeet-tdt-0.6b)

| Layer | Component | Config `_target_` |
|-------|-----------|-------------------|
| Frontend | `AudioToMelSpectrogramPreprocessor` | `nemo...AudioToMelSpectrogramPreprocessor` |
| Encoder | `ConformerEncoder` (FastConformer variant: dw_striding×8, kernel=9) | `nemo...ConformerEncoder` |
| Decoder | `RNNTDecoder` (LSTM pred net) | `nemo...RNNTDecoder` |
| Joiner | `RNNTJoint` with `num_extra_outputs=5` (TDT durations) | `nemo...RNNTJoint` |
| Decoding | `RNNTBPEDecoding` with `model_type="tdt"`, `strategy="greedy"` | — |
| Tokenizer | `SentencePieceTokenizer` (BPE) | — |

### Parakeet-CTC (e.g. nvidia/parakeet-ctc-1.1b)

| Layer | Component |
|-------|-----------|
| Frontend | `AudioToMelSpectrogramPreprocessor` (same) |
| Encoder | `ConformerEncoder` (FastConformer variant, same) |
| Decoder | `ConvASRDecoder` (single linear) |
| Joiner | — (none) |
| Decoding | `CTCBPEDecoding`, `strategy="greedy"` |
| Tokenizer | `SentencePieceTokenizer` (BPE) |

### Parakeet-RNNT

| Layer | Component |
|-------|-----------|
| Frontend | `AudioToMelSpectrogramPreprocessor` |
| Encoder | `ConformerEncoder` (FastConformer) |
| Decoder | `RNNTDecoder` (LSTM pred net) |
| Joiner | `RNNTJoint` (`num_extra_outputs=0`, standard RNNT) |
| Decoding | `RNNTBPEDecoding`, `strategy="greedy_batch"` |
| Tokenizer | `SentencePieceTokenizer` (BPE) |

### Canary (multilingual AED)

| Layer | Component |
|-------|-----------|
| Frontend | `AudioToMelSpectrogramPreprocessor` |
| Encoder | `ConformerEncoder` (FastConformer) |
| Proj | `nn.Linear` (if enc_hidden ≠ dec_hidden) |
| Transf Encoder | `TransformerEncoder` (optional extra layers) |
| Transf Decoder | `TransformerDecoderNM` (cross-attention decoder) |
| Head | `TokenClassifier` (linear to vocab, weight-tied with decoder embedding) |
| Decoding | `MultiTaskDecoding`, `strategy="greedy"` |
| Tokenizer | `CanaryTokenizer` (aggregate, with special prompt tokens) |
| Prompting | `PromptFormatter` (language/task/pnc control tokens) |

### Hybrid RNNT+CTC

| Layer | Component |
|-------|-----------|
| Frontend | `AudioToMelSpectrogramPreprocessor` |
| Encoder | `ConformerEncoder` (shared) |
| RNNT Decoder | `RNNTDecoder` |
| RNNT Joiner | `RNNTJoint` |
| CTC Decoder | `ConvASRDecoder` (auxiliary, from `aux_ctc.decoder`) |
| Decoding | `RNNTBPEDecoding` or `CTCBPEDecoding` (switchable via `cur_decoder`) |
| Tokenizer | `SentencePieceTokenizer` |

### Conformer-TDT-Stateless

| Layer | Component |
|-------|-----------|
| Frontend | same |
| Encoder | `ConformerEncoder` (standard: striding×4, kernel=31) |
| Decoder | `StatelessTransducerDecoder` (embedding-based, no LSTM) |
| Joiner | `RNNTJoint` with `num_extra_outputs=5` (TDT durations) |
| Decoding | `RNNTDecoding` with `model_type="tdt"` |
| Tokenizer | `SentencePieceTokenizer` |

---

## Recommended Reusable Module Map for Your Library

```
your-asr-lib/
├── processors/
│   ├── mel-spectrogram.ts       # AudioToMelSpectrogramPreprocessor (NeMo)
│   │                              STFT → mel → log → normalize. Covers all NeMo ASR.
│   │                              Params: sample_rate, window_size/stride, n_fft, features
│   ├── whisper-mel.ts           # Whisper: log10, clamp(max-8), (x+4)/4 (different from NeMo)
│   └── wav2vec2-feature-extractor.ts  # 7 conv layers on raw waveform (Google MedASR)
│
├── inference/
│   ├── graph.ts                 # Shared encoder / decoder-head / decoding descriptors
│   ├── backends/                # Backend capability probes and execution helpers
│   └── streaming/               # Overlap merge, partial/final state, long-audio orchestration
│
├── tokenizers/
│   ├── sentencepiece.ts         # SentencePieceTokenizer (BPE) — most common
│   ├── wordpiece.ts             # AutoTokenizer/WordPiece (HF-based)
│   ├── aggregate.ts             # AggregateTokenizer (multi-lang with ID offsets)
│   ├── canary.ts                # CanaryTokenizer (special prompt tokens)
│   ├── char.ts                  # CharTokenizer (character vocab)
│   └── whisper-tiktoken.ts     # Whisper BPE (tiktoken format)
│
└── models/                      # Concrete implementation families own real math and wiring
    ├── nemo-tdt/                # FastConformer + transducer/TDT implementation
    ├── hf-ctc-common/           # Wav2Vec2-Conformer + CTC implementation
    └── whisper-seq2seq/         # Whisper encoder-decoder implementation
```

---

## Key Dimensions / Wiring Points

When a new model is added, these are the numbers that wire everything together:

| Config Path | What It Wires | Example (Large) |
|-------------|---------------|-----------------|
| `preprocessor.features` | mel bins → encoder `feat_in` | 80 |
| `encoder.d_model` | encoder hidden dim = `model_defaults.enc_hidden` | 512 |
| `encoder.subsampling_factor` | time compression ratio | 4 or 8 |
| `encoder._feat_out` | encoder output dim (usually = d_model) | 512 |
| `decoder.prednet.pred_hidden` | prediction net hidden = `model_defaults.pred_hidden` | 640 |
| `joint.jointnet.encoder_hidden` | must = `enc_hidden` | 512 |
| `joint.jointnet.pred_hidden` | must = `pred_hidden` | 640 |
| `joint.jointnet.joint_hidden` | joint bottleneck dim | 640 |
| `joint.num_classes` | vocab size | tokenizer.vocab_size |
| `joint.num_extra_outputs` | TDT duration bins (0 for RNNT) | 5 for TDT |
| `decoder.feat_in` (CTC) | must = encoder._feat_out | 512 |
| `decoder.num_classes` (CTC) | vocab size | tokenizer.vocab_size |

---

## Other Vendors

**Google MedASR** (`google/medasr`) uses **Wav2Vec2-Conformer** + CTC. Different frontend (7 conv on raw waveform) and encoder. CTC decoder/decoding shared. See `GOOGLE-MEDASR_MODEL_ARCHITECTURE.md`.

**OpenAI Whisper** uses **encoder-decoder transformer** + autoregressive generation (AED). Different frontend (mel with log10/clamp/scale), plain transformer encoder (2× conv subsampling), transformer decoder with cross-attention. Nothing shared at layer level. See `WHISPER_ASR_MODEL_ARCHITECTURE.md`.

---

## Summary: What the AI response got right (and what to add)

The AI's proposed structure is largely correct. Here's what to make sure you don't miss:

1. **Frontend = `AudioToMelSpectrogramPreprocessor`** — one class, parameterized. Reusable across ALL models.
2. **Subsampling is INSIDE the encoder** (not a separate layer in NeMo), but for ONNX/WebGPU you may want to split it out.
3. **Encoder:** Conformer and FastConformer are the **same class** (`ConformerEncoder`), just different config. One implementation covers both.
4. **Decoder heads are topology-specific:**
   - CTC: trivial linear (`ConvASRDecoder`)
   - Transducer: prediction net (`RNNTDecoder` or `StatelessTransducerDecoder`) + joiner (`RNNTJoint`)
   - AED: transformer decoder + LM head
5. **Decoding strategies are separate from decoder heads** — they are search algorithms, not neural networks. In `asr.js`, the shared descriptor lives in `src/inference/graph.ts` and the concrete loop stays model-owned until multiple families genuinely share it.
6. **Tokenizer is a separate layer** — `SentencePieceTokenizer` for most models, `CanaryTokenizer` for Canary/multilingual. This is where token IDs become text.
7. **Char vs BPE:** Non-BPE models (e.g. `EncDecRNNTModel` vs `EncDecRNNTBPEModel`) use `cfg.labels` (character list) instead of a tokenizer. The mixin `ASRBPEMixin` adds tokenizer support.
