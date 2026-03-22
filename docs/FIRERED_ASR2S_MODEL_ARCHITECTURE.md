# FireRedASR2S Model Architecture

Reference: `N:\github\ysdede\FireRedASR2S` — industrial-grade all-in-one ASR system with ASR, VAD, LID, and punctuation modules.

**Paper:** [arXiv:2603.10420](https://arxiv.org/pdf/2603.10420)  
**Models:** [HuggingFace](https://huggingface.co/collections/FireRedTeam/fireredasr2s) | [ModelScope](https://www.modelscope.cn/collections/xukaituo/FireRedASR2S)

---

## System Overview

FireRedASR2S comprises four modules, all sharing the same frontend (Kaldi fbank):

| Module              | Task           | Architecture                            | Output                         |
| ------------------- | -------------- | --------------------------------------- | ------------------------------ |
| **FireRedASR2-AED** | Speech-to-text | Conformer encoder + Transformer decoder | Transcription, word timestamps |
| **FireRedASR2-LLM** | Speech-to-text | Conformer encoder + Adapter + LLM       | Transcription                  |
| **FireRedVAD**      | Voice activity | DFSMN                                   | Speech/non-speech segments     |
| **FireRedLID**      | Language ID    | Conformer + Transformer decoder         | Language label                 |
| **FireRedPunc**     | Punctuation    | BERT                                    | Punctuated text                |

---

## Shared Frontend (Layer 1)

**Source:** `fireredasr2s/fireredasr2/data/asr_feat.py`, `fireredasr2s/fireredvad/core/audio_feat.py`, `fireredasr2s/fireredlid/data/feat.py`

All modules use **Kaldi fbank** via `kaldi_native_fbank`:

| Param          | Value           | Notes                                                        |
| -------------- | --------------- | ------------------------------------------------------------ |
| `num_mel_bins` | 80              | Same as NeMo/Kaldi                                           |
| `frame_length` | 25 ms           |                                                              |
| `frame_shift`  | 10 ms           | 100 fps                                                      |
| `dither`       | 0.0 (inference) | 1.0 during training                                          |
| `snip_edges`   | True            | Kaldi default                                                |
| **CMVN**       | Optional        | Per-utterance cepstral mean/variance norm (Kaldi stats file) |

**Layout:** `[B, T, 80]` (time × mel).

---

## 1. FireRedASR2-AED

**Topology:** Attention-based Encoder–Decoder (AED). Same pattern as Whisper, Speech2Text — encoder + decoder with cross-attention.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ ASRFeatExtractor (Kaldi fbank 80 bins)
  │ optional CMVN
  ▼
  [B, T, 80]
  │
  ▼ Conv2dSubsampling (4× subsampling)
  │ 2× Conv2d(1→32, 3×3, stride 2) + ReLU
  │ Linear → [B, T/4, d_model]
  │ context = 7 (left 3, cur, right 3)
  ▼
  ▼ RelPositionalEncoding + ConformerEncoder
  │ N × RelPosEmbConformerBlock:
  │   FFN1 → RelPosMultiHeadAttention → ConformerConvolution → FFN2 → LayerNorm
  │ kernel_size=33
  ▼
  ▼ TransformerDecoder (batch_beam_search)
  │ Embedding + positional encoding
  │ N × DecoderLayer (self-attn + cross-attn + FFN)
  │ LM head (weight-tied with embedding)
  ▼
  CTC head (for timestamps only; forced alignment)
  ▼
  Beam search → token IDs → detokenize
```

### Key Components

- **Conv2dSubsampling:** 4× temporal reduction; outputs `d_model` (e.g. 256).
- **Conformer block:** FFN → MHA (relative pos) → Conv (depthwise) → FFN; Macaron-style.
- **Decoder:** Transformer decoder with cross-attention to encoder; beam search.
- **CTC:** Used only for word-level timestamps via `torchaudio.functional.forced_align` on CTC logits.

### Decoding

- Beam search with `softmax_smoothing`, `length_penalty`, `eos_penalty`.
- Optional external LM (LSTM) with `elm_weight`.
- `decode_max_len=0` → use encoder length as max.

---

## 2. FireRedASR2-LLM

**Topology:** Encoder–Adapter–LLM (Speech LLM). Reuses **FireRedASR2-AED encoder** weights; decoder replaced by a causal LLM.

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ ASRFeatExtractor (Kaldi fbank)
  ▼
  [B, T, 80]
  │
  ▼ FireRedASR2-AED encoder (same ConformerEncoder)
  │ enc_outputs [B, T_enc, d_model]
  ▼
  ▼ Adapter (downsample + linear project)
  │ downsample_rate (e.g. 2): concat adjacent frames → Linear(encoder_dim*ds, llm_dim) → ReLU → Linear
  │ speech_features [B, T_enc/ds, llm_dim]
  ▼
  ▼ Merge with LLM input (special <speech> token)
  │ Each <speech> token replaced by speech_features sequence
  │ inputs_embeds = LLM embed(input_ids) + speech_features at <speech> positions
  ▼
  ▼ LLM.generate() (causal LM)
  │ Qwen-style LLM (AutoModelForCausalLM)
  │ max_new_tokens = speech_len (or decode_max_len)
  ▼
  Detokenize → text
```

### Adapter

```python
# Adapter: encoder_dim → llm_dim with temporal downsample
x = x.view(B, T//ds, encoder_dim * ds)  # concat adjacent frames
x = linear1(x)  # encoder_dim*ds → llm_dim
x = relu(x)
x = linear2(x)  # llm_dim → llm_dim
```

### LLM Input Format

Prompt contains special token `<speech>` (or `default_speech_token_id`). Speech features replace that token with `speech_len - 1` frames; text tokens are placed at remaining positions.

---

## 3. FireRedVAD

**Topology:** DFSMN (Deep Feed-Forward Sequential Memory Network) — no attention, efficient for streaming.

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
  │   FSMN blocks with lookback/lookahead
  │   config: R, M, H, P, N1, S1, N2, S2
  ▼
  ▼ sigmoid(logits) → probs [0,1]
  ▼
  ▼ VAD postprocessor
  │ smooth_window_size, speech_threshold, min_speech_frame, etc.
  ▼
  timestamps (start, end)
```

**Streaming:** `FireRedStreamVad` uses `caches` in DFSMN forward for incremental computation.

**mVAD (multi-label):** Multiple output heads (speech/singing/music) with separate thresholds.

---

## 4. FireRedLID

**Topology:** AED — same Conformer encoder + Transformer decoder as FireRedASR2-AED, but decoder predicts **language ID** (single token or short sequence).

### Pipeline

```
Raw waveform [16 kHz]
  │
  ▼ Kaldi fbank (same as ASR)
  ▼
  ▼ ConformerEncoder
  ▼
  ▼ TransformerDecoder (lid_decoder)
  │ odim = lid_odim (number of languages)
  │ decode_max_len=2 (short output)
  ▼
  Beam search → language ID
```

**Vocab:** 100+ languages, 20+ Chinese dialects/accents.

---

## 5. FireRedPunc

**Topology:** BERT-based sequence labeling. Input: **text** (from ASR output); output: punctuated text.

Not an acoustic model; operates on tokenized text. Uses HuggingFace BERT + token classification head.

---

## Frontend Comparison with Other Families

| Family           | Frontend     | n_mels | frame | hop   | mel_scale     |
| ---------------- | ------------ | ------ | ----- | ----- | ------------- |
| **FireRedASR2S** | Kaldi fbank  | 80     | 25 ms | 10 ms | Kaldi         |
| NeMo             | Mel (Slaney) | 80     | 25 ms | 10 ms | Slaney        |
| Kaldi/Vosk       | Kaldi mel    | 80     | 25 ms | 10 ms | Kaldi         |
| GigaAM           | GigaAM mel   | 64     | —     | 160   | —             |
| Whisper          | Whisper mel  | 80     | 25 ms | 10 ms | Slaney, log10 |

---

## Conformer Encoder Details

**Source:** `fireredasr2s/fireredasr2/models/module/conformer_encoder.py`

- **Conv2dSubsampling:** 4× subsampling; context 7 (3 left + cur + 3 right).
- **RelPositionalEncoding:** Sinusoidal, extended for relative positions (positive + negative).
- **RelPosEmbConformerBlock:** FFN (Swish, 4× expand) → RelPos MHA → ConformerConvolution (GLU + depthwise conv) → FFN.
- **ConformerConvolution:** Depthwise conv, kernel_size=33; pointwise → GLU → depthwise → Swish → pointwise.

---

## Tokenizer (FireRedASR2-AED)

**Source:** `fireredasr2s/fireredasr2/tokenizer/aed_tokenizer.py`

- **ChineseCharEnglishSpmTokenizer:** Chinese chars as tokens; English via SentencePiece. Space handled via `▁`.
- **TokenDict:** Custom vocabulary for char/SPM pieces.
- Supports timestamp merge for SPM subword alignment.

---

## Reusable Module Mapping for @asrjs/speech-recognition

| FireRedASR2S              | @asrjs/speech-recognition module                                   |
| ------------------------- | ----------------------------------------------- |
| KaldifeatFbank            | audio/kaldi-mel.ts                              |
| Conv2dSubsampling         | model-owned FireRed implementation              |
| ConformerEncoder (RelPos) | model-owned conformer-family implementation     |
| TransformerDecoder        | model-owned seq2seq decoder implementation      |
| Adapter (encoder→LLM)     | model-owned speech-adapter implementation       |
| CTC (for timestamps)      | model-owned CTC head plus shared CTC descriptor |
| DFSMN                     | model-owned DFSMN-family implementation         |

---

## Input / Output Constraints

- **Audio:** 16 kHz, 16-bit mono PCM.
- **FireRedASR2-AED:** Up to 60 s recommended; >200 s can trigger positional encoding errors.
- **FireRedASR2-LLM:** Up to 40 s; batch beam search may cause repetition for very different lengths (use batch_size=1 or length-sorting).

---

## Cross-Reference

- **NeMo:** [NEMO_ASR_MODEL_ARCHITECTURE.md](../NEMO_ASR_MODEL_ARCHITECTURE.md)
- **Whisper:** [WHISPER_ASR_MODEL_ARCHITECTURE.md](../WHISPER_ASR_MODEL_ARCHITECTURE.md)
- **ONNX-ASR:** [ONNX_ASR_MODEL_ARCHITECTURE.md](./ONNX_ASR_MODEL_ARCHITECTURE.md)
- **Transformers:** [TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md](./TRANSFORMERS_ASR_SPEECH_LLM_GUIDE.md)
- **Master index:** [ASR_MODEL_FAMILY_INDEX.md](./ASR_MODEL_FAMILY_INDEX.md)
