# Reusable Components, Data Types, and Adapters

Based on the architectural review of NeMo, Whisper, MedASR, FireRedASR2S, and ONNX models, the `asr.js` library should be structured to maximize reuse at the **algorithm and topology level** while strictly isolating vendor-specific feature extraction and projection constraints.

This document serves as the implementation guideline for decomposing models into shareable elements. In the current single-package `asr.js` layout, reusable architecture descriptors belong in `src/inference/graph.ts`, while real encoder and decoder implementations remain inside `src/models/*` until they are proven shared.

---

## 1. System Data Types (Contracts)

Data types act as common boundaries allowing interchangeable parts.

| Type Name | Format | Role |
|-----------|--------|------|
| **RawAudio** | `Float32Array` \| `TensorMap` | Uncompressed PCM data, strictly 16 kHz mono. Input to all pipelines. |
| **AcousticFeatures** | `TensorMap` `[B, n_features, T_frames]` | Feature extraction output. Can be mel bins (e.g., 80) or convolutional channels (e.g., 512). |
| **EncodedSequence** | `TensorMap` `[B, T_enc, d_model]` | Acoustic model (encoder) latents. Subsampled contextually enriched vectors. |
| **TopologyLogits** | `TensorMap` `[B, T_dec, V(+1)]` | Distribution over vocabulary. Rank depends on topology (CTC is 3D, RNNT Joint is 4D). |
| **TokenSequence** | `Int32Array` \| `number[]` | Sequence of token IDs before detokenization. |
| **Transcript** | `object` | Final output. Contains `text` (or `utterance_text`), `words` (with timestamps/confidences), `tokens` (including `tokenIds`, `frameIndices`/`frameIds`, `logProbs`, `tdtSteps`), `logits`, and `metrics` (e.g., `preprocess_ms`, `encode_ms`, `decode_ms`, `rtf`). Supports `previousDecoderState` for streaming. |

---

## 2. Reusable Elements Taxonomy

Instead of grouping by branded family (e.g., `nemo-package`, `medasr-package`), elements are grouped by their technical role.

### A. Feature Extractors (Processors)
Mathematical implementations differ too widely to share logic. They share the boundary `RawAudio -> AcousticFeatures`.
- `processors/nemo-mel.ts`: 80-bin, Slaney norm, per_feature norm (NeMo, Parakeet).
- `processors/kaldi-mel.ts`: 80-bin, Hann window^0.85 (FireRed, Vosk/Kaldi).
- `processors/whisper-mel.ts`: 80-bin, log10, clamp max-8, (x+4)/4 (Whisper).
- `processors/wav2vec2-conv.ts`: 7-layer 1D conv directly on `RawAudio` (MedASR).
- `processors/identity.ts`: Pass-through raw chunks (T-one).

### B. Acoustic Encoders
Expects `[B, n_features, T_frames]` and outputs `[B, T_enc, d_model]`.
- Shared classification lives in `src/inference/graph.ts` via encoder descriptors like `fastconformer`, `wav2vec2-conformer`, `whisper-transformer`, and `dfsmn`.
- Concrete implementations stay inside model-family modules such as `src/models/nemo-tdt`, `src/models/hf-ctc-common`, and `src/models/whisper-seq2seq`.

### C. Decoder Topologies (Heads)
Neural mapping from `d_model` to vocabulary space. Shareable semantics are described centrally, but the math stays model-owned until it is truly reused.
- Shared classification lives in `src/inference/graph.ts` via decoder-head descriptors like `ctc-head`, `transducer-tdt`, `transformer-decoder`, and `speech-llm`.
- Concrete prediction nets, joiners, and language-model heads live under `src/models/*`.

### D. Search Strategies (Decoding Logic)
Algorithmic logic yielding token sequences. **No neural weights**.
- Shared classification lives in `src/inference/graph.ts` via strategy descriptors like `ctc-greedy`, `tdt-greedy`, and `aed-generate`.
- Concrete decoding loops remain model-owned until at least two families can share the same implementation without leaking family-specific assumptions.

### E. Tokenizers
Mappings from integers to text strings.
- `tokenizers/sentencepiece.ts` (NeMo models)
- `tokenizers/tiktoken.ts` (Whisper)
- `tokenizers/char.ts` (Simple character index)

---

## 3. Adapters and Presets (Wiring)

Models implemented in `src/models/` should stay focused and readable, but they are allowed to own real implementation logic when that logic is not yet proven reusable across families.

### Case 1: Standard Wiring Pattern
*Google MedASR* (`src/models/hf-ctc-common`):
```ts
// Wires:
1. wav2vec2-conv.ts (Feature Extraction)
2. model-owned Wav2Vec2-Conformer encoder
3. model-owned CTC head
4. model-owned CTC search using shared transcript semantics
```

### Case 2: The LLM Adapter Pattern (Speech LLM)
When bridging an acoustic encoder to a Large Language Model (e.g., `FireRedASR2-LLM`, `Voxtral`), we introduce an explicit `Adapter`.

- **Adapter (model-owned speech adapter)**: Downsamples and linearly projects the `Enc[B, T_enc, d_model]` tensor into the target `LLM[B, T_enc/ds, llm_dim]` latent space.
- **Substitution**: The projected speech features dynamically replace a designated `<speech>` token in the text prompt's embeddings.
- **Decoding**: Handled by a standard causal LLM generation loop instead of an ASR-specific cross-attention decoder.

### Case 3: Auxiliary Tasks (VAD / LID / Punc)
- **VAD** (`FireRedVAD`): Reuses feature extraction (`kaldi-mel.ts`) + DFSMN-style encoder family + threshold logic. No search.
- **LID** (`FireRedLID`): Reuses feature extraction (`kaldi-mel.ts`) + conformer-family encoder + transformer-style decoder with `max_len=2` beam search.

---

## Implementation Rule of Thumb

1. **If math differs**, do not merge. Keep `nemo-mel` and `kaldi-mel` cleanly separated inside `processors/`.
2. **If only hyper-parameters differ**, parameterize. A single descriptor family can classify Conformer, FastConformer, and related variants without forcing a premature shared implementation.
3. **If logic is algorithmic and truly shared**, lift it out. Until then, keep model-specific loops in `src/models/*` and share only the architecture descriptor in `src/inference/graph.ts`.
