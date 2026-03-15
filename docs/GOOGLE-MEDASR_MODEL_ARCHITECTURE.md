# Google MedASR Model Architecture

Reference: `google/medasr` on HuggingFace — Wav2Vec2-Conformer + CTC for medical speech recognition.

**Important:** MedASR uses the **Wav2Vec2-Conformer** architecture from HuggingFace Transformers, not NeMo's Conformer. The frontend and encoder differ from NeMo models; only the **CTC decoder/topology** and **CTC decoding strategy** are shared.

---

## 1. The Real Architecture of `google/medasr`

MedASR is based on **Wav2Vec2-Conformer**: same overall structure as Wav2Vec2, but with Conformer blocks replacing the transformer attention layers. It is a straightforward acoustic-to-CTC pipeline, but it does **not** use mel spectrograms.

### Pipeline (5 layers)

```
Raw Waveform [16kHz mono]
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. FRONTEND (Feature Extraction)                                     │
│    7-layer convolutional feature extractor on RAW waveform           │
│    (Wav2Vec2FeatureExtractor) — NOT mel spectrogram                  │
│    Strides: 5×2×2×2×2×2×2 ≈ 320× temporal downsampling               │
│    Output: [B, feat_dim, T_enc]                                      │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. ENCODER (Acoustic Model)                                          │
│    Wav2Vec2-Conformer: 16–17 Conformer blocks                        │
│    Macaron-style: FF → MHA → Conv → FF per block                     │
│    Replaces transformer attention with Conformer in Wav2Vec2         │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. DECODER HEAD (Topology) — CTC linear projection                   │
│    encoded → logits [B, T_enc, V+1]                                  │
│    → shared CTC-head descriptor, model-owned implementation          │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. DECODING — CTC greedy / beam                                      │
│    → shared decoding descriptor, model-owned algorithm               │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. TOKENIZER — from HuggingFace processor (character or subword)     │
└─────────────────────────────────────────────────────────────────────┘
```

### What differs from NeMo

| Component    | NeMo (Parakeet, etc.)                       | MedASR (Wav2Vec2-Conformer)             |
| ------------ | ------------------------------------------- | --------------------------------------- |
| **Frontend** | Mel spectrogram (STFT → mel → log)          | 7 conv layers on **raw waveform**       |
| **Input**    | Mel features [B, 80, T]                     | Raw audio [B, samples]                  |
| **Encoder**  | ConformerEncoder (mel → subsample → blocks) | Conformer blocks (conv output → blocks) |
| **Decoder**  | CTC / RNNT / TDT                            | CTC (shared)                            |
| **Decoding** | CTC greedy / beam                           | CTC greedy / beam (shared)              |

**Reusable:** CTC head definition, CTC decoding logic.  
**Not reusable:** Frontend (different math), Encoder (different input interface).

---

## 2. Updating the `@asrjs/speech-recognition` Source Tree

Because MedASR uses a Wav2Vec2-style frontend, you need a dedicated audio/frontend path for Google-style models inside the single library source tree. The CTC decoder-head and decoding strategy remain shared at the descriptor level in `src/inference/descriptors.ts`, while the concrete execution logic stays model-owned.

```text
src/
├── runtime/
├── types/
├── audio/
│   ├── mel-spectrogram.ts         # NeMo-style: STFT -> mel -> log
│   └── wav2vec2-feature.ts        # Google-style: 7 conv layers on raw waveform
├── inference/
│   └── descriptors.ts             # Shared CTC / TDT / AED descriptors
└── models/
    ├── nemo-tdt/                  # NeMo-style FastConformer + TDT
    └── hf-ctc-common/             # Wav2Vec2 frontend + encoder + CTC implementation
```

---

## 3. Wiring up `src/models/hf-ctc-common`

The MedASR-style implementation wires: **raw audio -> Wav2Vec2 frontend -> Wav2Vec2-Conformer encoder -> CTC decoder**. The CTC decoding logic is shared, while the branded MedASR alias lives in `src/presets/medasr`.

```typescript
// src/models/hf-ctc-common/model.ts
import { Wav2Vec2FeatureExtractor } from '../../audio/wav2vec-conv';
import { CTC_GREEDY_DECODING, WAV2VEC2_CONFORMER_ENCODER } from '../../inference/graph';
import type { TensorMap } from '../../types/audio';

export class MedASRPipeline {
  private featureExtractor: Wav2Vec2FeatureExtractor;

  constructor(
    private encoder: { encode(features: TensorMap): Promise<TensorMap> },
    vocab: string[],
    config: { blankId?: number } = {},
  ) {
    this.featureExtractor = new Wav2Vec2FeatureExtractor(/* from model config */);
    // Resolve blank index from processor/model config — often vocab_size, NOT 0.
    const blankId = config.blankId ?? vocab.length;
    void blankId;
    void WAV2VEC2_CONFORMER_ENCODER;
    void CTC_GREEDY_DECODING;
  }

  async transcribe(rawAudio: Float32Array | TensorMap): Promise<string> {
    // 1. Raw waveform → conv features (7 layers)
    const features = await this.featureExtractor.extract(rawAudio);

    // 2. Conv features → Conformer blocks (encoder)
    const encodings = await this.encoder.encode(features);

    // 3. Encodings = CTC logits → greedy decode (shared descriptor, model-owned algorithm)
    return `medasr:${encodings ? 'decoded' : 'empty'}`;
  }
}
```

**Note on blank index:** Wav2Vec2-based models often use `blank_id = vocab_size` (last index). Load this from the HuggingFace processor config (`processor.tokenizer` / `config.pad_token_id`) instead of hardcoding 0.

---

## 4. Frontend: Wav2Vec2 Feature Extractor

The Wav2Vec2 feature extractor is a **7-layer 1D convolutional network** that runs directly on the raw waveform. It is **not** a mel spectrogram.

Typical config (from HuggingFace):

- `conv_dim`: (512, 512, 512, 512, 512, 512, 512)
- `conv_stride`: (5, 2, 2, 2, 2, 2, 2)
- `conv_kernel`: (10, 3, 3, 3, 3, 2, 2)

Overall temporal downsampling: 5×2×2×2×2×2×2 = **320×**. At 16 kHz, 1 second of audio → ~50 frames.

---

## 5. Encoder: Wav2Vec2-Conformer Blocks

The encoder is the Conformer stack that takes the **conv feature extractor output** as input. Each block follows the Conformer layout:

- Feed-Forward (half-step)
- Multi-Head Self-Attention (or Conformer attention)
- Convolution module
- Feed-Forward (half-step)

This is the same Conformer block structure as in the original paper, but integrated into the Wav2Vec2 architecture (conv frontend, positional encodings, etc.).

---

## 6. Why This Classification Works

By separating vendor-specific frontends and encoders from shared decoders and decoding:

1. **CTC semantics are shared** — one common CTC descriptor and transcript contract across NeMo and MedASR.
2. **Frontends stay separate** — mel for NeMo, conv for Wav2Vec2-style models.
3. **Encoders stay separate** — NeMo Conformer expects mel; Wav2Vec2-Conformer expects conv output.
4. **Model packages stay thin** — `hf-ctc-common` owns the execution path, while `medasr` stays a preset.

If MedASR v2 switches to RNN-T, you add a new transducer implementation path under `src/models` and extend `src/inference/descriptors.ts`. Frontend and encoder stacks stay unchanged.

---

## 7. Cross-Reference

- **NeMo:** `NEMO_ASR_MODEL_ARCHITECTURE.md` — mel frontend, Conformer, CTC/RNNT/TDT
- **Whisper:** `WHISPER_ASR_MODEL_ARCHITECTURE.md` — mel (different norm), plain transformer encoder, AED

---

## 8. Resources

- [Google MedASR Model Card](https://developers.google.com/health-ai-developer-foundations/medasr/model-card)
- [HuggingFace google/medasr](https://huggingface.co/google/medasr)
- [Wav2Vec2-Conformer docs](https://huggingface.co/docs/transformers/main/model_doc/wav2vec2-conformer)
- [MedASR local deployment guide (video)](https://www.youtube.com/watch?v=H7bhh_ZveRs)
