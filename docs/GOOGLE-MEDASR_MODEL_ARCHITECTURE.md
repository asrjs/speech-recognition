# Google MedASR Model Architecture

Reference model: `google/medasr`

This repo uses `lasr-ctc` as the technical family name for MedASR-style execution.

Important distinction:

- research lineage: MedASR is a LASR/LAST-trained medical ASR model built around Conformer + CTC
- runtime implementation in this repo: the current port executes the exported ONNX artifact path used by `ysdede/medasr-onnx`
- branded preset: `medasr`
- technical family: `lasr-ctc`

---

## 1. Runtime View in `@asrjs/speech-recognition`

The current MedASR port in this repo is modeled as:

```text
PCM audio [16 kHz mono]
  ->
kaldi-style log-mel frontend [128 bins]
  ->
Conformer encoder
  ->
CTC logits
  ->
greedy CTC decode
  ->
MedASR tokenizer + transcript normalization
```

This means the reusable pieces inside `src/models/lasr-ctc` are:

- LASR CTC config and model-family wiring
- MedASR tokenizer loading and decoding
- CTC collapse, timing reconstruction, and sentence timing helpers
- ONNX Runtime session and artifact resolution
- JS mel frontend bridge used by the exported ONNX path

---

## 2. Why the Family Is Named `lasr-ctc`

`lasr-ctc` is the implementation boundary for:

- LASR/MedASR-style Conformer + CTC execution
- shared CTC transcript contracts
- MedASR-specific tokenizer and exported-artifact wiring

It is intentionally not named after Hugging Face hosting, because Hugging Face is only one artifact source. In code:

- `source.kind: 'huggingface'` stays valid for asset resolution
- architecture, family ids, types, transcript normalizers, and warning codes use `lasr-ctc`

---

## 3. Source Tree Mapping

```text
src/
├── inference/
│   └── descriptors.ts          # shared conformer / ctc descriptors
├── models/
│   ├── nemo-tdt/               # NeMo FastConformer + transducer/TDT
│   ├── lasr-ctc/               # MedASR-style Conformer + CTC family
│   └── whisper-seq2seq/        # Whisper encoder-decoder family
└── presets/
    └── medasr/                 # branded MedASR preset -> lasr-ctc
```

Inside `src/models/lasr-ctc`:

- `config.ts`: family defaults and descriptions
- `types.ts`: native transcript, artifact source, and executor contracts
- `ctc.ts`: greedy collapse and timing helpers
- `tokenizer.ts`: MedASR text tokenizer
- `mel.ts`: JS mel frontend used by the current ONNX path
- `ort.ts` / `executor.ts`: ORT session creation and artifact-backed inference
- `model.ts`: family registration, session lifecycle, and transcript shaping

---

## 4. Shared vs Family-Owned

Shared at descriptor level:

- conformer encoder classification
- CTC decoder-head classification
- CTC greedy decoding classification
- canonical transcript contracts

Family-owned in `lasr-ctc`:

- MedASR tokenizer behavior
- timing reconstruction details for this family's native transcript
- artifact resolution for exported MedASR ONNX bundles
- current mel frontend bridge used by the ONNX export path

This keeps the runtime modular without pretending MedASR is identical to NeMo or Whisper.

---

## 5. Current Preset Wiring

`src/presets/medasr` stays intentionally thin:

- resolves MedASR preset ids and aliases
- injects MedASR classification defaults
- injects default artifact source (`ysdede/medasr-onnx`)
- resolves to `family: 'lasr-ctc'`

Execution remains owned by `src/models/lasr-ctc`.

---

## 6. Design Rule

When future MedASR-adjacent models arrive:

- reuse `lasr-ctc` only if they fit the same Conformer + CTC + tokenizer/executor contract well
- keep new model-family logic separate if the frontend, topology, or session contract materially changes
- move helpers upward only after a second real family proves the reuse

---

## 7. Cross-Reference

- `NEMO_ASR_MODEL_ARCHITECTURE.md`: NeMo mel + FastConformer + RNNT/TDT/CTC
- `WHISPER_ASR_MODEL_ARCHITECTURE.md`: Whisper mel + transformer encoder-decoder
- `architecture.md`: runtime family vs preset layering
