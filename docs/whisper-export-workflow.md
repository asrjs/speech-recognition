# Whisper 4-Graph Export, Packaging, Publish Audit, and Node/WASM Validation

Status date: 2026-05-29
Branch: `feat/asr-pipeline-output-formats`
Scope: existing Whisper splitgraph export scripts and local Node CLI / WASM validation. WebGPU/browser automation, mixed dtype, q4/q4f16, and beam-search implementation are intentionally deferred.

## Current exported target

Published validation target:

- HF repo: `ysdede/whisper-large-v3-turbo-onnx-4graph`
- Model: `openai/whisper-large-v3-turbo`
- Format: 4-graph splitgraph (`encoder_model.onnx`, `decoder_init.onnx`, `decoder_step.onnx`, `decoder_align.onnx`)
- Variants: `fp32`, `fp16`, `q8`
- Clean publish layout: root config/tokenizer/preprocessor files plus one self-contained subdirectory per variant
- Large-model external data: safe path-based ONNX operations, one `.data` file per graph when external data is required

Variant sizes from the validated publish layout:

| Variant | Approx size | External data | Runtime note |
|---------|-------------|---------------|--------------|
| `fp32/` | 4.5 GB | all 4 graphs | reference/native baseline, not a browser default |
| `fp16/` | 2.3 GB | decoder graphs use `.data`; encoder inline | export-time FP16 only |
| `q8/` | 1.4 GB | none | dynamic int8 quantized compact candidate |

## Export tool location

```bash
cd /home/steam/github/asrjs/speech-recognition/tools/whisper-onnx-export
.venv/bin/python --version
```

Expected environment on Flexo:

- Python 3.12 venv at `tools/whisper-onnx-export/.venv/`
- ONNX Runtime 1.26.0
- Export scripts in `tools/whisper-onnx-export/`

Key scripts:

| Script | Purpose |
|--------|---------|
| `export_whisper.py` | Export one splitgraph variant |
| `audit_publish.py` | Check publish layout, manifest/ONNX agreement, external data, ORT load |
| `validate_variants.py` | Python/native accuracy/perf validation |
| `test_kv_export.py` | Export graph structure/unit checks |
| `test_e2e_tokens.py` | ONNX-vs-PyTorch token checks |
| `test_comprehensive.py` | Real speech/alignment/variant checks |

## Export commands

Use `--output-layout variant-dirs` (default). Each variant is self-contained.

```bash
cd /home/steam/github/asrjs/speech-recognition/tools/whisper-onnx-export

# FP32 reference variant. CPU avoids CUDA memory pressure for large-v3-turbo.
.venv/bin/python export_whisper.py openai/whisper-large-v3-turbo \
  /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --device cpu \
  --dtype float32 \
  --external-data auto \
  --variant fp32 \
  --output-layout variant-dirs

# FP16 variant. Export-time FP16 only; do not use post-export FP16 conversion.
.venv/bin/python export_whisper.py openai/whisper-large-v3-turbo \
  /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --device cuda \
  --dtype float16 \
  --external-data auto \
  --variant fp16 \
  --output-layout variant-dirs

# q8 variant. Dynamic int8 quantization from fp32 export path.
.venv/bin/python export_whisper.py openai/whisper-large-v3-turbo \
  /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --device cpu \
  --dtype float32 \
  --external-data auto \
  --variant q8 \
  --output-layout variant-dirs
```

Notes:

- `--external-data auto` is the default and is required for safe large-model export. It detects large encoders as well as large decoders.
- `--variant fp16` auto-aligns with export-time FP16. Post-export FP16 via `onnxconverter_common.float16` is broken for these graphs because ORT hits Cast type mismatches.
- `--variant q8` is the current preferred name. Older `int8-dynamic` terminology may appear in scripts/docs as an alias, but the publish layout uses `q8/`.
- `--external-data-one-file true` is the default. It repacks per-weight external-data files from `torch.onnx.export` into one `<graph>.onnx.data` file per graph.

## Expected publish layout

```text
whisper-large-v3-turbo-onnx-4graph/
├── README.md
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── tokenizer.json
├── fp32/
│   ├── manifest.json
│   ├── encoder_model.onnx
│   ├── encoder_model.onnx.data
│   ├── decoder_init.onnx
│   ├── decoder_init.onnx.data
│   ├── decoder_step.onnx
│   ├── decoder_step.onnx.data
│   ├── decoder_align.onnx
│   └── decoder_align.onnx.data
├── fp16/
│   ├── manifest.json
│   ├── encoder_model.onnx
│   ├── decoder_init.onnx
│   ├── decoder_init.onnx.data
│   ├── decoder_step.onnx
│   ├── decoder_step.onnx.data
│   ├── decoder_align.onnx
│   └── decoder_align.onnx.data
└── q8/
    ├── manifest.json
    ├── encoder_model.onnx
    ├── decoder_init.onnx
    ├── decoder_step.onnx
    └── decoder_align.onnx
```

Each variant dir also contains copied tokenizer/config/preprocessor files required by local loaders.

## Publish audit

Run audit before upload and after any local packaging change:

```bash
cd /home/steam/github/asrjs/speech-recognition/tools/whisper-onnx-export

.venv/bin/python audit_publish.py \
  /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --variants fp32 fp16 q8 \
  --smoke
```

Audit expectations:

- 0 failures
- no tensor-named external files remain in published layout
- every manifest path exists
- every ONNX external-data location/offset/length agrees with manifest metadata
- ONNX checker uses safe path-based validation
- ORT can load every graph
- SHA256 metadata is stable

## Python/native variant validation

Python/native validation remains useful for artifact sanity and speed/accuracy reports:

```bash
cd /home/steam/github/asrjs/speech-recognition/tools/whisper-onnx-export

.venv/bin/python validate_variants.py \
  --model-dir /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --fixtures ../../tests/fixtures \
  --variants fp32 fp16 q8 \
  --report ../../docs/reports/whisper-large-v3-turbo-variant-validation.md
```

Prompt fairness rule:

- Resolve fixture language from filename suffix first: `.tr.*` -> Turkish, `.en.*` -> English.
- Build one prompt token sequence per fixture.
- Reuse the same prompt IDs across `fp32`, `fp16`, and `q8`.
- Report prompt IDs before making accuracy conclusions.

Expected prompt IDs:

- English: `[50258, 50259, 50360, 50364]`
- Turkish: `[50258, 50268, 50360, 50364]`

## Node CLI / WASM validation

Primary runtime validation now lives in:

```text
tests/smoke/whisper-splitgraph-node-wasm-validate.mjs
```

Run through npm:

```bash
cd /home/steam/github/asrjs/speech-recognition

WHISPER_VARIANT_DIR=/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
WHISPER_MAX_NEW_TOKENS=64 \
npm run validate:whisper-variants
```

Equivalent direct command:

```bash
npm run build
node tests/smoke/whisper-splitgraph-node-wasm-validate.mjs \
  --model-dir /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --fixtures tests/fixtures \
  --variants fp32 fp16 q8 \
  --max-new-tokens 64 \
  --no-align \
  --no-strict \
  --report docs/reports/whisper-large-v3-turbo-variant-validation.md
```

What it verifies:

- fixture language suffixes: `.tr.*` -> Turkish, `.en.*` -> English
- prompt IDs match across variants for the same fixture
- generation controls: `language`, `task=transcribe`, `no_timestamps`, `max_new_tokens`, `suppress_tokens`, `begin_suppress_tokens`, greedy `temperature=0`, `num_beams=1`
- token IDs, decoded text, EOS behavior, exact token match vs fp32 baseline
- optional alignment path: shape, row sums, non-negative values, monotonic DTW timestamps

Backend policy in the current validator:

- `fp32`: onnxruntime-node CPU baseline. Large fp32 exceeds practical WASM memory on this host.
- `fp16`: onnxruntime-node CPU. The validator converts float16 logits/alignment tensors back to float32 before logit processing and argmax.
- `q8`: onnxruntime-web WASM CPU.

Alignment validation is available, but full large-variant alignment over every fixture is slow. Use a focused fixture for sanity:

```bash
mkdir -p /tmp/whisper-one-fixture
cp tests/fixtures/jfk2.en.wav /tmp/whisper-one-fixture/

node tests/smoke/whisper-splitgraph-node-wasm-validate.mjs \
  --model-dir /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --fixtures /tmp/whisper-one-fixture \
  --variants fp32 fp16 q8 \
  --max-new-tokens 32 \
  --report /tmp/whisper-align-strict.md
```

## Current Node/WASM validation results

Current generated report:

```text
docs/reports/whisper-large-v3-turbo-variant-validation.md
docs/reports/whisper-large-v3-turbo-variant-validation.json
```

At `max_new_tokens=64`, `--no-align`, `--no-strict`:

| Variant | Prompt/control parity | Token/text parity vs fp32 | Status |
|---------|-----------------------|---------------------------|--------|
| fp32 | baseline | baseline | pass |
| fp16 | pass | exact on 5/5 fixtures | pass |
| q8 | pass | exact on 3/5 fixtures | investigate |

Known q8 divergences:

- `ItsLifeJim.en.wav`: first token difference at token 46; wording diverges while the prefix remains close.
- `librivox.org-1600hz.en.wav`: fp32 emits EOS after the title; q8 continues into LibriVox boilerplate and reaches EOS later.

These are now real q8/runtime differences, not prompt-language artifacts.

## TypeScript/runtime gate

Run before handoff or push:

```bash
cd /home/steam/github/asrjs/speech-recognition
npm run typecheck
npm run lint
npm test
npm run build
npm run validate:whisper-variants
```

Latest verified gate:

- `npm run typecheck` passed
- `npm run lint` passed with 0 errors / 4 existing max-lines warnings
- `npm test` passed: 84 files / 411 tests
- `npm run build` passed
- `npm run validate:whisper-variants` passed in non-strict reporting mode

## HF upload

Only upload after audit and validation are accepted:

```bash
hf upload ysdede/whisper-large-v3-turbo-onnx-4graph \
  /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph .
```

Current task explicitly did not change published HF artifacts.

## Deferred / manual steps

- WebGPU smoke is intentionally not automated here. After Node/WASM validation passes, WebGPU should be tested manually in the browser/app.
- Beam search is not implemented yet. Design note is in `docs/plans/asr-pipeline-roadmap.md`.
- Mixed dtype and q4/q4f16 are deferred.
- External benchmark datasets are deferred until local runtime validation is stable.

## Next task

Investigate q8 WASM extended greedy decode divergence against fp32. Start with logits/top-k inspection around:

- `ItsLifeJim.en.wav`, token step 46
- `librivox.org-1600hz.en.wav`, token step 9 / EOS decision

Do not begin WebGPU automation until q8 divergence is understood or explicitly accepted.
