# Whisper Splitgraph Node/WASM Validation — Handoff

Date: 2026-05-29
Branch: `feat/asr-pipeline-output-formats`
Latest commits:

- `32f4dac docs: note future whisper beam search plan`
- `144c34e fix: decode fp16 logits in whisper node validation`
- `e4ff0b6 test: add node wasm whisper variant validation`
- `4838d05 fix: use explicit fixture languages for whisper validation`
- `e56f693 fix: make whisper variant prompts comparable`

## Scope completed

This session focused only on local Node CLI / WASM validation for the Whisper 4-graph splitgraph runtime.

Explicitly not done:

- no WebGPU tests
- no browser automation
- no mixed-dtype implementation
- no q4/q4f16 implementation
- no exporter refactor
- no published HF artifact changes
- no beam-search implementation

## Main files touched

| File | Purpose |
|------|---------|
| `tests/smoke/whisper-splitgraph-node-wasm-validate.mjs` | Node CLI validator for fp32/fp16/q8 splitgraph variants |
| `package.json` | `validate:whisper-variants` npm script now runs Node CLI validator in report mode |
| `docs/reports/whisper-large-v3-turbo-variant-validation.md` | Human-readable Node/WASM validation report |
| `docs/reports/whisper-large-v3-turbo-variant-validation.json` | Structured Node/WASM validation output |
| `docs/plans/asr-pipeline-roadmap.md` | Latest/next status, WebGPU manual note, future beam-search design note |
| `docs/whisper-export-workflow.md` | Current export/audit/test/verify workflow |

## What the Node validator covers

The validator is:

```text
tests/smoke/whisper-splitgraph-node-wasm-validate.mjs
```

It validates existing published variants against fp32 baseline:

- `fp32`: onnxruntime-node CPU baseline
- `fp16`: onnxruntime-node CPU
- `q8`: onnxruntime-web WASM CPU

It verifies:

1. fixture discovery under `tests/fixtures`
2. language from filename suffix:
   - `.tr.*` -> Turkish
   - `.en.*` -> English
3. prompt IDs are identical across fp32/fp16/q8 for the same fixture
4. generation controls:
   - `language`
   - `task=transcribe`
   - `no_timestamps=true`
   - `max_new_tokens`
   - `suppress_tokens`
   - `begin_suppress_tokens`
   - greedy `temperature=0`
   - `num_beams=1`
5. token outputs:
   - generated token IDs
   - decoded text
   - EOS behavior
   - token match vs fp32
6. optional alignment path:
   - alignment tensor shape
   - row sums around 1.0
   - non-negative values
   - monotonic DTW timestamps

## Important fixes made

### Prompt fairness

Prompt IDs are built once per fixture from the filename language suffix and reused across variants.

Expected prompts:

- English: `[50258, 50259, 50360, 50364]`
- Turkish: `[50258, 50268, 50360, 50364]`

This prevents comparing fp32/fp16/q8 with different task/language prompts.

### FP16 logits bug

`onnxruntime-node` returns fp16 tensor data as raw half bits (`Uint16Array`). The validator was previously passing those raw bits directly to logit processors and `argmax`, causing garbage fp16 output.

Fix:

- convert fp16 logits to float32 before timestamp/suppress processing and argmax
- convert fp16 alignment tensor data to float32 before row-sum/DTW checks

Result:

- fp16 now matches fp32 exactly on all 5 fixtures at `max_new_tokens=64`

### Frame count handling

Whisper config `max_source_positions=1500` describes encoder output time positions after conv downsampling. Input mel features still require 3000 frames.

The local loader/validator pads to 3000 mel frames and expects encoder/alignment length 1500.

## Current validation commands

Primary command:

```bash
cd /home/steam/github/asrjs/speech-recognition
npm run validate:whisper-variants
```

Direct command:

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

Focused alignment sanity:

```bash
cd /home/steam/github/asrjs/speech-recognition
mkdir -p /tmp/whisper-one-fixture
cp tests/fixtures/jfk2.en.wav /tmp/whisper-one-fixture/

node tests/smoke/whisper-splitgraph-node-wasm-validate.mjs \
  --model-dir /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --fixtures /tmp/whisper-one-fixture \
  --variants fp32 fp16 q8 \
  --max-new-tokens 32 \
  --report /tmp/whisper-align-strict.md
```

## Current results

Generated report:

```text
docs/reports/whisper-large-v3-turbo-variant-validation.md
docs/reports/whisper-large-v3-turbo-variant-validation.json
```

At `max_new_tokens=64`, `--no-align`, `--no-strict`:

| Variant | Result |
|---------|--------|
| fp32 | baseline |
| fp16 | exact token/text/EOS parity vs fp32 on 5/5 fixtures |
| q8 | exact on 3/5 fixtures; diverges on 2/5 in extended greedy decode |

q8 divergences:

- `ItsLifeJim.en.wav`: first diff at token 46. q8 chooses `It is obviously...`; fp32 chooses `Yet it is obviously...`.
- `librivox.org-1600hz.en.wav`: first diff at token 9. fp32 emits EOS after `Preface of A Year with the Birds.`; q8 continues into LibriVox boilerplate and reaches EOS later.

Interpretation:

- Prompt/control parity is correct.
- fp16 runtime validation is green.
- q8 is fairly compared, but strict extended-token parity is not green yet.
- q8 differences are likely quantized decoder behavior, not validation prompt bugs.

## Verification already run

Latest gate from this branch:

```bash
npm test -- tests/whisper-generation-config.test.ts --run
npm run typecheck
npm run lint
npm run build
npm test
npm run validate:whisper-variants
```

Observed:

- focused generation-config test: 13/13 passed
- typecheck passed
- lint passed with 0 errors / 4 existing max-lines warnings
- build passed
- full tests passed: 84 files / 411 tests
- Node/WASM validation report command passed in non-strict mode

Strict all-variant validation currently fails only because q8 extended token parity differs from fp32:

```bash
node tests/smoke/whisper-splitgraph-node-wasm-validate.mjs \
  --model-dir /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph \
  --fixtures tests/fixtures \
  --variants fp32 fp16 q8 \
  --max-new-tokens 64 \
  --no-align \
  --report /tmp/whisper-all-strict.md
```

Expected strict failure:

- `q8/ItsLifeJim.en.wav`: token/text mismatch vs fp32 `(46/64)`
- `q8/librivox.org-1600hz.en.wav`: token/text mismatch vs fp32 `(9/64)`

Strict fp32+fp16 validation passes.

## Export / audit / verify workflow

Canonical workflow doc:

```text
docs/whisper-export-workflow.md
```

Short form:

```bash
cd /home/steam/github/asrjs/speech-recognition/tools/whisper-onnx-export

# export variants
.venv/bin/python export_whisper.py openai/whisper-large-v3-turbo /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph --device cpu --dtype float32 --external-data auto --variant fp32 --output-layout variant-dirs
.venv/bin/python export_whisper.py openai/whisper-large-v3-turbo /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph --device cuda --dtype float16 --external-data auto --variant fp16 --output-layout variant-dirs
.venv/bin/python export_whisper.py openai/whisper-large-v3-turbo /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph --device cpu --dtype float32 --external-data auto --variant q8 --output-layout variant-dirs

# audit publish layout
.venv/bin/python audit_publish.py /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph --variants fp32 fp16 q8 --smoke

# Python/native validation
.venv/bin/python validate_variants.py --model-dir /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph --fixtures ../../tests/fixtures --variants fp32 fp16 q8 --report ../../docs/reports/whisper-large-v3-turbo-variant-validation.md

# Node/WASM runtime validation
cd /home/steam/github/asrjs/speech-recognition
npm run validate:whisper-variants
```

## Latest / next task

Current latest completed task:

- Node CLI / WASM validation strengthened enough to catch prompt/control, fp16 half-logit, EOS, text, token, and optional alignment issues.
- Export/test/verify workflow documented for handoff.

Next task:

- Investigate q8 WASM extended greedy decode divergence vs fp32.

Suggested first debugging target:

1. Add diagnostic mode to validator to dump top-k logits around a target fixture/step.
2. Compare fp32 vs q8 for:
   - `ItsLifeJim.en.wav`, step/token 46
   - `librivox.org-1600hz.en.wav`, step/token 9 EOS decision
3. Determine whether divergence is acceptable quantization behavior or a runtime/logit-processing bug.
4. Do not start WebGPU automation until q8 divergence is understood or explicitly accepted.

## Roadmap order

Use this order going forward:

1. Node/WASM greedy runtime validation
2. Manual WebGPU smoke in browser/app
3. Generation-controls parity gaps
4. Beam search design
5. Beam search implementation
6. Mixed dtype / q4 later

Beam-search design note is documented in `docs/plans/asr-pipeline-roadmap.md`; implementation is not started.
