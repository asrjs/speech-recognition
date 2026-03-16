# Canary 180M Flash Porting Reference

Reference tooling for porting `nvidia/canary-180m-flash` into
`@asrjs/speech-recognition`.

These scripts are intended to be run from a Python environment with NeMo ASR
installed. In this workspace, the `nemo` conda environment is the expected
baseline.

For the full step-by-step workflow and lessons learned, start with:

- [canary-aed-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\canary-aed-porting.md)

## Environment Check

Validate the shared NeMo environment before running anything else:

```powershell
conda run -n nemo python -c "import nemo.collections.asr, onnx, onnxruntime, soundfile, torch; print('nemo env ok')"
```

## Heavy Artifact Policy

The large saved reference JSON for Canary may be committed as
`canary-180m-flash-reference.json.gz` instead of raw JSON to keep pull request
diffs and AI review context smaller.

Check artifact status:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs status --target canary
```

Restore the raw JSON locally when a script or manual inspection needs it:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs unpack --target canary
```

Pack it back into `.json.gz` and delete the raw extracted file:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs pack --target canary --delete-originals
```

## Scripts

- `generate_canary_reference.py`
  - Runs the original NeMo model end to end on a fixture audio file.
  - Captures preprocessor features, encoder states, prompt ids, generated token
    ids, and final text in JSON.
- `export_canary_onnx.py`
  - Exports the Canary encoder wrapper, the autoregressive decoder step
    wrapper, tokenizer metadata, and runtime config metadata.
  - Can also attempt preprocessor export, but that is optional for the current
    port.
- `convert_onnx_fp16.py`
  - Converts exported FP32 ONNX models to FP16 while keeping IO types stable.
- `convert_onnx_int8.py`
  - Produces dynamic-quantized INT8 encoder and decoder variants for smoke and
    parity testing.
- `verify_canary_onnx.py`
  - Replays the exported ONNX artifacts against a saved PyTorch reference JSON
    and reports tensor/token/text parity for whichever encoder and decoder
    filenames you choose.

## Recommended Flow

### 1. Generate a PyTorch reference JSON

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/generate_canary_reference.py `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --output tools/data/results/canary/canary-180m-flash-reference.json
```

This JSON is the source of truth for:

- prompt ids
- feature shapes and lengths
- encoder output shapes
- manual greedy token ids
- normalized final text

If you want to keep the repo lightweight after regenerating it, pack it back
into `.json.gz`:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs pack --target canary --delete-originals
```

### 2. Export the runtime artifacts we actually need

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/export_canary_onnx.py `
  --output-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --skip-preprocessor `
  --device cpu
```

Expected outputs in `N:\models\onnx\nemo\canary-180m-flash-smoke`:

- `encoder-model.onnx`
- `decoder-model.onnx`
- `tokenizer.json`
- `config.json`

### 3. Produce FP16 variants

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/convert_onnx_fp16.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --artifacts encoder-model.onnx decoder-model.onnx
```

Expected additional outputs:

- `encoder-model.fp16.onnx`
- `decoder-model.fp16.onnx`

### 4. Produce INT8 variants

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/convert_onnx_int8.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke
```

Expected additional outputs:

- `encoder-model.int8.onnx`
- `decoder-model.int8.onnx`

### 5. Verify FP32 parity first

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify.json
```

### 6. Verify FP16 parity

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.fp16.onnx `
  --decoder-name decoder-model.fp16.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-fp16.json
```

### 7. Verify INT8 combinations independently

Decoder INT8 only:

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.onnx `
  --decoder-name decoder-model.int8.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-decoder-int8.json
```

Encoder INT8 only:

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.int8.onnx `
  --decoder-name decoder-model.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-encoder-int8.json
```

Full INT8:

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.int8.onnx `
  --decoder-name decoder-model.int8.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-int8.json
```

Current smoke-fixture result summary:

- FP32: exact token/text match
- FP16: exact token/text match
- encoder INT8 only: exact token/text match
- decoder INT8 only: exact token/text match
- full INT8: not exact, so it should not be the default artifact pairing

## Artifact Contract

Possible export files:

- required:
  - `encoder-model.onnx`
  - `decoder-model.onnx`
  - `tokenizer.json`
  - `config.json`
- optional:
  - `nemo128.onnx`

The TypeScript runtime currently supports keeping the frontend separate from
the encoder and decoder artifacts.

## Frontend Guidance

Do not treat ONNX mel export as mandatory work for Canary.

In this environment, the preprocessor export still fails on the NeMo STFT
complex path. More importantly, we already have strong JS-side frontend work in:

- `N:\github\ysdede\parakeet.js`
- `N:\github\ysdede\meljs`

Recommended rule for future ports:

1. lock down NeMo reference JSON first
2. prove encoder and decoder parity second
3. compare any JS frontend directly against the saved reference features
   with [node-canary-js-frontend-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-canary-js-frontend-parity.mjs)
4. only keep a dedicated ONNX preprocessor if it clearly improves the overall
   artifact/runtime tradeoff

Current saved comparisons:

- `tools/data/results/canary/canary-180m-flash-js-frontend-parity-asrjs.json`
- `tools/data/results/canary/canary-180m-flash-js-frontend-parity-meljs.json`
- `tools/data/results/canary/canary-180m-flash-js-frontend-parity-parakeetjs.json`
- `tools/data/results/canary/canary-180m-flash-end-to-end-js-vs-onnx-int8.json`

Current status:

- `asrjs` now records `validLengthMode: "centered"` in the saved parity JSON,
  matches the NeMo reference frame count on the smoke fixture, and is the
  default Canary preset frontend in this repo.
- The saved INT8 smoke comparison shows `asrjs` JS frontend matching the
  reference transcript text while the shared ONNX frontend path missed the comma
  after `And so,` on the same fixture.
- `meljs` and `parakeet.js` remain useful external references when comparing
  implementation ideas or regressions, but they still follow the ONNX-style
  valid-length contract on this fixture and therefore report `1100` valid frames
  instead of the Canary reference `1101`.
