# Canary AED Porting Workflow

Use this playbook when porting `nvidia/canary-180m-flash` or a similar
NeMo multitask AED model into `@asrjs/speech-recognition`.

This is a concrete, verified workflow based on the current Canary 180M Flash
port in this repo.

## Goals

1. Capture a trustworthy end-to-end NeMo reference first.
2. Export only the ONNX pieces we actually need for runtime execution.
3. Verify prompt ids, token ids, and final text before tuning anything else.
4. Treat encoder, decoder, tokenizer, and preprocessor as separable workstreams.
5. Keep a written record of which quantization combinations are safe.

## Prerequisites

Expected local paths in this environment:

- repo:
  - `N:\github\asrjs\speech-recognition`
- optional NeMo source for inspection:
  - `N:\github\ysdede\NeMo`
- optional JS frontend references:
  - `N:\github\ysdede\parakeet.js`
  - `N:\github\ysdede\meljs`
- working model export directory:
  - `N:\models\onnx\nemo\canary-180m-flash-smoke`

Expected Python environment:

- conda env name:
  - `nemo`
- key packages:
  - `nemo_toolkit`
  - `torch`
  - `onnx`
  - `onnxruntime`
  - `soundfile`
  - `scipy`

## Step 0. Validate The NeMo Environment

Before touching the model, confirm the expected conda environment can import
the packages used by the reference scripts:

```powershell
conda run -n nemo python -c "import nemo.collections.asr, onnx, onnxruntime, soundfile, torch; print('nemo env ok')"
```

If this fails, fix the environment first. The rest of the workflow assumes the
reference scripts can run against a working NeMo ASR install.

## Reference Files

Core scripts:

- [generate_canary_reference.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\canary-180m-flash\generate_canary_reference.py)
- [export_canary_onnx.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\canary-180m-flash\export_canary_onnx.py)
- [convert_onnx_fp16.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\canary-180m-flash\convert_onnx_fp16.py)
- [convert_onnx_int8.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\canary-180m-flash\convert_onnx_int8.py)
- [verify_canary_onnx.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\canary-180m-flash\verify_canary_onnx.py)

Reference outputs from this port:

- [canary-180m-flash-reference.json](N:\github\asrjs\speech-recognition\tools\data\results\canary\canary-180m-flash-reference.json)
- [canary-180m-flash-onnx-verify.json](N:\github\asrjs\speech-recognition\tools\data\results\canary\canary-180m-flash-onnx-verify.json)
- [canary-180m-flash-onnx-verify-fp16.json](N:\github\asrjs\speech-recognition\tools\data\results\canary\canary-180m-flash-onnx-verify-fp16.json)
- [canary-180m-flash-onnx-verify-encoder-int8.json](N:\github\asrjs\speech-recognition\tools\data\results\canary\canary-180m-flash-onnx-verify-encoder-int8.json)
- [canary-180m-flash-onnx-verify-decoder-int8.json](N:\github\asrjs\speech-recognition\tools\data\results\canary\canary-180m-flash-onnx-verify-decoder-int8.json)
- [canary-180m-flash-onnx-verify-int8.json](N:\github\asrjs\speech-recognition\tools\data\results\canary\canary-180m-flash-onnx-verify-int8.json)

## Step 1. Create A Working Branch

```powershell
git checkout -b feat/canary-180m-flash
```

## Step 2. Capture A NeMo Reference First

Run the original model end to end before exporting or rewriting anything:

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/generate_canary_reference.py `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --output tools/data/results/canary/canary-180m-flash-reference.json
```

What this gives you:

- waveform tensor payload
- preprocessor features and feature lengths
- encoder states, lengths, and mask
- resolved prompt slots and prompt ids
- manual greedy decoder token ids, token pieces, log probs, and text
- normalized transcript text

What to inspect immediately in the JSON:

- `runtime_config.family`
  - expected: `nemo-aed`
- `runtime_config.preset`
  - expected: `canary`
- `runtime_config.prompt_format`
  - expected: `canary2`
- `prompt.ids`
  - current default ASR prompt: `[7, 4, 16, 62, 62, 5, 9, 11, 13]`
- `decode.manual_greedy.text`
  - use this as the text parity target

## Step 3. Export ONNX Wrappers

Export the pieces the runtime actually needs:

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/export_canary_onnx.py `
  --output-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --skip-preprocessor `
  --device cpu
```

Current expected outputs:

- `encoder-model.onnx`
- `decoder-model.onnx`
- `tokenizer.json`
- `config.json`

Preprocessor note:

- this recommended flow skips preprocessor export on purpose
- for Canary, encoder/decoder parity is the main deliverable
- JS feature extraction is a first-class option for future runtime work

Optional experiment:

- if you explicitly want to test the NeMo frontend export path, rerun the same
  command without `--skip-preprocessor`
- in this environment, `nemo128.onnx` still fails because PyTorch ONNX export
  does not reliably support the NeMo STFT complex path
- this is not a blocker for the port

## Step 4. Produce FP16 Variants

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/convert_onnx_fp16.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --artifacts encoder-model.onnx decoder-model.onnx
```

Expected outputs:

- `encoder-model.fp16.onnx`
- `decoder-model.fp16.onnx`

## Step 5. Produce INT8 Variants

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/convert_onnx_int8.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke
```

Expected outputs:

- `encoder-model.int8.onnx`
- `decoder-model.int8.onnx`

Current smoke-export size reduction from this environment:

- encoder:
  - `439.95 MB -> 126.34 MB`
- decoder:
  - `301.50 MB -> 75.88 MB`

Important rule:

- do not assume `int8/int8` is safe just because both single-component int8
  variants exist
- verify each encoder/decoder combination explicitly

## Step 6. Verify FP32 Parity

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify.json
```

Current verified result:

- encoder shape match: `true`
- encoder length match: `true`
- decoder token ids match: `true`
- final text match: `true`

## Step 7. Verify FP16 Parity

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.fp16.onnx `
  --decoder-name decoder-model.fp16.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-fp16.json
```

Current verified result:

- encoder shape match: `true`
- decoder token ids match: `true`
- final text match: `true`

## Step 8. Verify INT8 Combinations Separately

### Decoder INT8 Only

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.onnx `
  --decoder-name decoder-model.int8.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-decoder-int8.json
```

Current verified result:

- token ids match: `true`
- final text match: `true`

### Encoder INT8 Only

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.int8.onnx `
  --decoder-name decoder-model.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-encoder-int8.json
```

Current verified result:

- token ids match: `true`
- final text match: `true`

### Full INT8

```powershell
conda run -n nemo python tools/model-debugging/reference/canary-180m-flash/verify_canary_onnx.py `
  --model-dir N:\models\onnx\nemo\canary-180m-flash-smoke `
  --reference tools/data/results/canary/canary-180m-flash-reference.json `
  --encoder-name encoder-model.int8.onnx `
  --decoder-name decoder-model.int8.onnx `
  --output tools/data/results/canary/canary-180m-flash-onnx-verify-int8.json
```

Current verified result:

- encoder shape match: `true`
- decoder token ids match: `false`
- final text match: `false`

Observed text difference on the smoke fixture:

- reference:
  - `And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.`
- full int8/int8:
  - `And so my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.`

Conclusion:

- encoder-only int8 looks acceptable on the smoke fixture
- decoder-only int8 looks acceptable on the smoke fixture
- full int8/int8 should not be treated as a safe default without broader testing

## Step 9. Integrate The Runtime

Current TypeScript integration for this port lives in:

- [src/models/nemo-aed/model.ts](N:\github\asrjs\speech-recognition\src\models\nemo-aed\model.ts)
- [src/models/nemo-aed/executor.ts](N:\github\asrjs\speech-recognition\src\models\nemo-aed\executor.ts)
- [src/models/nemo-aed/tokenizer.ts](N:\github\asrjs\speech-recognition\src\models\nemo-aed\tokenizer.ts)
- [src/presets/canary/factory.ts](N:\github\asrjs\speech-recognition\src\presets\canary\factory.ts)

Rules that mattered for Canary:

- model family should be `nemo-aed`, not `whisper-seq2seq`
- prompt building is part of runtime correctness, not just UI metadata
- tokenizer metadata must preserve special-token ids and language-code ids
- decoder verification must compare token ids, not only final text

## Step 10. Preprocessor Strategy

### Current Status

Current repo behavior:

- the ONNX preprocessor is optional and not required to prove encoder/decoder
  parity
- the NeMo preprocessor export is currently blocked by PyTorch ONNX STFT complex
  type limitations in this environment

### Recommended Direction For Future Work

Do not assume we need a NeMo ONNX preprocessor.

For this workspace, a stronger long-term path is likely:

1. use a JS frontend or feature extractor when it can match NeMo reference
2. keep encoder/decoder artifacts separate from frontend artifacts
3. use the PyTorch reference JSON to validate frontend parity directly
4. only keep or ship a NeMo ONNX frontend when it clearly beats the JS path

Existing external references worth reusing:

- `N:\github\ysdede\parakeet.js`
- `N:\github\ysdede\meljs`

### Validation Workflow For A JS Frontend

If we port or adapt a JS frontend for Canary:

1. generate the NeMo reference JSON first
2. run the same waveform through a JS frontend parity helper
3. compare:
   - feature tensor shape
   - valid feature length
   - max absolute difference
   - mean absolute difference
   - RMSE
4. only then switch runtime defaults from ONNX/shared preprocessor to JS
5. keep at least one CI-safe helper test around prompt/config/tokenizer behavior

Current helper script:

- [node-canary-js-frontend-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-canary-js-frontend-parity.mjs)

Example commands:

```powershell
node tools/model-debugging/scripts/node-canary-js-frontend-parity.mjs `
  --frontend asrjs `
  --output tools/data/results/canary/canary-180m-flash-js-frontend-parity-asrjs.json
```

```powershell
node tools/model-debugging/scripts/node-canary-js-frontend-parity.mjs `
  --frontend meljs `
  --output tools/data/results/canary/canary-180m-flash-js-frontend-parity-meljs.json
```

```powershell
node tools/model-debugging/scripts/node-canary-js-frontend-parity.mjs `
  --frontend parakeet.js `
  --output tools/data/results/canary/canary-180m-flash-js-frontend-parity-parakeetjs.json
```

Use thresholds only when a frontend is already close enough that a failing exit
code is useful:

```powershell
node tools/model-debugging/scripts/node-canary-js-frontend-parity.mjs `
  --frontend meljs `
  --max-abs-threshold 0.001 `
  --mean-abs-threshold 0.0001 `
  --rmse-threshold 0.0002 `
  --length-tolerance 0
```

What the helper currently tells us:

- it can prove whether a JS frontend is close enough to the saved NeMo
  reference to be considered for runtime use
- it can also prove when a frontend is not yet Canary-compatible, which is
  still valuable because it keeps us from shipping an unverified path

Current smoke-fixture results saved in `tools/data/results/canary/`:

- `canary-180m-flash-js-frontend-parity-asrjs.json`
  - `frontend.validLengthMode`: `centered`
  - `frontendValidLength`: `1101`
  - `referenceValidLength`: `1101`
  - `maxAbs`: about `2.1461`
  - `meanAbs`: about `0.0008919`
- `canary-180m-flash-js-frontend-parity-meljs.json`
  - `frontendValidLength`: `1100`
  - `referenceValidLength`: `1101`
  - `maxAbs`: about `0.7285`
  - `meanAbs`: about `0.000919`
- `canary-180m-flash-js-frontend-parity-parakeetjs.json`
  - currently identical to the `meljs` run on this fixture

Interpretation:

- `asrjs` uses the centered-frame contract for Canary and is the current runtime
  default in this repo because it preserves the saved smoke-fixture transcript
  text better than the shared ONNX frontend path
- `meljs` and `parakeet.js` remain useful external references, but on this
  fixture they still behave like the ONNX-compatible `1100`-valid-frame path
- the remaining Canary-specific work is tail-frame numerical drift reduction,
  not choosing between JS and ONNX as the default frontend

### Decision Rule

Prefer a JS frontend when all of the following are true:

- feature parity is close enough to preserve final token/text parity
- it removes a heavyweight ONNX artifact
- it reduces browser/runtime overhead
- it is maintainable inside `@asrjs/speech-recognition`

## Step 11. Final Repository Checks

Before closing a port:

1. run `npm run typecheck`
2. run `npm run build`
3. run `npm test`
4. update:
   - `tools/model-debugging/README.md`
   - `tools/model-debugging/playbooks/README.md`
   - `tools/model-debugging/reference/README.md`
   - model-specific reference README

## Current Artifact Matrix

Current smoke export directory:

- `N:\models\onnx\nemo\canary-180m-flash-smoke`

Current files:

- `encoder-model.onnx`
- `decoder-model.onnx`
- `encoder-model.fp16.onnx`
- `decoder-model.fp16.onnx`
- `encoder-model.int8.onnx`
- `decoder-model.int8.onnx`
- `tokenizer.json`
- `config.json`

Not part of the recommended Canary flow:

- `nemo128.onnx`

Why:

- the current runtime can rely on a shared artifact if needed
- a JS frontend may be the better long-term solution
- the PyTorch STFT export issue still makes dedicated preprocessor export noisy

Not currently produced in this environment when preprocessor export is tried:

- `nemo128.onnx`

## Lessons To Carry Forward

- capture prompt ids early for multitask AED models
- treat tokenizer export as part of runtime API design
- verify quantization combinations separately, not only “all fp16” or “all int8”
- do not let a frontend export issue block encoder/decoder porting progress
- if we already have strong JS mel code, compare it to reference before adding
  another ONNX frontend artifact
