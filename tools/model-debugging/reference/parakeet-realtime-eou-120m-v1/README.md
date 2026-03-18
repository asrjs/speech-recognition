# Parakeet Realtime EOU 120M v1 Porting Reference

Reference tooling for porting `nvidia/parakeet_realtime_eou_120m-v1` into
`@asrjs/speech-recognition`.

These scripts are intended to be run from a Python environment with NeMo ASR,
ONNX, and ONNX Runtime installed. In this workspace, the `nemo` conda
environment is the expected baseline.

## Heavy Artifact Policy

The large saved reference JSON for this RNNT port may be committed as
`parakeet-realtime-eou-120m-v1-reference.json.gz` instead of raw JSON to keep
pull request diffs and review context smaller.

Check artifact status:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs status --target parakeet
```

Restore the raw JSON locally when a script or manual inspection needs it:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs unpack --target parakeet
```

Pack it back into `.json.gz` and delete the raw extracted file:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs pack --target parakeet --delete-originals
```

## Scripts

- `generate_parakeet_realtime_reference.py`
  - runs the original NeMo model on a fixture audio file
  - captures waveform, preprocessor features, encoder outputs, manual greedy
    RNNT token ids, and final text
  - records both the raw text that includes `<EOU>` and the user-visible text
    with control tokens removed
- `export_parakeet_realtime_onnx.py`
  - exports the FastConformer encoder and fused decoder/joint wrapper used by
    the JS runtime
  - also writes `vocab.txt` and `config.json`
  - attempts preprocessor export, but the current PyTorch ONNX exporter still
    rejects the NeMo STFT path in this environment
- `convert_onnx_fp16.py`
  - converts exported FP32 ONNX models to FP16 while keeping IO dtypes stable
- `convert_onnx_int8.py`
  - produces dynamic-quantized INT8 encoder and decoder variants
- `verify_parakeet_realtime_onnx.py`
  - replays the exported ONNX artifacts against a saved PyTorch reference JSON
    and reports encoder/token/text parity

## Recommended Flow

### 1. Generate the NeMo reference JSON

```powershell
& 'C:\Users\steam\anaconda3\envs\nemo\python.exe' tools/model-debugging/reference/parakeet-realtime-eou-120m-v1/generate_parakeet_realtime_reference.py `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --output tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-reference.json
```

If you want to keep the repo lightweight after regenerating it, pack it back
into `.json.gz`:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs pack --target parakeet --delete-originals
```

### 2. Export the ONNX runtime artifacts

```powershell
& 'C:\Users\steam\anaconda3\envs\nemo\python.exe' tools/model-debugging/reference/parakeet-realtime-eou-120m-v1/export_parakeet_realtime_onnx.py `
  --output-dir N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx `
  --device cpu
```

Expected outputs:

- `encoder-model.onnx`
- `decoder_joint-model.onnx`
- `vocab.txt`
- `config.json`

### 3. Produce FP16 variants

```powershell
& 'C:\Users\steam\anaconda3\envs\nemo\python.exe' tools/model-debugging/reference/parakeet-realtime-eou-120m-v1/convert_onnx_fp16.py `
  --model-dir N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx
```

### 4. Produce INT8 variants

```powershell
& 'C:\Users\steam\anaconda3\envs\nemo\python.exe' tools/model-debugging/reference/parakeet-realtime-eou-120m-v1/convert_onnx_int8.py `
  --model-dir N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx
```

### 5. Verify ONNX parity

```powershell
& 'C:\Users\steam\anaconda3\envs\nemo\python.exe' tools/model-debugging/reference/parakeet-realtime-eou-120m-v1/verify_parakeet_realtime_onnx.py `
  --model-dir N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx `
  --reference tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-reference.json `
  --output tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-onnx-verify.json
```

### 6. Compare the Node/WASM runtime against the saved reference

```powershell
node tools/model-debugging/scripts/node-asrjs-parakeet-realtime-parity.mjs `
  --model-dir N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx `
  --reference tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-reference.json `
  --output tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-asrjs-parity.json
```

## Notes

- The NeMo tokenizer vocab is a plain one-token-per-line `vocab.txt`, not a
  `token id` mapping file.
- `<EOU>` should be preserved in raw/native parity outputs but removed from the
  user-visible transcript text.
- The current working frontend for this model is the shared JS NeMo mel path
  with centered valid length and raw log-mel output.
- Do not copy a generic normalized `nemo128.onnx` artifact from a TDT model
  into this model directory. It does not match the saved native reference.
- More generally, do not plan on exporting `nemo80.onnx` or `nemo128.onnx`
  preprocessors for new NeMo ports here. Extend the shared pure-JS frontend
  instead.
- If you want to validate the JS frontend directly, run:

```powershell
node tools/model-debugging/scripts/node-canary-js-frontend-parity.mjs `
  --frontend asrjs `
  --reference tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-reference.json `
  --output tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-js-frontend-parity-asrjs.json
```
