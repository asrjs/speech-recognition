# GigaAM v3 E2E RNN-T official reference chain

Official source: [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)
`v3_e2e_rnnt` (MD5 `2730de7545ac43ad256485a462b0a27a`). Russian punctuation RNN-T.
Do not start from a third-party ONNX conversion.

- Checkpoint: `N:\models\gigaam\official-cache\v3_e2e_rnnt.ckpt`
- Tokenizer: `N:\models\gigaam\official-cache\v3_e2e_rnnt_tokenizer.model`
- ONNX: `N:\models\onnx\gigaam\v3-e2e-rnnt`
- Oracle audio: official `example.wav` (Pushkin), not JFK

## Ladder

```powershell
$env:PYTHONPATH = 'N:\github\salute-developers\GigaAM'
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'

& $PY tools/model-debugging/reference/gigaam-v3-e2e-rnnt/capture_gigaam_rnnt_reference.py `
  --audio N:\models\gigaam\official-cache\example.wav `
  --download-root N:\models\gigaam\official-cache `
  --tensor-dir N:\models\gigaam\v3-e2e-rnnt\captures `
  --output tools/data/results/gigaam/v3-e2e-rnnt-example-reference.json

& $PY tools/model-debugging/reference/gigaam-v3-e2e-rnnt/export_gigaam_rnnt_onnx.py `
  --download-root N:\models\gigaam\official-cache `
  --output-dir N:\models\onnx\gigaam\v3-e2e-rnnt `
  --dtype float32

& $PY tools/model-debugging/reference/gigaam-v3-e2e-rnnt/compare_gigaam_rnnt_onnx.py `
  --reference tools/data/results/gigaam/v3-e2e-rnnt-example-reference.json `
  --onnx-dir N:\models\onnx\gigaam\v3-e2e-rnnt `
  --output tools/data/results/gigaam/v3-e2e-rnnt-example-onnx-cpu.json

$env:GIGAAM_RNNT_ONNX_SMOKE = '1'
$env:NODE_OPTIONS = '--max-old-space-size=16384'
npx vitest run tests/gigaam-rnnt-onnx-backends.test.ts
```

Chrome WebGPU (Vite `:8765` in `webgpu-agent-test`):

```powershell
node scripts/run-gigaam-rnnt-webgpu.mjs
```

Weights stay outside Git. Experimental, no preset.
