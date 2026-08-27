# GigaAM Multilingual CTC official reference chain

This folder captures and exports **official** GigaAM multilingual CTC, not a
third-party ONNX conversion.

- Official source: [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)
- Weights: `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/multilingual_ctc.ckpt`
- Official MD5: `5379d887c53ccd9cb95981e2a1832720`
- Hugging Face mirror: `ai-sage/GigaAM-Multilingual` revision `ctc`
- License: MIT
- Local clone used here: `N:\github\salute-developers\GigaAM`
- Local checkpoint: `N:\models\gigaam\official-cache\multilingual_ctc.ckpt`
- Local ONNX export: `N:\models\onnx\gigaam\multilingual-ctc`

The library family `src/models/gigaam-ctc` stays experimental until native
ORT, WASM, and WebGPU parity against this official chain pass. Do not treat a
mocked graph test as WebGPU support.

## 1. Official PyTorch reference

```powershell
$env:PYTHONPATH = 'N:\github\salute-developers\GigaAM'
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'
& $PY tools/model-debugging/reference/gigaam-multilingual-ctc/capture_gigaam_reference.py `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --download-root N:\models\gigaam\official-cache `
  --tensor-dir N:\models\gigaam\multilingual-ctc\captures `
  --output tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json `
  --device cpu
```

## 2. Official ONNX export

```powershell
$env:PYTHONPATH = 'N:\github\salute-developers\GigaAM'
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'
& $PY tools/model-debugging/reference/gigaam-multilingual-ctc/export_gigaam_onnx.py `
  --download-root N:\models\gigaam\official-cache `
  --output-dir N:\models\onnx\gigaam\multilingual-ctc `
  --dtype float32
```

## 3. Native ORT vs PyTorch

```powershell
$env:PYTHONPATH = 'N:\github\salute-developers\GigaAM'
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'
& $PY tools/model-debugging/reference/gigaam-multilingual-ctc/compare_gigaam_onnx.py `
  --reference tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json `
  --onnx-dir N:\models\onnx\gigaam\multilingual-ctc `
  --output tools/data/results/gigaam/multilingual-ctc-jfk-short-onnx-cpu.json
```

## 4. JS frontend vs official features

```powershell
node tools/model-debugging/reference/gigaam-multilingual-ctc/compare_gigaam_js_frontend.mjs `
  --reference tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json `
  --output tools/data/results/gigaam/multilingual-ctc-jfk-short-js-frontend.json
```

Weights and ONNX files stay outside Git.

## 5. JS features → native ORT text

```powershell
npx vitest run tests/gigaam-ctc-frontend-diagnose.test.ts
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'
& $PY tools/model-debugging/reference/gigaam-multilingual-ctc/compare_gigaam_js_onnx.py `
  --reference tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json `
  --js-features N:\models\gigaam\multilingual-ctc\captures\jfk-short.js-features.npy `
  --onnx-dir N:\models\onnx\gigaam\multilingual-ctc `
  --output tools/data/results/gigaam/multilingual-ctc-jfk-short-js-onnx-cpu.json
```

## 6. WASM (onnxruntime-web) then WebGPU

```powershell
$env:GIGAAM_CTC_ONNX_SMOKE = '1'
$env:NODE_OPTIONS = '--max-old-space-size=8192'
npx vitest run tests/gigaam-ctc-onnx-backends.test.ts
```

Chrome WebGPU (NVIDIA Blackwell, official fp16) matched official jfk-short
text: load 6.44s, transcribe 2.88s. Node remains `WEBGPU_NO_ADAPTER`.

```powershell
cd N:\github\asrjs\webgpu-agent-test
npm run dev
node scripts/run-gigaam-webgpu.mjs
```

The family stays experimental until a preset is published. No weights in Git.
