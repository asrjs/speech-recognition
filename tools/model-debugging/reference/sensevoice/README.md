# SenseVoiceSmall official reference tooling

Official ladder only. Do not treat `OpenVoiceOS/sensevoice-small-onnx` as the
oracle. Weights stay under `N:\models\` and are not committed.

Provenance:

- Git: https://github.com/FunAudioLLM/SenseVoice
- Local clone: `N:\github\FunAudioLLM\SenseVoice` (`6991744856587fa44379e8b5dcc432debffeb1be`)
- Weights: HuggingFace `FunAudioLLM/SenseVoiceSmall` revision
  `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`
- License: FunASR Model Open Source License (`model-license`)

```powershell
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'
$env:PYTHONPATH = 'N:\github\FunAudioLLM\SenseVoice'

# 1. Official FunASR clip-level oracle (vad_model=None)
& $PY tools/model-debugging/reference/sensevoice/capture_sensevoice_reference.py `
  --model-dir N:\models\sensevoice\SenseVoiceSmall `
  --audio tools\data\fixtures\audio\jfk-short.wav `
  --output tools\data\results\sensevoice\sensevoice-small-jfk-short-reference.json `
  --language en `
  --no-use-itn `
  --hash-model-files

# 2. Official unquantized ONNX
& $PY tools/model-debugging/reference/sensevoice/export_sensevoice_onnx.py `
  --model-dir N:\models\sensevoice\SenseVoiceSmall `
  --output-dir N:\models\onnx\sensevoice\small

# 3. Native ORT vs FunASR text
& $PY tools/model-debugging/reference/sensevoice/compare_sensevoice_onnx.py `
  --reference tools\data\results\sensevoice\sensevoice-small-jfk-short-reference.json `
  --onnx-dir N:\models\onnx\sensevoice\small `
  --output tools\data\results\sensevoice\sensevoice-small-jfk-short-onnx-cpu.json `
  --language 4 `
  --textnorm 15
```

Official ONNX inputs are LFR+CMVN `speech` `[B,T,560]`, not raw 80-bin fbank.
The library executor now detects that contract and applies FunASR LFR+CMVN
from `am.mvn`. Folded OpenVoiceOS `features` `[B,T,80]` remains accepted.

```powershell
$env:SENSEVOICE_ONNX_SMOKE = '1'
$env:NODE_OPTIONS = '--max-old-space-size=8192'
npx vitest run tests/sensevoice-onnx-backends.test.ts
```

Provenance hashes: `tools/data/results/sensevoice/sensevoice-small-provenance.json`.
The family stays experimental. X-ASR and Qwen are left intact.
