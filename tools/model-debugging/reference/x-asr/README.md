# X-ASR-zh-en official reference tooling

Official ladder only. The oracle is sherpa-onnx on the project-published
Zipformer2 transducer graphs, not an arbitrary ONNX.

- Git: https://github.com/Gilgamesh-J/X-ASR (Apache-2.0)
- Weights: HuggingFace `GilgameshWind/X-ASR-zh-en`
- Streaming: true stateful Zipformer2 caches (`chunk-160ms` bounded candidate)

```powershell
$PY = 'N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe'

& $PY tools/model-debugging/reference/x-asr/capture_xasr_sherpa.py `
  --model-dir N:\models\x-asr\zh-en\chunk-160ms-model `
  --audio tools\data\fixtures\audio\jfk-short.wav `
  --output tools\data\results\x-asr\x-asr-zh-en-160ms-jfk-short-sherpa.json `
  --chunk-ms 160 --provider cpu

& $PY tools/model-debugging/reference/x-asr/inspect_xasr_onnx.py `
  --model-dir N:\models\x-asr\zh-en\chunk-160ms-model `
  --chunk-ms 160 `
  --output tools\data\results\x-asr\x-asr-zh-en-160ms-graph-io.json
```

The family stays experimental. No weights in Git.
