# Browser validation

These probes exercise real browser integration surfaces without adding browser
automation to the published package. They are run from the library checkout and
use a sibling demo as the application under test.

## Streaming microphone smoke

Run the capture/controller/HUD path without loading a model:

```powershell
python C:\Users\steam\.codex\skills\webapp-testing\scripts\with_server.py `
  --server "npm --prefix N:\github\asrjs\streaming-demo run dev -- --host 127.0.0.1" `
  --port 3000 `
  -- python tools/browser-validation/streaming-demo-mic-smoke.py `
  --audio tests/fixtures/ItsLifeJim.en.wav
```

For a real local Parakeet browser inference path, add the model directory and
select the model explicitly:

```powershell
python C:\Users\steam\.codex\skills\webapp-testing\scripts\with_server.py `
  --server "npm --prefix N:\github\asrjs\streaming-demo run dev -- --host 127.0.0.1" `
  --port 3000 `
  -- python tools/browser-validation/streaming-demo-mic-smoke.py `
  --audio tests/fixtures/ItsLifeJim.en.wav `
  --model-id parakeet-realtime-eou-120m-v1 `
  --model-dir N:\models\onnx\nemo\parakeet-realtime-eou-120m-v1-onnx `
  --backend wasm `
  --mode speech-detect
```

The optional `--output` and `--screenshot` flags save structured evidence.
Fake-device results verify the browser capture and inference path; they do not
replace a physical microphone check or establish broad model quality.
