# Browser validation

These probes exercise real browser integration surfaces without adding browser
automation to the published package. They run from the library checkout;
application probes use a sibling demo, while owner-level probes use the built
public API directly.

## Offline capture owner lifecycle

This probe needs an already installed Python Playwright and Chromium executable;
it never installs a browser/dependency or starts a server. Refresh `dist` first:

```powershell
node node_modules/typescript/bin/tsc -p tsconfig.json
node scripts/copy-runtime-assets.mjs
python tools/browser-validation/capture-lifecycle-smoke.py --help
python tools/browser-validation/capture-lifecycle-smoke.py `
  --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" `
  --audio tests/fixtures/ItsLifeJim.en.wav `
  --output tools/data/results/harness/capture-owner-new-run.json
```

Playwright fulfills requests from the local `dist` tree plus an explicit local
`pako` ESM import-map entry; other requests are rejected. It uses an isolated
headless browser context and fake WAV device, but real `getUserMedia`,
`AudioContext`, AudioWorklet and the public `dist/browser.js` capture export.
Two independent sessions require live/nonzero finite PCM, contiguous 256-frame
16 kHz chunk timing, actual worklet loading, shared concurrent stop, closed
contexts, ended owned tracks and no further chunks during a 150 ms observation.

Receipts are exclusively created (never overwritten) and hash the audio,
probe, runner, capture source and every served module before/after execution.
The runner does not automatically establish source-to-build equivalence: run
the build commands above before relying on current-source acceptance. This is
capture-owner evidence, not controller, VAD/ASR, physical microphone, arbitrary
device-rate resampling, cleanup-failure injection, soak or WebGPU acceptance.
The browser's capture processing may alter the WAV; this is not exact PCM parity.

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
