# Browser validation

These probes exercise real browser integration surfaces without adding browser
automation to the published package. They run from the library checkout;
application probes use a sibling demo, while owner-level probes use the built
public API directly.

## Automatic Chrome URL, model load and saved ASR result

The existing runner is `tests/smoke/orukeet-auto-browser.py`, not a new
inference implementation. See [the demo guide](../../examples/demo/README.md#automatic-chrome-model-test-and-json-capture)
for server startup, long-input, cancellation and controller commands.
It opens installed Chrome headlessly at an `auto=1` URL, selects local ONNX
files and audio, transcribes twice by default, compares tokens/text against
the supplied original-engine oracle, awaits disposal, closes its own Chrome,
and exclusively creates a PASS/FAIL JSON receipt. It never downloads weights
or installs Chrome. This adapter is Orukeet-specific, not a universal model
runner; the historical generic app flow lives in sibling `webgpu-agent-test`.

From the repository root, with the existing demo running on 8767:

```powershell
python tests/smoke/orukeet-auto-browser.py --help
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --verify-controller --output tools/data/results/orukeet/auto-browser-fresh-run.json
```

Choose a fresh output name every time. If 8767 is already occupied, verify its
page before reusing it; do not stop another agent's server. A missing server is
an environment prerequisite, not evidence of a decoder bug. A smoke PASS is
same-PCM token/text and requested lifecycle evidence, not full numerical parity,
fp16 promotion, native streaming or broad quality acceptance.

## Packaged TEN-VAD model worker lifecycle

The existing offline runner also accepts `--scenario ten-vad`. Build/copy the
runtime assets first, then choose a fresh output filename:

```powershell
node node_modules/typescript/bin/tsc -p tsconfig.json
node scripts/copy-runtime-assets.mjs
python tools/browser-validation/capture-lifecycle-smoke.py --scenario ten-vad --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --audio tests/fixtures/jfk2.en.wav --output tools/data/results/harness/ten-vad-FRESH.json
```

This calls the public adapter with its real packaged WASM worker, compares exact
probability/flag bytes across reset/uneven chunks and reset-versus-fresh silence,
and requires two actual disposal acknowledgements and worker termination.
No downloads or sibling server. In-browser probe failures save FAIL JSON and
exit nonzero. It does not certify VAD accuracy, original-engine parity, physical
capture, speech-detect controller integration or memory reclamation/soak.
See [the failure, fix and final receipt](../../docs/reports/ten-vad-worker-lifecycle-2026-10-08.md).

## Speech-detection controller: file PCM and real TEN diagnostics

After the same build/copy steps above, run:

```powershell
python tools/browser-validation/capture-lifecycle-smoke.py --scenario ten-vad-controller --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --audio tests/fixtures/jfk2.en.wav --output tools/data/results/harness/ten-controller-FRESH.json
```

The public controller receives identified file PCM through its supported capture
input boundary; the default starter, real model worker, probability buffer and
utterance publication remain unchanged. It requires two identical speech runs,
exact model-to-buffer values and segment-to-utterance PCM, silent restart,
stale-callback rejection, actual worker disposal and rejected post-dispose start.
TEN remains diagnostics-only; the rough gate determines boundaries. This is
not microphone capture, ASR, original-engine parity or VAD accuracy acceptance.
The required fixture must contain speech; an all-silence input correctly FAILs.
See [receipt, scope and next gate](../../docs/reports/ten-vad-controller-pcm-2026-10-08.md).

## Real capture through delayed TEN model readiness

After the build/copy steps above, use the existing runner with a fresh receipt:

```powershell
python tools/browser-validation/capture-lifecycle-smoke.py --scenario ten-vad-capture --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --audio tests/fixtures/jfk2.en.wav --output tools/data/results/harness/ten-capture-FRESH.json
```

This uses real getUserMedia, AudioWorklet, public capture/controller and packaged
TEN worker, with a fake WAV device. Only delivery of the real worker INIT message
is delayed by 250 ms. Two starts require complete captured-frame/model-hop
coverage, contiguous offsets, exact captured-PCM slices in rough-gate utterances,
closed contexts/tracks and actual worker disposal. The shared controller retains
owned startup PCM until the starter is ready; exceeding the configured ring
capacity fails rather than silently losing audio. TEN is still diagnostics-only.
This does not establish ASR, physical microphone, WebGPU or original-engine parity.
See [failure, fix and evidence](../../docs/reports/capture-model-startup-2026-10-08.md).

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

## Offline manual microphone controller lifecycle

Use the same fresh build and installed browser prerequisites above:

```powershell
python tools/browser-validation/capture-lifecycle-smoke.py `
  --scenario microphone-controller `
  --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" `
  --audio tests/fixtures/ItsLifeJim.en.wav `
  --output tools/data/results/harness/manual-controller-new-run.json
```

The public controller uses its real default starter and monitor in explicit
manual mode, with real capture resources. The probe verifies exact captured-PCM
delivery on stop/flush, restart, active disposal without an extra utterance,
stable disposed state and rejection of restart after disposal. A controlled
Promise gate delays delivery of an already-live real capture handle and verifies
disposal closes its context and tracks. This is not a permission-dialog race,
speech-detect/VAD, ASR, WebGPU or cleanup-failure test. Capture-owner controls
remain separately runnable using `--scenario capture-owner` (the default).

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
