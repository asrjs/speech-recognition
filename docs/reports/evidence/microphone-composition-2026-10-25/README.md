# Microphone evidence

`real-run-final/receipt.json`, `manual.json` and `ten-vad.json` are the final
real-weight receipts. `artifact-hashes.json` identifies the model and actual SDK
controller files. The source fixture SHA is checked before the runner starts.

The acceptance page is `frameworks/tmp/mic-acceptance`. It uses the built SDK,
the actual capture and starter factories, a borrowed local model, and Chrome's
recorded-audio device. These factories are wrapped only to observe tracks,
utterances and VAD state. No model, capture or VAD result is mocked. Runtime
assets are served by the established `demos/scripts/serve-suite.mjs` server.
The four page/config/server source files are also archived under `harness/`;
restore them to that framework path before reproducing, since their local import
paths are relative to that location. The temporary test servers were stopped
after acceptance; the main seven-app suite remains on port 4190.

From `frameworks`, build and serve the page:

```powershell
node node_modules/vite/bin/vite.js build --config tmp/mic-acceptance/vite.config.ts
node tmp/mic-acceptance/serve.mjs
```

From the core checkout, use a **new** output directory:

```powershell
C:/Python311/python.exe docs/reports/evidence/microphone-composition-2026-10-25/real-microphone.py --core . --model-dir N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx --url http://127.0.0.1:4342/ --asset-origin http://127.0.0.1:4342 --output tmp/microphone-new-run
```

The local server reads existing model files; no remote weights are downloaded.
Chromium receives a derived fixture with five seconds of silence and `%noloop`;
the original WAV is unchanged. Text matching removes punctuation/case only and
requires the full JFK phrase. The runner asserts ready TEN-VAD, context updates,
ready/stopped state, ended tracks, successful reuse, single graph requests and
awaited disposal. Physical microphone status remains `UNVERIFIED`.

The first three setup receipts failed before model inference. Run 1 exposed the
Python server's `.mjs` MIME; run 2 failed WASM transport; run 3 hit Vite dev's
public-module import restriction. Run 4 and the final production-artifact run
passed. The build needed ES worker output for split worker modules, matching
the existing main demo configuration. Those failures remain available.

The initial full unit run found a cleanup-error ordering regression. The source
release order was restored; that acceptance test was not weakened. The final
full run passed. Core source/example typecheck and framework typecheck include
the public capabilities; the original base controller fixture remains valid.
