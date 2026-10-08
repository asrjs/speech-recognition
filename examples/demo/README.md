# asr.js Whisper 4-Graph Browser Demo

Runs the Whisper large-v3-turbo 4-graph splitgraph path in a browser through the
library pipeline. The demo uses the core package for model loading, mel
processing, decode policy, KV cache handling, and transcript mapping.

```bash
cd examples/demo
npm run dev
```

Open the printed local URL in a real Chrome/Edge window with WebGPU enabled for
WebGPU profiles. WASM profiles require cross-origin isolation for threaded WASM;
the Vite config sets the required headers.

Default model repo:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph
```

The q8 WASM preset is the safest browser fallback. The fp16io encoder + fp32
decoder preset is the current WebGPU target.

Canary 180M Flash smoke page:

```text
https://localhost:8765/canary-smoke.html
```

It uses the approved local smoke bundle under `N:/models` and the fixed
`jfk-short.wav` fixture. The Vite filesystem allow-list is intentionally
limited to this repository and `N:/models`.

Orukeet local-folder smoke page:

```text
https://localhost:8765/orukeet-smoke.html
```

Pick an Orukeet ONNX folder, a 16 kHz mono WAV, a runtime backend and an
encoder precision. It loads through the public browser API, transcribes the
fixture repeatedly, awaits disposal, and prints the receipt as JSON. Real local
inference only; it is not numerical parity against the original NeMo engine.

## Automatic Chrome model test and JSON capture

The sibling `webgpu-agent-test` contains the historical `?auto=1` Chrome
runner and `/__test_result__` JSON capture system. Stock Parakeet v3 was
rechecked successfully on 2026-10-08. The tracked Orukeet demo now supports
the same automatic URL workflow, using its existing public local-folder API.
The runner selects local files, waits for two transcriptions and awaited
disposal, compares token IDs and text with the official oracle, closes its
Chrome instance, and saves a fresh PASS/FAIL JSON receipt. No model download.

From the repository root, start the demo (8767 avoids an existing sibling
server on 8765):

```powershell
node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1 --port 8767 --strictPort
```

In another terminal at the repository root:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --output tools/data/results/orukeet/auto-browser-new-run.json
```

Use the server's printed scheme (`http` if local certificates are absent).
Default runner URL uses 8765, matching `npm run dev`. Python Playwright and
installed Chrome are prerequisites; the runner never installs them. Use a
new output filename for each run; existing receipts are protected.
`--backend wasm`, `--precision fp16`, and `--repeat 1` are explicit options,
not automatic fallback policies.

Verified receipt: `tools/data/results/orukeet/auto-browser-jfk-2026-10-08-03.json`:
fp32 WebGPU hybrid, two runs, 38 oracle token IDs and exact text, awaited
model disposal and browser closure. This is a short-fixture output smoke,
not feature/encoder numerical parity, long-audio, streaming, or soak acceptance.
The first receipt (`...-01.json`) remains a FAIL: text matched but the default
canonical segment detail omitted tokens; the demo now uses the supported
`canonical+native` envelope to capture native debug IDs and canonical metrics.

### Long-input, same-PCM browser control

The demo now defaults to the existing deterministic WAV decoder exposed by
`browser/media` (using the public entrypoint's source in this source-checkout
demo). It requires mono/16 kHz input and never silently resamples. The previous
AudioContext path remains an explicit `--audio-strategy audiocontext` control.
Supply `--expected-pcm-sha256` to require the official effective float32 PCM;
without it the status explicitly says `PCM_UNBOUND`, even if tokens/text match.
Served JS/TS/runtime response bytes are hashed at the fulfillment boundary,
not taken from Chrome's evictable inspector cache. Runner source is also bound.

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/results/orukeet/librivox-repeat8-16k-pcm16-2026-10-02.wav --oracle tools/data/results/orukeet/official-long-repeat8-summary-2026-10-02.json --expected-pcm-sha256 caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46 --output tools/data/results/orukeet/auto-browser-long-new-run.json
```

Verified two-repeat result: `auto-browser-long-repeat8-2026-10-08-06.json`.
All 722 tokens and text match, PCM identity matches, component backend labels
match, code capture and awaited cleanup pass. See
[the scoped report](../../docs/reports/orukeet-auto-browser-long-2026-10-08.md).
This does not pass feature numerical, streaming, fp16, held-out quality or soak gates.

### Cooperative decode cancellation and reuse

Add `--verify-cancellation` to the long-input command and choose a fresh output
filename. Keep `--repeat 2` or greater. The page uses `cancel=1` to request
AbortController cancellation only after a nonzero, incomplete decode-progress
event. It requires `PipelineAbortedError` at the decode stage without a complete
event, then transcribes again on the same model and PCM before awaited disposal.
The two normal results (one before and one after cancellation) must still match
every original-engine token ID and full text; a successful abort alone cannot pass.
Without the flag this optional lifecycle control is not exercised.

Cancellation is cooperative at decoder boundaries, not hard interruption of an
in-flight ORT/GPU call. This source-checkout example is separate from the classic
sibling browser-demo's user cancellation/unload UI and from GPU reclamation/soak.
