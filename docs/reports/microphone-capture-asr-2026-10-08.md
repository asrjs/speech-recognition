# C90 — Actual microphone capture through public controller to Orukeet ASR

Status: bounded real-browser integration PASS. Whole-product/model acceptance
remains incomplete. No core runtime or model math changed in this slice.

## Implementation and boundaries

The existing `tests/smoke/orukeet-auto-browser.py` gains opt-in
`--verify-capture`; its base URL/file-oracle flow is unchanged. After ordinary
oracle transcription and disposal, an asynchronous browser job loads another
local model and awaits two microphone sessions and cleanup. The Python timeout
bounds the job; missing capture/model disposal or inference fails the receipt.

`examples/demo/src/capture-controller.js` is a model-neutral helper using public
capture/controller APIs and a caller-provided transcription function. It owns
PCM copies, contiguous coverage, utterance delivery, replay consistency and
lifecycle evidence. The Orukeet-owned adapter supplies local model loading and
canonical/native envelope extraction. Family decoding remains in the library.
The synchronous onUtterance callback explicitly tracks and awaits asynchronous
ASR work; controller stop alone is not assumed to await transcription.

Chrome uses a fake WAV microphone, but actual getUserMedia, AudioContext,
AudioWorklet and the public manual controller run. Two starts on the same
loaded model each require at least 125 contiguous 256-frame chunks at 16 kHz,
finite/nonzero PCM, one stop-flushed utterance exactly equal to captured PCM,
nonempty tokens/text, exact same-PCM public API replay and expected component
backends. Closed contexts, ended tracks, one worklet load/owned stop per session,
no post-stop chunks, controller disposal and rejected restart are checked.
Model cleanup errors fail independently. The existing served-code capture also
binds the newly delivered helper/controller modules.

## Real evidence

Receipts in `tools/data/results/orukeet/`:

- `auto-browser-c90-capture-2026-10-08-01.json`: PASS; two base 38-token/text
  official-oracle outputs plus two actual capture/ASR sessions.
- `manifest-chrome-c90-regression-2026-10-08-01.json` and its `.browser.json`:
  independent base manifest regression PASS, all 19 checks.

Each capture session delivered 32,000 samples / 125 chunks and emitted these
11 tokens: `[1976,547,7877,1103,309,530,596,3213,404,667,7883]`.
Text: “And so, my fellow Americans.” Same captured-PCM replay matches exactly.
All primary/replay metrics report WebGPU encoder, WASM decoder/joint and JS
feature extraction. Context/track/controller/model/browser cleanup passed;
no page exceptions or runner-rejected network requests were recorded. The
console still records a 404 and Vite development WebSocket failures blocked by
Chrome's local-network checks; this is not a console-error-free run.

Captured float32 PCM SHA-256 (little-endian):

- Session 0: `20bd5c13fb224af086604e7af59ab1502b4145373eb813afa77603a631fd76d7`
- Session 1: `4f7d888a2b6eb15aedfdad61a80de3222201bfa7fa7ca207ad97506a111f6ffd`

These are observations, not future capture goldens. The capture receipt binds
299 unique served paths / 306 deliveries; all 281 C83 pinned paths remain
byte-identical. The base delivered-code contract was not regenerated. Only the
two runnerSha256 fields in `tools/model-debugging/ports/oruk-orukeet.port.json`
changed to the reviewed runner SHA-256:
`cbf786e95275ac0278858a9f455f5760487722a4e401825cd9e06bab0baf6707`.
Original weights, graphs, oracle, tolerances and fixture identities are unchanged.

## Repeat

From the repository root, start the existing demo if needed (do not stop an
unowned server; verify its page before reuse):

```powershell
node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1 --port 8767 --strictPort
```

Use the printed scheme; this checkout has local HTTPS certificates. Installed
Chrome and Python Playwright are prerequisites, not automatically installed.
Choose a new receipt filename each time:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --verify-capture --output tools/data/results/orukeet/auto-browser-capture-FRESH.json
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-chrome-FRESH.json
```

## Validation and remaining gates

21 auto-browser unittest methods, 6 browser-adapter methods and 24 manifest
methods PASS (51 total, not counting subcases). New negative controls reject
missing/partial capture, malformed PCM/token/replay/backend evidence and absent
cleanup. Node syntax, Python compilation, example TypeScript check and scoped
diff checks PASS. Independent read-only review found no verified issues.
The full core suite was not rerun: no core source changed in C90.

This is manual capture integration and library replay consistency, **not** an
independent official-engine oracle on captured PCM. Raw captured audio was not
saved; only hashes/metadata were retained. Browser microphone processing differs
from file decoding. Speech-detect/VAD utterance-to-ASR, physical microphone,
native streaming, cleanup-failure injection and memory reclamation/soak remain
open. Timing is observed, not a matched performance comparison or speedup.
Orukeet's strict feature numerical gate remains failed; full feature/encoder
numerics, held-out quality and fp16 promotion are not certified by this smoke.
Next bounded integration gate: speech-detect utterances to actual ASR, or save
exact capture PCM and compare that input with the original engine.
