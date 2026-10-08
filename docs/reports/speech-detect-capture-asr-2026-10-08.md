# C92 — Actual automatic speech-detect capture to ASR

## Outcome and boundaries

Bounded real-browser consumer workflow PASS: actual fake WAV microphone,
getUserMedia/AudioWorklet, public microphone controller, automatic rough-gate
utterances, real TEN diagnostics, local Orukeet fp32 WebGPU encoder/WASM
decoder/joint, same-input public API replay and awaited cleanup.

This extends C90/C91's existing helper and runner, not a second decoder. Manual
mode remains default. The model-neutral helper owns capture, segment identity,
serialized caller transcription and lifecycle evidence. The Orukeet adapter
owns local model loading and native token extraction. No core/family math,
threshold, quantization policy or canonical transcript contract changed.

TEN remains diagnostics-only; rough energy determines segment boundaries.
This is utterance-based offline ASR, not native streaming, TEN-driven VAD
accuracy, physical microphone or held-out quality acceptance. The source demo
supplies a real TS worker URL through the supported worker factory boundary;
packaged default-worker acceptance remains separate. Capture processing changes
PCM and restart segment/text details: identical original fake WAVs do not imply
identical capture inputs. Do not require equal segmentation across those inputs.

## Acceptance and implementation

- Two actual capture sessions, each at least 10 seconds with a natural `silence`
  utterance published before explicit stop. Preserve additional `max-duration`
  and stop-flushed utterances; do not assume nonoverlapping windows.
- Contiguous finite 16 kHz capture; actual extracted segment bounds, segment
  PCM and published utterance PCM equal their exact captured slice, including
  signed zero. Logical metadata bounds are not used as extracted bounds.
- Nonempty actual ASR per accepted utterance; separately transcribe each exact
  PCM window through the public API and compare complete token IDs/text. Queue
  the caller's ASR while capture continues; do not issue concurrent model calls.
- Complete real TEN result-hop coverage and offsets, valid probabilities/flags,
  identical public probability-buffer values, ready state and rough-only gate.
  Microphone stop does not drain TEN automatically; explicitly await coverage
  before restart. Disposal must acknowledge and terminate the actual worker.
- Await pending ASR, capture contexts/tracks, controller and model disposal;
  repeated disposal publishes no extra utterance and post-dispose start rejects.
- `--save-capture-pcm` preserves two full-session f32le sidecars, without PCM16
  conversion/resampling. Before any write, Python recomputes each captured
  window's three recorded hashes from full-session bytes and frame bounds.
  Receipt-only TEN probability hashes are not an independent native VAD oracle.
- Required optional mode with missing/corrupt evidence fails; unrequested mode
  skips. Original full-file PCM/token/text and delivered-code checks remain.

## Reproduction

Use the existing 8767 demo after verifying its page. Do not stop another agent's
server. If absent, start from the repository root:

```powershell
node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1 --port 8767 --strictPort
```

Run from the root in a separate shell with a never-used output filename:

```powershell
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --verify-capture --capture-mode speech-detect --save-capture-pcm --output tools/data/results/orukeet/c92-speech-FRESH.json
```

Installed Chrome and Python Playwright are reused; no downloads/installs. To
run the manual/controller regression, omit `--capture-mode speech-detect` and
add `--verify-controller`, with another fresh filename. The independent base
manifest command is in `tools/browser-validation/README.md`.

## Verified evidence

Receipts under `tools/data/results/orukeet/`:

- `auto-browser-c92-speech-capture-2026-10-08-01.json`: initial new-mode PASS,
  preserved before adding independent Python slice-hash validation.
- `auto-browser-c92-speech-capture-2026-10-08-02.json`: final runner PASS,
  SHA-256 `0fa5f6f1c45927480e6c4d8efee163054514aeacf8b77d81b59869b564589547`.
  Chrome 154.0.8037.98; both sessions capture 160,000 samples / 625 chunks,
  real TEN has 625/625 hops and 433/432 positive flags. Session 0 yields three
  natural-pause utterances and one stop flush (11/4/12/13 tokens); session 1
  yields two natural pauses and one max-duration utterance (10/4/20 tokens).
  All seven outputs exactly match their own same-PCM public replays. Both
  base-file runs match 38 original-engine tokens and full text independently.
  There are no page errors, code-capture errors or blocked requests.
- Final `.json.capture-0.f32le`: 640,000 bytes,
  SHA-256 `d6128523c1405d62b2c2c90f75f8bd25afaea29830a71724017302fc8ec33639`.
- Final `.json.capture-1.f32le`: 640,000 bytes,
  SHA-256 `afda7f61aaab28722bb90533156d0c56031e47c96bb2a5c3dfd42124a849faaa`.
- `auto-browser-c92-manual-regression-2026-10-08-01.json`: final runner's
  original manual capture/controller/timing/file-oracle checks PASS.
- `manifest-chrome-c92-regression-2026-10-08-01.json` and its browser child:
  independent base-file PASS, all 19 checks including unchanged C83 delivered
  code, runner/artifact/oracle identity and actual browser/model cleanup.

Runner SHA-256: `45101052f1ffcc741aed0618931450168fe6eadd34f4e2dac9bc8a16e512e219`.
Only the manifest's two runner digests were updated; C83 delivered-code pin
and original weights/artifacts/oracles remain unchanged. 31 runner, 7 adapter
and 24 manifest lightweight Python tests PASS (62 methods plus subcases).
Node syntax/Python compile/diff checks PASS. Read-only implementation review
found no verified defect; actual browser runs provide separate acceptance.

## Next unresolved gate

Run the existing original NeMo inspector on each identified utterance's exact
float32 capture slice and compare full tokens/text before claiming original
engine parity on these automatic boundaries. Preserve full-session and slice
identities; do not substitute the fake-device WAV or full-file oracle for a
window oracle. Per-window stage numerics, timestamp accuracy, production feature
conformance, fp16 promotion, quality, memory/soak and packaged-consumer speech
capture remain unresolved. C91's original-engine manual-input results do not
automatically certify the new automatic windows. The whole-product goal stays
active; this bounded workflow does not complete it.
