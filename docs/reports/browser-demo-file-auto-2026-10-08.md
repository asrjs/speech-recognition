# C55 / C14 — Classic browser-demo automatic file transcription

The recovered Chrome harness now exercises the actual sibling application's
file-upload workflow, separately from its local-model warm-up. Default UI-only
and local warm-up modes remain unchanged. No sibling source, model math,
weights, original-engine oracle or acceptance tolerance was edited.

## Verified positive control

`tools/data/results/harness/browser-demo-file-v3-2026-10-08-02.json` is
`PASS_COMPANION_FILE_TRANSCRIPTION`. The earlier `-01` receipt also passed,
before the final scope wording and extra check labels were added.

Installed Chrome 154.0.8037.98 used an NVIDIA Blackwell adapter; selected stock
Parakeet TDT v3 artifacts were the existing fp16 encoder and fp32 decoder/joint,
with the app's WebGPU-hybrid selection. This is an existing-model app control,
not a new fp16 Orukeet approval or proof of every graph node's GPU placement.

After app load/warm-up, Playwright selected `Harvard-L2-1.ogg` through the real
audio input. The displayed result and filename/history entry passed exact text
checks: `The boy was there when the sun rose.` The app returned to `Model ready`
and re-enabled file input. The receipt binds audio SHA256
`6e8379d6a016c3b9606f952d2ef5049740476dc98f2050e9bf30769af610935c`,
selected graph/tokenizer hashes, runner/consumer sources and served build bytes.
ORT sidecars came from installed local files; outside requests were blocked.
Owned Chrome and preview were closed, with an explicit ECONNREFUSED port probe.

## Reproduce

Run from the library checkout with existing Node, Python Playwright and Chrome;
no new install or model download is required. The runner builds/serves a fresh
in-memory companion preview, opens its loopback URL, loads local artifacts,
uploads audio, checks the result, closes its processes and writes new JSON.
Always choose a fresh output path; existing receipts are never overwritten.

```powershell
python tests/smoke/browser-demo-smoke.py `
  --consumer-root N:/github/asrjs/browser-demo `
  --local-model-root N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx `
  --encoder-precision fp16 `
  --audio tools/data/fixtures/audio/Harvard-L2-1.ogg `
  --expected-text "The boy was there when the sun rose." `
  --output tools/data/results/harness/browser-demo-file-v3-FRESH.json
```

The expected string is caller-supplied smoke text, consistent with the earlier
stock-v3 public-session control in `local-public-hybrid-v3-final-2026-10-03.json`.
It is not represented as an independently bound original-engine oracle. Audio
file identity is not effective decoded PCM identity. Numerical/token parity,
timestamps, quality, streaming, app cancellation and explicit model disposal
remain separate acceptance gates. Browser closure is not a GPU reclamation test.

## Regression checks and limits

Python compilation and targeted diff checks pass. Eleven browser helper tests,
nine Orukeet automatic-runner tests and fourteen local-Parakeet tests pass
(34 total). New negatives reject missing model/text options and partial,
case-changed, punctuation-changed or empty expected results. Real-browser
success establishes the rendered selectors, not those unit tests alone.

The real-browser negative control `browser-demo-file-v3-2026-10-08-03-negative.json`
deliberately supplied a wrong expected string. The actual transcript was retained,
the runner returned exit 1 / `FAIL_COMPANION_UI_SMOKE`, and it still saved the
failed receipt and closed its preview port. This is an expected negative, not
a model failure. Both final positive and negative runs bind the same runner hash.

Input/output safety preflight remains before browser launch: missing/unreadable
inputs or a protected/existing output fail without a new receipt. Failures after
launch save a failed JSON receipt with cleanup evidence. Next C14 slice is
app-level cancellation/disposal; do not collapse it into this successful file flow.
