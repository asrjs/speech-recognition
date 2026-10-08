# C81 — Valid silence in the actual browser model workflow

Status: bounded G1/G5/G6 same-PCM full-path output smoke PASS. Numerical model
acceptance remains incomplete; this is not held-out quality or silence-soak proof.

## Failure first, then repair the evidence boundary

The existing pure-silence browser receipts were head-only or frozen-encoder
recurrence controls. This run used the production JS Feature Extractor, actual
WebGPU encoder and WASM decoder/joint through the demo's public local-model API,
not injected features/encoder outputs or a replacement decoding loop.

The original one-second zero PCM fixture and NeMo oracle already existed:
`pure-silence-1s-16k-pcm16.wav`, WAV SHA-256
`57d8dd674034eb744bd8dfa71344ee02e1a8e709563cef10108eda63663675c8`;
effective 16 kHz mono, 16,000 float32 samples, no padding, f32le SHA-256
`4f7988030a00d082fe445e00a2ac5dab502300ff1b80e8592dd569867b60ef74`.
Original oracle: `official-pure-silence-1s-summary-2026-10-01.json`, SHA-256
`b26ec8256944cbb8dff8b55516e1d07f00f293538e515cca56707be3cafaff79`.
Expected output is empty text and explicit empty token IDs, not missing inference.

First fresh receipt `auto-browser-pure-silence-hybrid-2026-10-08-01.json`
remains FAIL, SHA-256
`4900421a5d3cbf441b8b976469864489d6f3eca80ad1c6e323f71148e5b95d0d`.
Both outputs already matched the oracle and PCM, actual backends and disposal;
the page alone rejected valid silence with `Error: Empty transcript`.

The demo now accepts empty text while requiring string canonical text and an
explicit array of safe nonnegative native token IDs. Missing diagnostics are
not manufactured into `[]`. The same demo-private payload validation applies
to normal repeats and optional controller requests; no generic runtime or
model-family code changed. The external runner still checks every complete
output against the bound original oracle. A speech fixture returning empty
output still fails its oracle; an empty list of inference records cannot pass.

## Tests and actual browser evidence

Seven initial regressions RED before the main fix, then an additional controller
missing-ID regression RED before sharing validation. Final 13 demo tests PASS,
including valid silence, speech-repeat mismatch, malformed/missing text/IDs,
controller missing versus explicit empty IDs, awaited cleanup and button reset.
These use the real page event handler with stub DOM/model/audio/controller;
they are not inference or real-controller acceptance. Existing Python browser
runner's 19 tests PASS. Syntax, scoped test lint, example typecheck and diff
checks PASS. Independent read-only review approved the final refinement.
Final full current-tree suite: 199 files PASS / 7 SKIP; 1,657 tests PASS /
34 SKIP, exit 0, 39.65 seconds. This tree includes protected inherited edits;
it is not a clean release-tree certification.

Final current-source real Chrome receipt:
`tools/data/results/orukeet/auto-browser-pure-silence-hybrid-2026-10-08-03.json`.
SHA-256: `af5263b768a0e08075afea81bde86f6f9f5f033f8a030ddb26055bf40171319e`.
Chrome 154.0.8037.98, local own fp32 graphs, two repeated empty token/text
outputs, matching original PCM, observed WebGPU encoder/WASM decoder+joint,
13 decode iterations per run, awaited disposal and browser closure PASS.
All served HTML/JS/TS/WASM bytes are captured by the existing runner. TLS
verification remained enabled in this child; local dev certificate handling is
the runner's explicit loopback-only policy.

Intermediate fixed-page silence receipt `...-02.json` also passes. Speech
control `auto-browser-silence-fix-jfk-2026-10-08-01.json` passes two exact
38-token/text results on official PCM, SHA-256
`642d839e2bf31298fc4a2a79cd41c13ed1f292fc0fd55bae8d50ba275ee2facd`.
That speech run preceded the controller validation refinement; its delivered
code is dated evidence, not a claim that it executed the final page bytes.

```powershell
# With the existing reviewed demo server on 8767; choose a fresh receipt name.
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/results/orukeet/pure-silence-1s-16k-pcm16.wav --oracle tools/data/results/orukeet/official-pure-silence-1s-summary-2026-10-01.json --expected-pcm-sha256 4f7988030a00d082fe445e00a2ac5dab502300ff1b80e8592dd569867b60ef74 --output tools/data/results/orukeet/auto-browser-pure-silence-new.json
node node_modules/vitest/vitest.mjs run tests/orukeet-smoke-demo.test.ts
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
```

No silence-specific production shortcut, token suppression, oracle edits,
weights/downloads, tolerance or acceptance waiver was introduced. Existing
strict feature numerics still fail; matching final empty output does not erase
their normalization differences. Pure WASM encoder, streaming, fp16, broader
quality, memory recovery and soak are not certified by this hybrid run.
The manifest's C78 code snapshot remains untouched and stale for C80/C81;
review delivered-code changes explicitly before a separate snapshot refresh.
