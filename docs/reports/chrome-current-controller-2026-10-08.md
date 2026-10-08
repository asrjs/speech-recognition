# C89 — Current Chrome ASR and windowed controller verification

Status: actual Orukeet fp32 WebGPU/WASM output smoke PASS; full numerical model
acceptance remains open. No source, weight, oracle or code-pin change.

## Current evidence

After C88 (`ecebd038`), the existing automatic URL runner was exercised against
the already-running local demo on HTTPS port 8767. No server was restarted and
no browser, dependency or model was downloaded.

`tools/data/results/orukeet/auto-browser-c89-seed-2026-10-08-01.json` passes two
38-token and exact-text transcriptions against the official same-PCM JFK oracle.
The opt-in controller also passes: it receives 176,000 source frames and submits
seven overlapping ASR requests totaling 832,000 request-frames. It finalizes to
the oracle text and resets to frame zero, empty text and
not-finalized state. Timing acceptance means local-to-global coordinate transfer,
not a latency budget. This is chunk-fed windowed offline ASR, not native model
streaming, microphone capture or model-driven VAD segmentation.

The initial suspicion that C88 invalidated the C83 browser pin was disproven.
The served-code comparison finds 291 unique paths / 298 deliveries, with every
one of the 281 C83 entries unchanged in both size and SHA-256, no missing paths
and no conflicting duplicate bodies. Ten extra paths belong to the opt-in
windowed controller. `src/realtime.ts` exports `src/runtime/controller.ts`;
the demo's `windowed-controller.js` imports that API. This is distinct from
`src/runtime/browser-controller.ts`, the microphone controller fixed by C88,
which is not delivered by this demo. An independent read-only agent verified
this correction. C83 and the manifest were therefore deliberately preserved.

The independent manifest run subsequently passes all 19 checks without updating
the pin:

- Parent: `tools/data/results/orukeet/manifest-chrome-c89-current-2026-10-08-01.json`.
- Child: same path with `.browser.json` appended; SHA-256
  `bad93ad9887d469bd6399b58854a6a63f99e05d72d64307de13474d60b0d5dd2`.
- Chrome 154.0.8037.98, NVIDIA Blackwell adapter, JS features, fp32 WebGPU encoder
  and WASM decoder/joint. Both outputs have 38 exact tokens and exact text.
- Effective PCM SHA-256
  `ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
- Source/artifact/oracle/runner bindings, exact expected delivered code, awaited
  model disposal and browser closure pass; no page errors or blocked requests.

The seed inherited `NODE_TLS_REJECT_UNAUTHORIZED=0` from its parent environment
and emitted that warning; it is preserved rather than presented as a secure
environment control. The independent manifest command sets that variable to
`1` only in its child shell. The runner's explicit loopback development-certificate
exception remains. No global TLS setting was modified.

Lightweight regression: 19 automatic-runner and six browser-adapter tests PASS.

## Reproduce

From the repository root, with the verified existing demo on 8767:

```powershell
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-chrome-FRESH.json
```

For the separate windowed controller scope, use a new output path:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --verify-controller --output tools/data/results/orukeet/auto-browser-controller-FRESH.json
```

Choose fresh filenames; existing receipts fail preflight instead of overwriting.
Check the actual page before reusing a server. A served-code mismatch still
requires independent review; a source edit outside the delivered graph alone
does not justify rewriting the code contract.

This is not full feature/encoder numerical parity, held-out human-label quality,
fp16 promotion, native streaming, physical microphone or GPU-memory soak. The
Orukeet production-feature gate remains FAIL/DEFERRED at unchanged tolerances.
Next product integration gate is actual capture through utterance publication
into ASR, separately from the verified fake-device TEN diagnostics and file-fed
windowed controller. Do not combine their separate PASS receipts into an end-to-end
microphone ASR claim.
