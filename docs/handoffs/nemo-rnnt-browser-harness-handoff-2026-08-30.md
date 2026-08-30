# Handoff: eou-120m browser harness slice (handed over 2026-08-30)

Status: the slice was STOPPED and rolled back by operator instruction. The
completed, pushed work (TDT grid batching, RNNT opt-in grid batching,
quant-aware default) is unaffected. This document hands the browser-harness
slice to the next agent with the operator correction recorded up front.

## OPERATOR CORRECTION (binding for the retry)

Model assets must be loaded from the Hugging Face remote repos and/or local
Hugging Face model folders, using the library proper source resolution and
JS-compatible fs paths. Do NOT wire browser demo pages to direct N:/models
paths through vite dev-server filesystem access or raw URL artifacts. The
direct-URL wiring was the mistake that produced the external-data load
failure below; it bypasses the repo-listing guards in the artifact
materialization step (the optional-sidecar probe consults the repo listing
and correctly returns undefined when a sidecar is not listed - a bare URL
locator cannot do that check).

## Mission

Browser demo + measurement page for the Parakeet Realtime EOU 120M (RNNT)
preset inside N:/github/asrjs/webgpu-agent-test (Chrome headless real-WebGPU
harness, the promotion-grade measurement authority), so the RNNT
grid-batching A/B (shipped as 31e94bd, Node-only evidence so far) extends to
the browser.


## Reference integrations (study these first)

- N:/github/asrjs/streaming-demo - operator-confirmed multi-model demo that
  already wires parakeet-realtime-eou-120m-v1 alongside TDT v2/v3 (see
  src/app/constants.js). This is the integration reference for how the
  eou-120m preset is selected and loaded.
- N:/github/asrjs/parakeet-ui-webgpu/src/constants.js - second reference.
- webgpu-agent-test harness pattern: parakeet-tdt.html +
  src/parakeet-tdt-webgpu.js + scripts/run-parakeet-tdt-webgpu.mjs plus the
  test-result capture plugin. The runner launches headless Chrome with
  WebGPU flags and polls for a result payload POSTed to the capture
  endpoint.

## What was attempted and rolled back

Files created then DELETED (keep them deleted or rebuild correctly):
parakeet-rnnt.html, src/parakeet-rnnt-webgpu.js,
scripts/run-parakeet-rnnt-webgpu.mjs, plus a /parakeet-eou/ route added to
gigaam-asset-plugin.mjs (reverted). The page wired a direct-URL source
(artifacts pointing at a /parakeet-eou/ route backed by N:/models paths).

Observed failure chain (keep for diagnosis):
1. Direct-URL source led the artifact materializer to synthesize a decoder
   filename (basename without extension).
2. The optional sidecar probe (decoder filename + .data) with an empty repo
   listing resolved a URL locator (no listing guard is possible for direct
   sources), so decoderDataUrl became truthy.
3. createOrtSession then set the ORT externalData session option, ORT Web
   fetched the bogus locator, and the session create failed with: failed to
   load external data file: /parakeet-eou/decoder_joint-model.
4. The fp32 decoder_joint-model.onnx itself is self-contained (binary scan
   found no external-data strings; onnxruntime-native loads it fine), so the
   failure is purely an artifact-flow bug, not a model problem.
5. A secondary probe (onnxruntime-web in Node against the raw file path)
   failed with: fetch failed - probe artifact only: ORT Web fetches URLs,
   it does not read bare filesystem paths. Do not repeat that probe.

## Correct approach for the retry

1. Use the huggingface source kind (remote repoId
   ysdede/parakeet-realtime-eou-120m-v1-onnx, revision 6d6be8e9113b4aa8 from
   src/presets/parakeet/catalog.ts) or the equivalent local HF model folder
   source - let the library resolve encoder/decoder/vocab and handle
   external-data probes via the repo listing.
2. Keep the measurement essentials from the rolled-back page (they were
   sound): model id, JS fbank preprocessor (this export has no nemo128
   sidecar), quant flags for encoder/decoder, gridBatching passthrough with
   absent = library default, warmup + repeat runs, per-run
   decoderGridBatchRuns metric, jfk-short.wav oracle text (and so my fellow
   americans ask not what your country can do for you ask what), and the
   result payload shape matching the TDT runner.
3. Run the A/B cells: fp32 decoder grid ON vs OFF (Node evidence predicts a
   win), int8 ON vs OFF (Node evidence shows transcript divergence risk; the
   page should make any DIFF visible via its fixed oracle).
4. After browser evidence: update
   docs/reports/nemo-rnnt-grid-batching-2026-08-30.md and the goal file.

## Verified context that carries over

- Shipped commits: 8c32b9b (TDT grid batching + gate), 827079a (RNNT opt-in
  grid batching), 31e94bd (quant-aware default: fp32 ON, int8/fp16 opt-in).
- Node real-artifact A/B (exact transcript parity unless noted): jfk-short
  fp32 216.2 vs 265.0 ms (18 percent win), fp16 parity, int8 354.7 vs 211.0
  (40 percent regression); tr-tdk-18s fp32 342.5 vs 408.0 (16 percent win),
  int8 grid ON produced a DIFFERENT transcript than OFF. Evidence JSONs:
  tools/data/results/nemo-rnnt/parakeet-eou120m-*.json.
- Operator guidance: eou-120m is a small model - fp32 is the right operating
  point; no quantization pressure.
- Tooling gotchas for exec-cell file edits: raw template strings break on
  dollar-brace interpolation (use a placeholder char substituted
  afterwards); tool-call JSON decoding halves literal backslashes (repair
  path and escape lines via a python fixer using chr(92)); python patchers
  must read/write with newline preserved and detect CRLF vs LF; keep exec
  outputs small.

