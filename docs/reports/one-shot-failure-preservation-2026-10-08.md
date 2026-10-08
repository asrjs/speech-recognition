# C73 — One-shot transcription preserves the original failure

Status: bounded G2/G6 public-API lifecycle PASS; model/backend gates unchanged.

The actual root convenience APIs `transcribeSpeech`,
`transcribeSpeechFromMonoPcm` and `transcribeSpeechBatch` previously replaced a
transcription/cancellation rejection with a later cleanup rejection from their
`finally` block. Twelve new real-runtime regression cases reproduced this loss.

Single and batch helpers now retain both failures in an `AggregateError`, with
the original transcription rejection first and as `cause`. A nested cleanup
AggregateError is preserved by identity, not flattened. A rejection containing
undefined or null is still a failure. When only transcription or only cleanup
fails, the existing exact error identity is unchanged. Cleanup remains awaited;
mono PCM inherits the single-input fix without another execution path.

Ownership, public signatures, backend selection, transcript semantics and family
math are unchanged. Existing load/session-acquisition failure handling in the
built-in loader is outside this slice and must not be inferred fixed here.

## Evidence and reproduction

Owned files: `src/runtime/load.ts`,
`tests/high-level-pipeline-lifecycle.test.ts`, this report.

- Initial targeted run: 12 RED / 29 PASS, exposing lost primary errors.
- Final focused run: 67 PASS across high-level lifecycle/API/built-in loader;
  27 added cases use the real public helpers, runtime, loader and handle leases.
  The backend/model inference and resource failures are simulated, not real ORT
  or memory-reclamation evidence.
- Final full current-tree suite: 1,552 PASS / 34 SKIP, 198 files PASS / 7 SKIP,
  exit 0, 31.32 seconds. Protected inherited changes remain part of that tree.
- Source/examples typecheck PASS; scoped ESLint has no errors and two max-lines
  warnings. Test formatting and scoped diff checks PASS.
- Full source-file Prettier check already fails on HEAD's unrelated formatting.
  A formatter touched those older sections; those formatting-only edits were
  removed before final checks. Only the two helper bodies and their API comments
  remain changed. No blanket source-format PASS is claimed.
- Independent read-only review APPROVE; its cancellation-only identity test
  suggestion was added and passed before the final suite.

```powershell
npm run typecheck
node node_modules/vitest/vitest.mjs run tests/high-level-pipeline-lifecycle.test.ts tests/high-level-api.test.ts tests/builtins-loader.test.ts
node node_modules/eslint/bin/eslint.js src/runtime/load.ts tests/high-level-pipeline-lifecycle.test.ts
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
$env:NODE_USE_SYSTEM_CA = '1'
node node_modules/vitest/vitest.mjs run
```

The final suite used these TLS settings only in its test process; no global
environment or toolchain settings changed. No model download, export, inference
oracle, browser receipt refresh or sibling write occurred.

## Next existing model/product work

Companion cancel/unload still awaits sibling write authority; Redux original
runtime still needs its approved environment prerequisite. A read-only Orukeet
audit identified a distinct diagnostic hypothesis for the next numerical slice:
round only production packed FFT's final complex-bin components to float32 before
the unchanged direct squared-magnitude expression. Existing complex-power and
FFT-operation probes change different boundaries. This hypothesis is UNTESTED,
not an improvement claim or a production patch. Use the existing boundary probe
and captured JFK frames, unchanged gates and a bounded case; do not repeat the
rejected C60 product-rounding sweep. Orukeet acceptance remains incomplete.
