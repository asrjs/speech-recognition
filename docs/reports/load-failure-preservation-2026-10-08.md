# C78 — Public loader failure preservation

Status: bounded G2/G6 lifecycle PASS; real Orukeet output smoke PASS.
Model numerical, held-out quality and full delivery acceptance remain open.

## User-visible change and evidence

`loadSpeechModel` and `loadBuiltInSpeechModel` no longer replace a session/load
failure with a subsequent owned-runtime or acquired-model cleanup rejection.
Both are retained in an `AggregateError`, primary first and as `cause`, without
flattening nested errors. A sole primary error retains its identity, including
`undefined`/`null`. Asset cancellation still maps to `PipelineAbortedError('load')`
and now retains the original asset error as its cause. Cancellation progress is
emitted even when cleanup fails. All cleanup is awaited before rejection.

The runtime/model ownership boundary is unchanged. Tests use the actual public
runtime and registered neutral family/backend; the owned-runtime cases intercept
only runtime allocation and restore the factory afterward. No inference engine
failure injection or real GPU error-path release is certified by these probes.

Before implementation: 28 new checks failed, 54 passed. Final added coverage is
44 tests across root/builtins and borrowed/owned runtimes, including nested
primary/cleanup failures, cancellation, single-failure identity and delayed
cleanup rejecting with Error, undefined or null. Focused suites: 105 PASS.
Source and examples typechecks PASS. Scoped ESLint: no errors, one max-lines
warning in the already-large lifecycle test. Full current-tree suite: 198 files
PASS / 7 SKIP, 1,596 tests PASS / 34 SKIP, exit 0 (34.98 seconds).
The full suite includes protected inherited work; it is not a clean release tree.
No family math, graph, tokenizer, weight, backend policy or dependency changed.

```powershell
node node_modules/typescript/bin/tsc -p tsconfig.json --noEmit
node node_modules/typescript/bin/tsc -p tsconfig.examples.json --noEmit
node node_modules/vitest/vitest.mjs run tests/high-level-pipeline-lifecycle.test.ts tests/builtins-loader.test.ts tests/owned-session.test.ts
# In a dedicated child shell; do not inherit disabled TLS verification.
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
node node_modules/vitest/vitest.mjs run
```

## Real Chrome regression and snapshot boundary

Fresh receipt: `tools/data/results/orukeet/auto-browser-load-failure-fix-2026-10-08-01.json`
SHA-256: `578880064628a4f0f48b576142108c7543711a8aa5e40aeed1d435bbc83de370`.
Chrome 154.0.8037.98, local fp32, WebGPU encoder plus WASM decoder/joint,
unchanged official effective PCM: two exact 38-token/text matches, disposal and
browser closure PASS. This exercises the normal public browser acquisition and
inference path, not actual-engine acquisition/cleanup failure injection.

Run the existing demo on 8767 and choose a fresh output:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --output tools/data/results/orukeet/auto-browser-load-failure-fix-new.json
```

The existing manifest's expected browser-code snapshot belongs to pre-C78 source.
The fresh delivery retains the same complete path set; eight served paths have
changed byte hashes (builtins plus importer/HMR propagation). No baseline was
silently rewritten. Therefore this fresh direct smoke does not imply that the
old manifest's `expected_code` check accepts current source. Preserve both old
PASS receipts and the old contract as dated evidence until a reviewed refresh.
The existing loopback server was reused; no server was stopped or replaced.

## Remaining scope and next action

This fixes the builtins acquisition boundary, not every lower-level family or
runtime failure path. A throwing user progress callback can still supersede an
error during cancellation notification; callbacks are not covered here. Next
inspect that real callback contract and add a discriminating regression before
changing semantics, or review/refresh the explicit browser-code contract against
current served bytes. Keep Orukeet feature numerical failures and Redux original
runtime prerequisites visible; no model was promoted by this lifecycle change.
