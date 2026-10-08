# C80 — Preserve cancellation notification failures

Status: bounded G2/G6 lifecycle PASS; model/delivery acceptance unchanged.

## User-facing contract

Root `loadSpeechModel` and `loadBuiltInSpeechModel` no longer lose an original
load cancellation or its cleanup failure when a synchronous user progress
callback throws while receiving `phase: cancelled`. After awaited cleanup,
the rejection retains `[primary, cleanup?, notification]` in an `AggregateError`,
with the normalized/original primary as `cause`. Nested cleanup errors and
callback rejection values including `undefined` and `null` retain identity.
Asset cancellation still normalizes to `PipelineAbortedError('load')` with the
asset error as its cause. No callback failure means the existing error path.

The guard is confined to the cancellation notification in the acquisition
catch. It does not suppress callback errors elsewhere, promise-returning
callbacks, or change hook-before-direct callback short-circuit behavior.
No family math, backend selection, weights, transcript or export changed.

## Verification

48 new RED regressions before the source edit; 48 PASS afterward. They cover
root/builtins, owned/borrowed runtimes, hook/direct callbacks, cleanup success
and failure, Error/undefined/null callback throws, and cleanup-before-notification
ordering using the real runtime with a registered neutral family/backend.
Owned-runtime tests intercept allocation only and restore the factory.

153 focused tests PASS. Both source and example typechecks PASS. Scoped lint
has no errors and one existing max-lines warning in the lifecycle test file.
Full current-tree suite: 198 files PASS / 7 SKIP; 1,644 tests PASS / 34 SKIP,
exit 0, 32.70 seconds. Protected inherited changes are included in this tree;
this is not a clean release-tree certification. Independent read-only review
approved the final source/test diff and reran 142 lifecycle tests.

```powershell
node node_modules/vitest/vitest.mjs run tests/high-level-pipeline-lifecycle.test.ts tests/builtins-loader.test.ts tests/owned-session.test.ts
node node_modules/typescript/bin/tsc -p tsconfig.json --noEmit
node node_modules/typescript/bin/tsc -p tsconfig.examples.json --noEmit
node node_modules/eslint/bin/eslint.js src/runtime/builtins.ts tests/high-level-pipeline-lifecycle.test.ts
# Dedicated child-shell setting, not a global configuration change:
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
node node_modules/vitest/vitest.mjs run
```

## Actual browser boundary

Fresh receipt:
`tools/data/results/orukeet/auto-browser-cancel-notification-2026-10-08-01.json`.
SHA-256: `897f5e5ffeaed91ac06f8b931bfad4e39abce54e2d1bae4a657fac7a4849101e`.
Chrome 154.0.8037.98; 38 exact token IDs per repeat.
Direct runner reports `PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE`: local fp32
Orukeet, WebGPU encoder and WASM decoder/joint, two complete original-oracle
token/text matches on the official effective PCM, awaited disposal and Chrome
closure. This proves the normal browser path, not injected real-engine failure
release or callback failure in Chrome. The runner records delivered code bytes;
the manifest's C78 expected-code snapshot is not refreshed or certified for
this changed source. Retain previous receipts as dated evidence.

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --output tools/data/results/orukeet/auto-browser-cancel-notification-new.json
```

The browser child inherited disabled TLS verification from the host environment
(warning recorded in command output); network routing was loopback-only and
local development certificates were explicitly ignored by the existing runner.
This run is not a TLS/network-security acceptance claim. Full Vitest ran with
TLS verification enabled in its dedicated child shell.

Next: explicitly review delivered-code changes before refreshing a separate
manifest snapshot. Orukeet feature numerical/quality gates and the Redux
original-runtime prerequisites remain open; do not repeat rejected FFT controls
or claim model acceptance from this lifecycle slice.
