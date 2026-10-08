# C86 — Real TEN-VAD worker reset and disposal

Status: bounded real-browser packaged-model lifecycle PASS. No ASR promotion.

## Actual failure and minimal correction

The public `TenVadAdapter` initialized the packaged WASM and processed all
687 complete JFK hops, but its first reset failed with memory access out of
bounds. The worker passed the opaque handle value to `_ten_vad_destroy`.
The [official C header](https://raw.githubusercontent.com/TEN-framework/ten-vad/main/include/ten_vad.h)
requires a pointer to the handle for create/destroy, unlike process, which
takes the handle value. The local original worker in
`N:/github/ysdede/keet/src/lib/vad/tenvad.worker.ts` corroborates that ABI.

Both reset and disposal now pass `handlePtr`. Independent review also found
unchecked native destroy return codes: a nonthrowing failure could still
produce a success acknowledgement. Both calls now throw on nonzero status
before recreation/freeing/acknowledgement. Two fault-injection controls failed
before the guard and pass afterward. These four worker ABI controls execute
the actual worker module with a stubbed C boundary; they are not model evidence.
No model math, probability threshold, audio conversion or public API changed.

## Reused real-browser harness

Extended the existing offline capture/controller runner with `--scenario ten-vad`
and a model-owned probe. It uses public `dist/browser.js` / `browser-media.js`,
default real module workers and packaged assets, not a second VAD implementation.
Explicit local fixture/asset routes are hash-bound; external page/worker HTTP
requests are rejected. No server, sibling edits, weights or installations needed.
The runner now saves a failure receipt for in-browser probe errors and exits
nonzero, while preserving exclusive creation and post-run input hashes.

JFK is decoded to 176000 float32 samples at 16 kHz, PCM SHA-256
`ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
687 complete 256-frame hops are processed; 128 trailing samples remain partial
and are cleared by reset, not silently called a finalized inference frame.
Aligned and uneven chunk schedules produce identical probability and flag
bytes after reset. Speech probability SHA-256:
`dcc82cc557bb0c19f5be11d8101384bce25751086d4867563904be7cdc3360c6`.
463 hops have model speech flags. Reset-versus-fresh one-second silence yields
identical 62-hop outputs with zero speech flags; the remaining 128 samples are
also intentionally partial. Silence probability SHA-256:
`46b701af636ebce57f137b91ba52f21759324ef868988f1817ca053fd932b5ae`.
The probe observes two real successful DISPOSE acknowledgements, no worker
ERROR responses and two terminated workers; adapter idle alone cannot pass.

Packaged WASM SHA-256:
`1ec0b9640683987e15a4e54e4ce5642b2447c6e5d82b1be889b5099c75434fc3`.
Packaged bootstrap SHA-256:
`aef64f79af72c278512695285b3c933adb070e64f9a594fa4424750ee1f11ee9`.
Both source and distribution contain these assets; the sibling WASM matches.
The glue lacks UTF8ToString, so the INIT version fallback `ptr@5091` is not
a verified model semantic version; the actual asset bytes provide identity.

## Receipts and validation

Preserved failure: `tools/data/results/harness/ten-vad-model-lifecycle-2026-10-08-01.json`,
SHA `7bc020d60fce8574c13c2b5cd060e930e94910f306256396fbf5ce4438509674`.
Progress identifies `reset-after-speech`, with the successful initial model run.
Final formatted-probe/current-source receipt:
`tools/data/results/harness/ten-vad-model-lifecycle-2026-10-08-05.json`,
SHA `dbd39c9ad9d15bc7f246dfc75737bbe9f409877611704c7588063bb4d0ef5870`.
Chrome 154.0.8037.98, actual WASM worker; browser closes and all recorded
inputs/served module hashes match. Earlier successful receipts are retained,
but -05 binds the final probe, worker guards and compiled distribution.

28 focused tests PASS; full current-tree Vitest 1670 PASS / 34 SKIP, 34.52 s.
TypeScript build and example typecheck, Python/JS syntax, scoped lint/diff
checks PASS. Existing capture-owner browser scenario independently re-passes:
`tools/data/results/harness/capture-owner-post-ten-vad-2026-10-08-01.json`.
Two real fake-device capture sessions close contexts/end tracks; this is not
physical microphone evidence. Read-only independent review approves.

```powershell
node node_modules/typescript/bin/tsc -p tsconfig.json
node scripts/copy-runtime-assets.mjs
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
python tools/browser-validation/capture-lifecycle-smoke.py --scenario ten-vad --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --audio tests/fixtures/jfk2.en.wav --output tools/data/results/harness/ten-vad-FRESH.json
node node_modules/vitest/vitest.mjs run tests/ten-vad-worker-abi.test.ts tests/browser-ten-vad.test.ts
```

## Limits and next gate

This is model-backed worker reset/chunking/teardown, not original-engine parity,
VAD accuracy against human labels, ASR, physical capture, VAD-to-utterance
controller acceptance or a memory soak. The separate TenVadBackend chunking
adapter still uses energy fallback. Default microphone presets may disable TEN;
neither these results nor their capture-only controls certify TEN integration.
The adapter's existing disposal-error swallowing is not changed; the probe
explicitly rejects ERROR or absent acknowledgements. Partial-initialization
resource cleanup remains a separate gate. Next exercise actual TEN observations
through the public speech detector/controller with identified PCM windows.
The Orukeet code pin and its unrelated numerical gates are not auto-refreshed.
