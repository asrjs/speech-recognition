# TEN-VAD repeat fix — 2026-10-23

Source fix: src/runtime/streaming-detector.ts. Regression: tests/streaming-detector-vad-history.test.ts.

A finalized-frame floor prevents old neural history from replaying prior utterances; active speaking status also prevents an immediate silence finalization during VAD smoothing warmup. Rejected segments consume history too; start resets the floor. Public API/model decode unchanged.

Validation: 8 deterministic regression cases; full library 1867 passed / 36 skipped; typecheck passed; lint 0 errors / 13 existing warnings; root import 44. Demos: 59 tests + four builds. Real Parakeet v3 + TEN-VAD controlled microphone: old package FAIL (47 queued transcriptions, repeated text); patched package PASS_CONTROLLED_FIXTURE (4 segments/jobs/completions, full normalized oracle match). Rough-only control also matches. Installed detector matches built detector SHA256.

Full Turkish report and immutable before/after evidence: [companion demo report](../../../demos/docs/reports/ten-vad-repeat-fix-2026-10-23.md). Physical microphone UNVERIFIED; CI not run for local changes. Prior failure receipts retained.
