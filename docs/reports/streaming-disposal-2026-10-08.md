# C71 — Shared streaming disposal wait

Status: G2/G4 lifecycle PASS; model/engine acceptance unchanged.

The publicly exported `DefaultStreamingTranscriber` previously returned early
from repeated dispose calls, before the first queue drain/session release finished.
A session release failure was lost to later callers. The class now memoizes its
cleanup promise before synchronous abort dispatch, including listener reentry.
Every caller waits for the same completion/failure; session release happens once.
Terminal-use guards, stale-result suppression, audio/decode/output contracts and
model-family ownership are unchanged. No model-specific production branch added.

Seven new tests use the actual public class with controlled session resources:
pending release, concurrent/later rejection (Error/undefined/null), synchronous
release throw, active/queued work drain, terminal use, and abort-listener reentry.
All seven were RED before the fix, then PASS. These tests establish orchestration,
not native/GPU resource reclamation or ASR quality.

```powershell
npm run typecheck
node node_modules/vitest/vitest.mjs run tests/streaming-disposal.test.ts tests/streaming.test.ts
node node_modules/eslint/bin/eslint.js src/inference/streaming/transcriber.ts tests/streaming-disposal.test.ts
node node_modules/vitest/vitest.mjs run
```

Source/examples typecheck and scoped lint PASS; 12 focused tests PASS.
Full current-tree regression: 198 files PASS / 7 SKIP; 1,525 tests PASS / 34 SKIP,
36.72 s, exit 0. Opt-in backend skips are not inference acceptance. The full run
includes inherited dirty changes; no claim those unrelated edits are committed.
Independent read-only review APPROVE; reviewer separately verified 20 streaming
tests and source/examples typecheck PASS.

Affected direct consumer: `examples/streaming.ts`. Higher-level loaded handles
already memoize their wrapper disposal separately; owner regressions also pass.
The earlier automatic Chrome receipt does not certify this subsequent core edit;
its controller uses a different shared implementation. Original-engine numerical,
long controller aggregate-text and held-out quality gates remain open.
