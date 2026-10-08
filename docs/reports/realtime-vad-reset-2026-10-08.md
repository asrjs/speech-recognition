# C83 — Obsolete VAD errors do not escape controller reset

Status: bounded G2/G4/G6 controller lifecycle PASS (deterministic callbacks).
Whole-product/model/native/browser acceptance remains incomplete.

## User-visible outcome and root cause

`RealtimeTranscriptionController` supports an optional injected
`VoiceActivityDetector`. A push writes PCM then awaits its `analyze()` call.
When `reset()` advances the session generation during that await, a successful
old VAD result was already discarded, but a rejected old VAD result escaped
as a rejection of the old public `pushAudio()` promise. The inference callback
path already suppresses errors for obsolete generations. Four failing
regressions reproduced the VAD asymmetry before the fix.

The localized catch around detection now returns `null` only after its captured
generation becomes stale. Active-generation failures rethrow the exact value,
including `undefined` and `null`; no failure is turned into a speech/silence
observation. The existing successful-result guard and serialized operation
queue remain unchanged. Reset does not interrupt an unresolved detector promise
or reset/dispose the injected detector; those ownership/API decisions were not
changed. Family feature math, decoding and transcript/timestamp contracts stay
unchanged. Segmented utterance timestamps intentionally remain local (C63).

Affected public workflow: rolling `pushAudio()` with injected async VAD, reset,
then reuse. The current browser starter does not inject this `vad` option; its
microphone segmentation follows a separate detector/utterance path. This is
not a claim that physical-device or model-backed VAD inference was tested.

## Validation and evidence

New `tests/realtime-vad-reset.test.ts` uses explicit promise gates, not timing
sleeps. Four obsolete-rejection cases RED before / PASS after; final nine tests
also prove obsolete success is discarded, queued old pushes are dropped, reset
state remains unchanged, fresh inference resumes and active error identity is
retained. No unhandled rejection is used as an assertion. These synthetic
callbacks test orchestration, not numerical inference or actual engine release.

```powershell
node node_modules/vitest/vitest.mjs run tests/realtime-vad-reset.test.ts tests/vad-controller.test.ts tests/realtime-window-offsets.test.ts tests/realtime-latency.test.ts tests/browser-controller.test.ts tests/browser-controller-dispose.test.ts
node node_modules/typescript/bin/tsc --noEmit
node node_modules/typescript/bin/tsc --noEmit --module NodeNext --moduleResolution NodeNext --target ES2022 --lib ES2022,DOM --strict --skipLibCheck false tests/realtime-vad-reset.test.ts
node node_modules/eslint/bin/eslint.js src/runtime/controller.ts tests/realtime-vad-reset.test.ts
node node_modules/vitest/vitest.mjs run
```

49 focused PASS; source and strict standalone test typechecks PASS; scoped lint
and diff checks PASS. New test Prettier check PASS. Whole controller formatting
check flags its pre-existing multiline `withTimeOffset` call only; verified
formatter diff does not include this change, and unrelated formatting was left
untouched. Independent read-only review approves the fix and all nine tests.
Full current-tree suite exit 0: 200 files PASS / 7 SKIP, 1,666 tests PASS /
34 SKIP, 34.62 seconds. Protected inherited edits remain in this tree; this is
not a clean release-tree certification. Full suite started before formatting
the new test; the formatted test was rerun afterward (9 PASS), with no semantic
or source change after the full suite.

No weight download, graph export, engine install, sibling mutation or new
Chrome run was performed. C82's browser snapshot certifies its recorded older
delivered bytes, not this changed controller module; keep it historical until
explicit source review and a fresh independent browser receipt. Do not refresh
the pin just to obtain PASS. Orukeet feature numerics, fp16, held-out quality,
soak/resource recovery and required delivery matrix remain open.

Next: actual injected model-backed VAD reset/reuse evidence when an appropriate
owned fixture is available; independent first-delivery product/model gates may
continue without treating this deterministic lifecycle pass as model promotion.
