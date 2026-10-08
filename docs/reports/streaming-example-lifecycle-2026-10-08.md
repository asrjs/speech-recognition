# Streaming example lifecycle — C70

Status: bounded G2/G5 example lifecycle PASS. Not model inference acceptance.

## User-visible change

`examples/streaming.ts` protects model/session/wrapper acquisition as well as
push/finalize. It awaits wrapper cleanup and independently attempts runtime-owner
cleanup even if the wrapper fails. Primary errors (including null/undefined) and
all cleanup errors are preserved in an AggregateError with the original cause.
Successful results are not returned until cleanup completes; cleanup-only failure
rejects. The example does not separately dispose an already runtime-owned model.
The existing model, audio, options and partial/final envelope are unchanged.

Ownership was checked against `DefaultStreamingTranscriber.dispose`,
`DefaultSpeechRuntime.dispose/trackLoadedModel`, and NeMo TDT model session tracking.
The wrapper releases its injected session; runtime releases loaded models and
remaining sessions. No generic runtime or family math changes were needed.
This recipe uses windowed offline ASR and synthetic silence, not native streaming
or a recognition-quality fixture.

## Validation

Initial 11 tests: 8 RED / 3 PASS on the old implementation. Final 14 consumer
simulation tests PASS: acquisition failures (load/session/constructor), push/finalize
failures, preserved primary and nested cleanup errors, cleanup-only rejection,
ordered cleanup, pending operation and pending wrapper/runtime cleanup.

```powershell
npm run typecheck
node node_modules/vitest/vitest.mjs run tests/streaming-example.test.ts tests/detailed-native-example.test.ts tests/runtime.test.ts tests/streaming.test.ts
node node_modules/eslint/bin/eslint.js examples/streaming.ts tests/streaming-example.test.ts
git diff --check -- examples/streaming.ts
```

Source/examples typecheck PASS; 41 focused tests PASS; scoped ESLint and formatting
PASS. The new tests simulate consumer dependencies, not actual engine resources.
Related runtime/streaming tests exercise the existing ownership implementations.
An additional 15 streaming owner/control/consumer tests PASS (56 focused total).
Independent read-only review: APPROVE, no actionable findings.

## Limits and next step

No original-engine numerical gate, model quality, native-engine release or live
streaming acceptance is closed here. The previous automatic Chrome run remains
separate evidence in `auto-browser-resume-jfk-2026-10-08-01.json`: real hybrid,
two exact 38-token/text repeats, same PCM, controller/timing and cleanup PASS.
It predates this example-only edit and does not exercise this recipe.

Next bounded core lifecycle check: concurrent `DefaultStreamingTranscriber.dispose`
calls currently return early on subsequent calls while initial cleanup is pending.
Prove it with the actual wrapper and a deferred session disposer, then memoize
the in-flight cleanup without changing model-specific decoding or acceptance.
Long controller aggregate text and Orukeet numerical gates remain open.
