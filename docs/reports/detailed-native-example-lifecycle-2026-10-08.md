# C69 — Detailed/native public example owns cleanup

Status: example lifecycle regression PASS, G2/G5. Actual engine release and
model/host numerical acceptance remain separate; no core runtime/math change.

`examples/detailed-and-native.ts` previously created a runtime, loaded a model
and created a session without any cleanup. Seven initial regressions failed:
success, pending transcription, load/session/transcription failures and cleanup
error reporting. The fixed recipe awaits transcription before entering cleanup
and always awaits its runtime owner. Original envelope and detailed/native
options are unchanged.

Live ownership inspection: `DefaultSpeechRuntime` tracks/disposes loaded models;
the NeMo TDT model owns its sessions. Therefore this example calls only
`runtime.dispose()`, avoiding redundant explicit child disposal. An initial
passing explicit session/model/runtime version was simplified after ownership
inspection, not because an acceptance assertion was weakened. Tests now exercise
the real DefaultSpeechRuntime's loaded-model ownership with simulated
inference/session resources; they do not load weights or make engine claims.

If work fails but cleanup succeeds, the original throwable is rethrown unchanged.
If cleanup fails, an AggregateError retains cleanup errors and any primary
throwable as cause, including null/undefined. Nested owner aggregates remain
intact instead of being flattened or hidden. Cleanup failure cannot report
successful transcription.

Validation: 10 example tests plus 12 runtime and 29 native-local-example tests
PASS (51 total); source/examples `npm run typecheck`, scoped ESLint and diff
checks PASS. Cases include pending transcription, pending asynchronous owner
cleanup, partial acquisitions, nested owner errors and falsy primary throws.
Commands: `node node_modules/vitest/vitest.mjs run tests/detailed-native-example.test.ts
tests/runtime.test.ts tests/native-local-example.test.ts`; `npm run typecheck`;
`node node_modules/eslint/bin/eslint.js examples/detailed-and-native.ts
tests/detailed-native-example.test.ts`.

Scope: synthetic silence PCM in a code recipe is not a new native-engine port or
quality test ("native" here names the response flavor). No model, dependency,
public export or sibling application changed. Next inspect other existing
consumer recipes for initialization outside cleanup scopes, starting with
`examples/streaming.ts`; do not infer they are safe from a happy-path finally.
