# C63 — Rolling controller sentence coordinates

Status: runtime contract PASS (synthetic inference); real-model long rolling
window acceptance remains UNTESTED. Track G2/G4/G6, following C62.

The public controller had a coordinate inconsistency: rolling windows offset
words, segments and timed tokens, but left canonical sentences relative to the
window. Partial/final payloads also omitted the optional sentence field.
A new regression first failed at the second window: expected sentence
start/end 0.875/1.5 seconds, received local 0.125/0.75 seconds.

The six-line fix in `src/runtime/controller.ts` offsets sentence coordinates
with the existing window origin and forwards canonical sentences into partial
and final payloads. Family decoders, merging, timing tolerances and the separate
segmented `transcribeUtterance` local-coordinate contract are unchanged.

Validation:

- New public-API test `tests/realtime-window-offsets.test.ts` feeds 34 seconds
  of uniquely identifiable PCM through an 8-second ring buffer and 6-second
  maximum windows. It independently compares every callback PCM to the original
  absolute slice; verifies nonzero origins and ring eviction; compares words,
  segments, sentences and tokens against local result plus exactly one offset;
  checks untimed token fields and original result immutability; verifies every
  synthetic committed sentence survives finalization and reset clears state.
- `node node_modules/typescript/bin/tsc --noEmit`: PASS.
- Focused controller/realtime tests: 50 PASS across seven files.
- Full `node node_modules/vitest/vitest.mjs run`: 195 files PASS / 6 SKIP,
  1492 tests PASS / 33 SKIP. Opt-in skipped real engine tests are not acceptance.
- Read-only review: approved; rolling offsets remain separate from segmented
  one-call utterance semantics.
- Fresh installed-Chrome run after this fix:
  `tools/data/results/orukeet/auto-browser-controller-resume-2026-10-08-01.json`
  reports `PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE`. Two full-clip runs
  match official token IDs/text and effective PCM; controller covers 11 seconds
  in six chunks and seven requests, final text/reset pass. All windows start
  at zero, so this does not validate nonzero-origin model timestamps.
- Runner unit tests: 12 PASS; rolling-offset regression re-run: 1 PASS.

This proves runtime data/coordinate handling with a synthetic callback, not
browser/GPU/model numerical parity or actual capture. No new download/export,
model default, package export or sibling app change. Existing Orukeet feature
numerical FAIL and quality/precision gates remain open.

Next: extend the existing Orukeet Chrome controller receipt with original and
offset canonical timestamp fields, request timestamp-bearing canonical detail,
and run bounded real inference with nonzero mature-cursor window starts. Do not
assume short zero-start controller results certify long rolling behavior, or
that text-only pending output survives audio eviction without timestamped
commit progression. Preserve full oracle-output checks and failed evidence.
