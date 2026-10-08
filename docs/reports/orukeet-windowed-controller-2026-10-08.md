# Orukeet automatic Chrome windowed-controller regression

Status: PARTIAL / actual controller model window FAIL. No numerical gate waiver.

The recovered installed-Chrome runner now optionally exercises the public
RealtimeTranscriptionController through `--verify-controller` (`controller=1`).
The example remains model-neutral: fixed-size PCM ingestion and rolling offline
ASR windows; model loading and family-specific decoding stay in the existing
Orukeet page/public model callback. No native streaming state or microphone claim.

## Verified evidence

- 31 Vitest tests pass: new helper (3), browser-controller (6), controller-dispose
  (14), realtime-latency (8). Helper tests independently compare each callback
  window's PCM against the original slice, and preserve reset evidence on failure.
- 12 Python runner regression tests pass. Optional gate requires contiguous
  complete chunk coverage, actual component backend labels, multiple requests,
  partial/final events, exact final oracle text and reset. Final update text is
  independently required to agree with the oracle (read-only review finding fixed).
- Actual installed Chrome receipt:
  `tools/data/results/orukeet/auto-browser-controller-jfk-2026-10-08-01.json`.
  Root FAIL. Normal two full-clip transcriptions match all 38 official IDs and
  text, full effective PCM identity is bound to the official digest, served code
  capture and backend checks pass; model disposal and browser closure pass.
- First 32000-sample controller window fails with physical encoder output
  `[1, 1024, 26]`, valid length 25. Its PCM digest and request range remain in
  the receipt; no successful request/partial event is fabricated. Controller
  reset state is zero frames, not finalized, empty text.
- Fresh default-mode control after the final runner gate edit:
  `auto-browser-controller-baseline-jfk-2026-10-08-01.json` is
  `PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE`, two exact 38-token/text runs,
  controller explicitly `SKIP_NOT_REQUESTED`, awaited cleanup. The default
  repaired automation remains usable independently of the new optional failure.

## Next step / limits

Investigate `src/models/nemo-tdt/executor.ts` layout resolution: both physical
axes exceed the valid length in short padded windows. Existing equality and
padded-axis comparisons reject that case. Determine layout from an explicit
graph/decoder feature contract; do not simply choose the larger axis, hardcode
1024, accept ambiguous axes, remove cropping, or alter the oracle.

Add a minimal short-padded regression plus ambiguous/invalid-length controls
before any core fix, then rerun this fresh Chrome mode. The preserved failed
receipt must not be overwritten. Full-clip oracle does not establish per-window
numerical/token parity, and a matching final transcript would not certify long
rolling windows, native streaming, capture, fp16, held-out quality or soak.
