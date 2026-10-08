# C56 — Requested missing inputs leave a failed browser-harness receipt

The classic companion browser runner previously validated selected artifacts and
hashed input files before entering any receipt-writing error boundary. Missing
consumer, model or requested audio therefore exited without the promised result.
This was reproduced by three negative subprocess cases before changing code.

`tests/smoke/browser-demo-smoke.py` now saves
`FAIL_COMPANION_INPUT_PREFLIGHT`, exit 1, the error and requested input paths,
with explicit `browser_started: false` / `preview_started: false`. It does not
start the build, Chrome or model inference for those failures. The selected
artifact and successful-run hash checks remain unchanged. Output safety stays
outside this error boundary: a protected/existing output is never overwritten
even to record failure. Invalid CLI combinations still exit argparse validation.
Dependency/import failures are not covered by this input-only boundary.

Negative live receipt:
`tools/data/results/harness/browser-demo-input-preflight-negative-2026-10-08.json`.
It deliberately requests `INTENTIONAL-MISSING-C56.wav` with existing stock-v3
model selection; exit 1 and the failed preflight receipt are the expected result.

Twelve browser helper tests, nine automatic Orukeet tests and fourteen
local-Parakeet helpers pass (35 total); Python compilation and targeted diff
checks pass. This is not model acceptance or a rerun of the C55 file-model row.
The updated runner also passed a fresh installed-Chrome UI-only regression:
`tools/data/results/harness/browser-demo-ui-preflight-regression-2026-10-08.json`.
Owned preview closure was verified with ECONNREFUSED; no weights were loaded.

## Next actual app workflow

Live sibling source inspection shows no public cancel/unload action:
`N:/github/asrjs/browser-demo/src/App.jsx` file transcription has no abort signal,
and model release is internal to loading/error/unmount. Browser closure cannot
prove the React cleanup awaited engine disposal. The existing primary
`tests/smoke/local-parakeet.mjs` public-API cancel/reuse/dispose control is useful
but is not that application's cancellation/unload journey.

Sibling write authorization has been requested to add those real user controls.
Do not mark C14 app cancellation/disposal accepted before implementation and real
browser evidence. Meanwhile independent library/model work remains actionable;
the whole-product goal remains active and Orukeet numerical gaps are unchanged.
