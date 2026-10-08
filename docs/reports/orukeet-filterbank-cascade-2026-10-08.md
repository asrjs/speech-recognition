# C58 — Filterbank × normalization interaction on production power

Status: MEASURED / combined-fix hypothesis REJECTED for speech. Production
feature parity remains FAIL. No production processor or tolerance changed.

The existing `probe_filterbank_features.py` now also runs the existing diagnostic
width-8 float32 cascade normalizer on each raw boundary. This tests whether the
filterbank arithmetic mismatch only becomes consequential after reference-shaped
normalization. It is a six-condition comparison, not a new inference engine.
All three raw paths use identical captured production power; baseline output is
checked against the unmodified imported processor. The cascade helper's source
hash joins the existing capture/build/diagnostic identities in the receipt.

## Results at unchanged atol=1e-5 / rtol=1e-4

| Input / raw condition | Production normalization failures | Diagnostic cascade failures |
| --- | ---: | ---: |
| JFK / production filterbank | 313 | 315 |
| JFK / actual captured NeMo filterbank | 313 | 315 |
| JFK / rounded triangle coefficients | 313 | 315 |
| Pure silence / all three conditions | 12,800 | 0 |

JFK cascade maximum absolute error is `0.0003426074981689453`, first divergent
flat index 45; silence cascade is byte-value exact to the normalized reference
(max absolute error zero). Actual-buffer and triangle controls keep the same
raw/features; this does not turn the speech result into a passing gate. All
receipts retain production failures separately from diagnostic comparisons.

Evidence:

- `tools/data/results/orukeet/filterbank-cascade-jfk-2026-10-08.json`
- `tools/data/results/orukeet/filterbank-cascade-silence-2026-10-08.json`

The unchanged compiled `dist/audio/js-mel.js` used here has SHA256
`067a902ceea1f260ce811f2bc04a7112e1a3a0e0cc17b99c6da9aa15a1a6a32f`.
This is a build-bound diagnostic, not general certification of source revisions.
Both original-engine captures bind official Orukeet weights SHA256
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
No model was re-exported, downloaded or loaded for this cached-stage experiment.

## Reproduction

Use the existing NeMo reference overlay interpreter (its location is historical;
this is **not** resumed Nemotron work). It supplies NumPy for capture/comparator
reading; the temporary feature candidate executes in Node, not native Torch.
Choose a fresh output path:

```powershell
tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/probe_filterbank_features.py --capture tools/data/results/orukeet/official-jfk-filterbank-stages-2026-10-03.json.gz --capture-sha256 e6e3c145a82ac609c87b132295a0c9157865c1999e58f88a65d36667db8154d7 --weight-sha256 031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56 --sample-id jfk-short --output tools/data/results/orukeet/filterbank-cascade-jfk-FRESH.json
```

For silence use `official-pure-silence-filterbank-stages-2026-10-03.json.gz`,
capture SHA256 `d8116556e370b92494052d3c80bdf3c9e3b56e59a9c6e95da7d8d033082599cb`
and sample ID `pure-silence-1s`. Diagnostic exit 0 / MEASURED means the experiment
ran, not that feature or model acceptance passed.

Five matrix-probe tests, six captured-filterbank guards, six Python normalization
tests and five JS normalization tests pass (22). New regressions keep production
FAIL visible beside a diagnostic PASS and reject nonfinite cascade features.
The first schema-extension edit was reverted after an old synthetic fixture
lacked the new field; explicit new-contract tests then reproduced missing
comparisons/nonfinite handling before the minimal implementation was reapplied.
Original assertions/tolerances remain intact. Both real inputs passed original
instrumentation/output-immutability guards.

## Resume boundary

Do not rerun the old f32-butterfly or thread sweeps; the README overlay and later
receipts already reject them. Likewise the speech/silence captured-PCM full
boundary control was already run in `captured-pcm-boundary-handoff-2026-10-03.json`;
another unchanged full run would not discriminate a new cause. Isolate a new
portable FFT/window/power arithmetic hypothesis against those captured inputs
before another expensive pipeline or shared processor change. Normalization of
the identical official raw boundary passes, while production speech still fails.
Silence normalization evidence alone cannot authorize this combined processor
as the model solution. Browser output/lifecycle passes do not waive numerics.
