# C64 — Real rolling-window timing receipts

Status: timing transfer PASS; long controller/full-clip text comparison FAIL.
Tracks G4/G5/G6; follows C63. Overall model/product acceptance remains incomplete.

The existing URL-driven Chrome runner now exercises a detailed canonical model
response. The neutral demo controller clones window-local words, segments,
sentences and tokens into `inferenceResults[].localTiming`, and global canonical
and partial fields into `updates[]`. `inferenceIndex` associates only updates
created by that operation's inference. A no-inference finalize update records
`null`, rather than borrowing the previous call. Missing arrays remain null.
No family math, decoder, merger or runtime defaults changed.

Validation:

- JS syntax and TypeScript no-emit checks PASS.
- Focused realtime/controller suite: 52 tests PASS across seven files.
- New 34-second simulation verifies nonzero origins and immutable snapshots.
  A separate two-sentence regression reproduces finalize without a new window.
  The initial single-sentence fixture did not exercise that branch; actual
  merger/window source showed it stays pending, so the fixture was corrected
  before verifying the null-index fix. Read-only review approved both changes.
- Real Chrome repeat8 fixture: 149.7125 seconds / 2,395,400 samples, public
  local load and WebGPU encoder + WASM decoder/joint, unchanged fp32 artifacts.
  Receipt `tools/data/results/orukeet/auto-browser-controller-timing-long-2026-10-08-02.json`:
  75 inference calls, 76 updates, 73 nonzero origins, one no-inference finalize.
  Direct inspection compared all 2,525 recorded timing items against local
  values plus exactly `startFrame/sampleRate`; canonical and partial both
  matched, no mapping/time/payload mismatches. This is coordinate transfer,
  NOT original-engine timestamp or stage numerical parity.
- Two regular full-clip runs still match all 722 original-engine IDs and exact
  text. Effective PCM identity, served-code binding, model disposal, controller
  reset and browser closure PASS. Overall receipt remains FAIL because the
  controller's accumulated text differs from the full-clip oracle. First
  mismatch in run 01 is character 73 (`LibriVox` versus `Librivox`); differences
  are not limited to casing. No normalization or acceptance change was made.
- Receipt 02 SHA-256:
  `d72b51a26dae4da17813b35557713c73f7895c9e1f71f4788728d433a27dc825`.
  Run 01 is preserved: it exposed the finalize-only indexing defect in the
  initial diagnostic, not a production timestamp failure.

Reproduce from repository root with the already-running local demo:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/results/orukeet/librivox-repeat8-16k-pcm16-2026-10-02.wav --oracle tools/data/results/orukeet/official-long-repeat8-summary-2026-10-02.json --expected-pcm-sha256 caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46 --verify-controller --output tools/data/results/orukeet/FRESH-controller-timing.json --timeout-ms 600000
```

Two sequential bounded browser runs were used; no model/export/download/install
or sibling edits. Receipts are local evidence, not tracked acceptance assets.
Next: make timestamp receipt comparisons a tested automatic runner gate, and
localize long-text differences using original-engine captures of identical
window PCM. Changed window context is not equivalent to full-file decoding;
do not label every mismatch a merger/decoder bug or waive the full-text gate.
Existing feature numerical FAIL and broader quality/precision gates stay open.
