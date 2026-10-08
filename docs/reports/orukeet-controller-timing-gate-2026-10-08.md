# C65 — Automatic controller timing acceptance gate

Status: automatic coordinate-transfer gate PASS; long accumulated-text gate
remains FAIL. G4/G6, following C64. No model acceptance or numerical waiver.

The existing Chrome runner now always checks timing when `--verify-controller`
is requested, recording `controller_timing_valid` and its narrow scope. Overall
PASS requires it in addition to the unchanged full token/text/oracle, coverage,
backend, reset and cleanup gates. Unrequested scope is `SKIP_NOT_REQUESTED`.

The validator checks ordered, complete inference links, exact integer frame
ranges and source/request/update association; unchanged words/segments/sentences/
tokens; local timestamps plus one absolute window offset; matching partial
payloads; finalization-only null detail; and actual observed timed fields.
`--require-controller-nonzero-origin` additionally requires observed timed detail
for all four fields at nonzero starts, and cannot be used without the controller.
No model-specific runtime code or decoder was changed.

Validation:

- Python compile checks and 18 runner tests PASS, including 20 corruption
  mutations (missing records, wrong indices, double offsets, untimed-token
  fabrication, NaN/bool timestamps, altered IDs, bool/float window frames),
  final-only behavior, empty nonzero detail and CLI dependency.
- Initial test fixture shared object references and accidentally shifted one
  item three times; printed fixture evidence localized this diagnostic defect,
  corrected with independent per-field copies. Real-record comparison then
  exposed JSON `4` versus Python `4.0`; a failing regression preceded the
  corrected comparator. Only equivalent JSON numeric spelling is normalized;
  exact value comparison and bool distinction remain, no tolerance was widened.
- Independent review found frame bool/int equality; RED mutations reproduced
  it and strict endpoint types fixed it. Final review approved.
- Fresh installed Chrome 154.0.8037.98 JFK receipt:
  `tools/data/results/orukeet/auto-browser-controller-timing-gated-jfk-2026-10-08-01.json`.
  Overall PASS; controller/timing, exact official PCM and two 38-token/text
  repeats, served code, reset/disposal/browser closure PASS. All short-window
  origins remain zero; this is not nonzero-origin coverage.
- Fresh long automatic receipt:
  `tools/data/results/orukeet/auto-browser-controller-timing-gated-long-2026-10-08-01.json`.
  `controller_timing_valid=true`, nonzero requirement enabled; 75 inferences,
  76 updates, two exact 722-token/text full-clip runs and PCM/cleanup PASS.
  Overall FAIL because accumulated windowed text is not the full-file oracle.
  This long receipt precedes only the stricter frame-type guard; the final
  comparator was also checked against the preserved long timing records.

Commands: `python -m py_compile tests/smoke/orukeet-auto-browser.py
tests/smoke/test_orukeet_auto_browser.py`; `python -m unittest discover -s
tests/smoke -p test_orukeet_auto_browser.py`. Use the existing demo README's
browser recipes with fresh output names; append
`--require-controller-nonzero-origin` to its long controller command.

Next diagnostic (not executed): compare original-engine output on identical
window PCM, starting with receipt request 3, frames 51,200–128,000 at 16 kHz.
The source is PCM16 mono WAV; lossless sample slicing can reproduce its f32le
digest `31497b359206a88e148279be1e8f7cd9b0373eab66b29014b67347b4d7b28868`.
The existing `tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py`
accepts `--audio`, `--restore-path`, `--device cpu`, `--cpu-threads 4` and
`--timestamps`; it does not accept a PCM-array/window parameter. Cached original
`models/orukeet/source/orukeet-v0.1.0.nemo` and the established NeMo interpreter
both exist. Re-read the playbook and coordinate CPU/RAM before one plain capture;
do not start the three-pass stage capture or reinstall the overlay environment.
This would distinguish changed window recognition from merging, not waive the
existing long-text or feature numerical gates.
