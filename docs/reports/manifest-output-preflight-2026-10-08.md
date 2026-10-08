# C76 — Manifest runner output protection before execution

Status: bounded G6 safety PASS; model numerical/backend gates unchanged.

## Outcome and baseline

The one-command port harness previously executed native/browser commands before
discovering an existing final receipt at exclusive write time. Browser cases
also shared `{output}.browser.json`; selecting multiple browser cases could
reuse a stale companion receipt or fail only after expensive work. Exclusive
writing alone did not forbid a new output inside the model artifact directory.

`port_manifest.py` now reuses the model-neutral `require_new_outputs` helper
after manifest digest binding and case selection, before the first subprocess.
Protected inputs include the manifest, bound original weights, graph/sidecar
files, runtime configuration, fixtures and oracles. The local model root is a
protected directory. Selected browser companion destinations and the final
receipt are checked together. Multiple browser cases with the current shared
suffix fail explicitly; select one case per invocation with a fresh output.
List mode does not create outputs or launch processes. Exclusive final writes
remain in place to reject a competing file creation after preflight.
After checking original paths (including dangling symlinks), caller-relative
output is resolved once for child command placeholders and the final write;
the subprocess's repository-root working directory cannot relocate its receipt.

No inference implementation, family math, tolerance, manifest schema or browser
acceptance criterion changed. This protects CLI orchestration, not arbitrary
commands embedded in a trusted manifest or direct internal `run_case` calls.

## Verification

- New negative controls reached the subprocess before the fix (seven failing
  subcases), proving preflight was missing. Subprocess mocks raise immediately
  so tests cannot accidentally run inference or overwrite source material.
- 24 manifest tests PASS, including existing identity/comparison controls,
  existing parent/browser receipts, protected inputs/directories, duplicate
  browser destinations, listing, non-root caller path identity, and a real tiny
  child-process failure receipt. Independent review found the cwd mismatch;
  its new regression failed before that fix and passed afterward.
- 18 existing automatic Orukeet browser runner regressions PASS.
- Python compilation and targeted diff checks PASS.

```powershell
python -m py_compile tools/model-debugging/scripts/port_manifest.py tests/test_port_manifest.py
python -m unittest discover -s tests -p test_port_manifest.py
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
```

No new browser/GPU run was needed for path-preflight acceptance. The immediately
preceding real Chrome receipt is linked in
[C75's report](chrome-auto-browser-resume-2026-10-08.md).

## Remaining work

The existing manifest `runner: browser` still means companion warm-up, not
fixture/oracle acceptance. Orukeet's full automatic inference runner remains
separate and model-specific. Connecting a fixture-aware browser adapter must
retain that distinction and independently bind artifacts/audio/oracle, actual
execution, code provenance, lifecycle and exact repeated outputs. This patch
does not claim that integration or full Orukeet numerical acceptance is done.
