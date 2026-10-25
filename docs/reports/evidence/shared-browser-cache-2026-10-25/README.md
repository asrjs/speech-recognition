# Shared browser cache verification

From the sibling `frameworks` workspace, build the core, demos and unified suite
as described in its README. Serve the unified suite at 4190. In another terminal,
start the same built suite with `PORT=4191` for the negative origin control
(PowerShell: `$env:PORT='4191'; npm run preview:suite`).

Run from `frameworks/`:

```powershell
$env:ASRJS_TEST_PORT='4190'
python ../speech-recognition/docs/reports/evidence/shared-browser-cache-2026-10-25/browser-cache-proof.py
```

The test uses Playwright with installed Chrome and a fresh temporary browser
profile. It intercepts only its synthetic SDK/asset URLs, serves compiled core
modules, and checks real IndexedDB. The deterministic 64 KiB payload is not an
ASR model. It does not modify the user's browser profile or cache.

Saved result: one network fetch, six cache hits across seven app paths, exact
SHA-256 equality and one fixture row; legacy ASR/FireRed reads did not copy rows.
The different-port control stayed isolated. Existing failure/timeout logs are
included alongside the final passing full suite. No timeout or transcript oracle
was weakened. The auxiliary test server can be stopped after the run.
