# Audio source evidence

Run core checks from the speech-recognition checkout:

```powershell
npm run build
npm run typecheck
node node_modules/vitest/vitest.mjs run --maxWorkers=2
node tests/smoke/browser-root-import-graph.mjs
C:/Python311/python.exe docs/reports/evidence/audio-source-2026-10-25/browser-audio-source.py --root .
```

The browser runner needs Playwright and Chrome. It serves this checkout on an
ephemeral localhost port, uses the built SDK, generates a deterministic WAV,
and intercepts the built-in loader module with a test double. Browser audio
decode/resampling and resource cleanup are real; model loading and inference
are mocked. No model weights are requested. See `browser-audio-source.json`.

Core test logs preserve the first failed root-export snapshot and the final
passing run. The historical export baseline was not edited; the test explicitly
allows the new API function while retaining exact equality for all exports.

Framework checks run in sibling `frameworks`: `npm run typecheck`, `npm test`
and `npm run build:packages`. Typecheck includes the adapter tests so capability
assertions are compiled, rather than relying on Vitest's runtime-only execution.

`browser-request-selection.json` is a separate request-selection/page-smoke
receipt. Its model requests were intentionally aborted; fetch errors do not
prove inference failure. The runner remains at
`frameworks/tmp/verify-audio-source-pages.py` and uses the unified suite at port
4190. No model/backend or real-weight acceptance is implied by page rendering.
