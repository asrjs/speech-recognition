# Model loading evidence

Scope: catalog-to-executor precision selection, fallback event ordering, warning
state lifecycle, consumer compatibility and demo builds. This is not a new
real-weight inference receipt.

Core commands, from `speech-recognition`:

```powershell
npm run build
npm run typecheck
node node_modules/vitest/vitest.mjs run --maxWorkers=2
node tests/smoke/browser-root-import-graph.mjs
```

From sibling `demos`: `npm test` and `npm run build:suite`.
From sibling `frameworks`: `npm run build:packages`, `npm run typecheck`,
`npm test` and `node scripts/build-suite.mjs --build-frameworks`.

The browser runner was executed at
`frameworks/tmp/verify-model-load-requests.py`. The archived copy assumes that
location. To reproduce, copy it back to that path, start the unified suite at
port 4190 and run it from the framework workspace with Python and Playwright.
It aborts all Hugging Face asset traffic, mocks directory listing responses,
and uses fresh browser contexts. `Failed to fetch` is intentional evidence of
the blocked download, not an inference result. The six checks verify the first
requested encoder only; decoder selection and fallback behavior have separate
core regressions with mocked sessions. Seven-page smoke establishes loading and
main-landmark rendering, not full UI or model acceptance.

The new SDK archive and its source/dist hash manifest are in `demos/vendor`.
The parent checked every installed dist file against that manifest: 1711 matched.
No real model weights were downloaded by this review.
