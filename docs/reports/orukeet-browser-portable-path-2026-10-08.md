# C59 — Checkout-independent Orukeet browser runtime path

Status: PASS for the bounded source-checkout dev workflow, G3/G5/G6.
The active whole-product goal and model numerical acceptance remain unfinished.

## Change and verified scope

The example no longer embeds `N:/github/asrjs/speech-recognition` in its
ORT WASM prefix or repository filesystem allow-list. Vite derives both from
its config location, URL-encodes path segments and supplies the runtime prefix
through `import.meta.env.ASRJS_ORT_WASM_BASE`. Core API and inference math did
not change. Existing Canary access to `N:/models` remains explicitly scoped.

Four Node tests cover relocated checkout paths, spaces, URL hash/query characters,
the injected prefix and exact allow-list. Eleven existing automatic browser
runner tests pass. JavaScript syntax and targeted diff checks pass.

Fresh actual Chrome 154.0.8037.98 receipt:
`tools/data/results/orukeet/auto-browser-portable-path-2026-10-08-02.json`.
Own packed fp32 Orukeet, WebGPU encoder + WASM decoder/joint, JS features;
149.7125-second mono 16 kHz WAV, 2,395,400 samples, no resampling or padding.
Both normal runs match all 722 official token IDs and the full 1,899-character
text. Effective float32 PCM matches the original-engine digest
`caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46`.
Delivered code/runtime bytes and runner identity are bound; page errors and
blocked requests are empty. Disposal is awaited and Chrome closure succeeds.
Loading observed 16,617 ms; this is not a matched speed improvement claim.

## Retained failures and limits

Receipt `...-01.json` is retained FAIL: a first attempt used a Vite global
define, but this HTML is served from `public/` without Vite's client injection.
Vite's actual dev define plugin skips module replacement; `import.meta.env`
is injected into the module by import analysis instead. The failed source
change was reverted before the second attempt and its receipt was not replaced.

An in-process Vite transform test passed its assertions but did not exit,
including a second isolated watcher/optimizer-disabled attempt. Those test
edits were reverted, and only the identified owned test processes were stopped.
The final test suite uses pure config/path checks; actual environment injection
is exercised by the source-bound real Chrome receipt, not claimed from the
non-exiting unit test. No unrelated server or OMP process was stopped.

This does not establish a production build/deployment recipe, a moved-checkout
real-browser run, POSIX browser execution, or an installed-package consumer.
The fixed Canary fixture root and repository-level installed ORT dependency
remain requirements of their respective workflows. Feature numerics,
streaming, fp16, held-out quality and memory/soak gates remain unchanged.

## Reproduction

Use the server and long same-PCM commands in `examples/demo/README.md` with a
fresh output path. Validation:

```powershell
node --check examples/demo/vite.config.js
node --check examples/demo/src/orukeet-smoke.js
node --test tests/smoke/demo-paths.test.mjs
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
```

Next: keep using this recovered automatic browser harness for affected changes;
continue a new, evidence-backed FFT experiment for the open feature boundary.
Classic sibling app cancel/unload UI still needs separate write authorization.
