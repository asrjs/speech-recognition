# Handoff: automated Chrome runner (C94, paused)

User requested immediate handoff on 2026-10-08. Stop development until resumed.
The Codex goal is already `paused`; do not mark it complete. This report is the
latest resume overlay, not a replacement for `docs/AGENT_ENTRYPOINT.md`.

## Checkout and ownership

- Checkout: `N:/github/asrjs/speech-recognition`, main.
- HEAD: `afe80102c9b20e94b50b16e041e022e9b7f5f104`.
- No commit or staging was performed in this slice.
- Owned uncommitted work: `examples/demo/src/capture-controller.js`,
  `tests/smoke/orukeet-auto-browser.py`,
  `tests/smoke/test_orukeet_auto_browser.py`, and new
  `tests/capture-demo-built-runtime.test.ts`.
- Preserve all other dirty files, especially goal/task/entrypoint documents,
  Nemotron and Qwen work. Do not reset, clean or broadly stage this checkout.
- Prior C93 committed seven automatic utterance token/text comparisons against
  original Orukeet NeMo weights. Do not reuse those results for new capture PCM.

## System found

Current runner: `tests/smoke/orukeet-auto-browser.py`. It launches installed
Chrome with Playwright, opens the local demo URL, selects a local model directory,
performs inference and writes an exclusive fresh JSON receipt. The page is
`examples/demo/public/orukeet-smoke.html`, served at `/orukeet-smoke.html`.
This is an Orukeet adapter, not a universal model browser runner.

Generic orchestration: `tools/model-debugging/scripts/port_manifest.py`, with
`tools/model-debugging/ports/oruk-orukeet.port.json`, run
`chrome-oracle-jfk-webgpu-hybrid`.

Historical sibling runner:
`N:/github/asrjs/webgpu-agent-test/scripts/run-parakeet-tdt-webgpu.mjs` uses
`?auto=1`, `/__test_results__` and saved results; it is Parakeet TDT v2/v3-specific.
No sibling changes were made.

## Work in progress

Added opt-in `--capture-library dist`, requiring
`--verify-capture --capture-mode speech-detect`. The initial source-demo file
oracle is still a separate control. Built capture uses a separate blank page,
`dist/browser.js`, the actual default TEN worker and assets. The example helper
accepts a supplied public API and an external Worker observer rather than forcing
a source TS worker factory. Source default behavior remains available.

The runner records delivered code, rejects required built asset omissions and
core source leakage, records observer restoration, hashes required built inputs
before/after, and refuses saving unverified captured PCM. This is Vite-served
dist evidence, NOT installed/packed npm acceptance.

## Validation and exact failure

- TypeScript build: `node node_modules/typescript/bin/tsc -p tsconfig.json` PASS.
- Runtime asset copy: `node scripts/copy-runtime-assets.mjs` completed.
- Python runner targeted suite: 34 tests PASS.
- Vitest capture-built-runtime (4), Orukeet demo (13), windowed demo (5):
  22 tests PASS.
- Actual Chrome C94: **FAIL**, not acceptance.
- Receipt:
  `tools/data/results/orukeet/auto-browser-c94-dist-capture-2026-10-08-01.json`.
- Error: `Capture failed; refusing unverified PCM sidecars`.
- Capture error: `Error: Speech ASR missing output`; captured runs array empty.
- `capture_valid=false`, `capture_build_valid=false`.
- `browser_closed=true`, `modelDisposed=true`, `observerRestored=true`.
- Capture `disposed=false`: do not claim controller/worker cleanup was certified.
- No page errors or blocked network requests were recorded.
- Actual built default worker and TEN JS/WASM asset requests are present in the
  receipt, but the failure leaves final worker lifecycle markers absent.
- No saved verified capture PCM/original-engine comparisons for C94.
- Runner process session 43425 reached terminal exit 1. A Ctrl-C request at
  handoff returned its normal failure JSON (no KeyboardInterrupt evidence).
- Existing Vite server on 8767 was not started/owned here and was not stopped.
- Independent OS process enumeration was denied; do not infer from that check
  that all machine browser processes are absent.

Runner SHA256 now:
`8e945c0da77eff4110f7f59d39e81e18d706ccbf0289ca9725d1fd2702ee9b6f`.
Manifest runner pins have NOT been updated; reconcile only after review and
successful regression. Do not refresh acceptance code pins merely to get PASS.

## Resume in order

1. Read entrypoint, complete safety rules, relevant goal/task routing and this
   report. Reconcile live Git state. Read systematic-debugging skill before fixes.
2. Inspect the helper's `Speech ASR missing output` assertion, pending ASR queue,
   capture duration/waits and recorded diagnostics. Determine whether inference
   finishes later, capture yields no accepted utterances, or another error occurs.
   The cause is not established. Do not weaken the assertion or build gate.
3. Preserve C94-01 failure. Make only a discriminating minimal change after
   inspecting actual APIs. Re-run focused checks, then fresh real Chrome.
4. Run source capture regression too. Validate all lifecycle/code/PCM gates.
5. If capture succeeds, compare each NEW exact saved window PCM to official
   original weights; old C93 token evidence does not certify new capture input.
6. Review/update manifest runner hashes, run manifest checks, document limits,
   and commit ONLY owned verified changes. Keep whole-product/model parity open.

Fresh run (change output suffix; never overwrite failure receipts):

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --verify-capture --capture-mode speech-detect --capture-library dist --save-capture-pcm --output tools/data/results/orukeet/auto-browser-c94-dist-capture-2026-10-08-02.json
```

If the local server is absent, inspect configured HTTPS/cert settings first.
Existing startup command (repo root):
`node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1 --port 8767 --strictPort`.
No downloads, installs, uploads, push or model-order changes are authorized by
this handoff. Orukeet remains first; Parakeet TDT derivatives follow, CTC held
and Nemotron paused. Strict feature numerical parity remains unresolved.
