# Automatic Chrome model test recovery — 2026-10-08

## Recovered system and repair

The historical sibling `webgpu-agent-test` URL automation is preserved. The
tracked candidate-capable path is `tests/smoke/orukeet-auto-browser.py` and
`examples/demo/public/orukeet-smoke.html` (served by Vite at
`/orukeet-smoke.html`). It opens installed Chrome headlessly with
`?auto=1`, selects local model/audio files, runs actual library inference twice,
awaits model disposal and browser closure, and exclusively creates a fresh JSON
receipt. The manifest runner wraps this path with artifact/oracle/code checks;
it does not implement a second decoder. Existing server 8767 was reused, not
restarted or stopped. An attempted second Vite startup safely rejected the
occupied port; it did not create another server.

The current direct runner succeeds. The stale C81 delivered-code contract,
however, rejects the same new receipt with only `expected_code=false`; the
other 18 checks pass. The reviewed controller change `f65c14c1` is loaded by
this demo, even when optional controller verification is not requested.

A separate `browser-code-contract-c83.json` now pins the current responses.
C81 and earlier snapshots remain unchanged. The new snapshot has the same
281 unique paths, 288 deliveries and consistent duplicate bodies; exactly
seven response records change. Only controller.ts changes in the corresponding
source lineage; the other six source files are unchanged and their current
transformed responses include propagated Vite HMR timestamps. Old response
bodies are unavailable, so this is NOT a verified timestamp-only textual diff.
An independent read-only reviewer and the primary agent each fetched the seven
live loopback HTTPS responses and verified exact size/hash correspondence.
No numerical tolerance, inference code, original weights/graphs, PCM or oracle
was changed. Manifest identity now references the separately reviewed C83 pin.

New snapshot SHA-256:
`50a8ac2ab8bd234bf00e32ea43d1b3c415625604f039fb91dc421b30375d237f`.
Seed receipt: `tools/data/results/orukeet/auto-browser-user-recovery-2026-10-08-01.json`,
SHA-256 `bda29a651fdbe3d792306f9228e41946f4ad5bc45c3271d9e027387a8f56e48b`.
It records Chrome 154.0.8037.98, two exact 38-token/text JFK results, official
effective f32le PCM, fp32 WebGPU encoder + WASM decoder/joint, JS processor,
and successful awaited model/browser teardown.

## Reproduction

From this repository root, start the server only if it is not already running:

```powershell
node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1 --port 8767 --strictPort
```

In another terminal, choose a never-used receipt filename:

```powershell
$env:NODE_TLS_REJECT_UNAUTHORIZED = '1'
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-chrome-FRESH.json
```

The browser's explicit loopback development-certificate exception remains;
global TLS verification is not disabled. No new weights, dependencies or
browser binaries are downloaded. The manifest case's URL uses port 8767 and
the server's established HTTPS configuration. A changed server transform,
source or dependency must fail code identity until separately reviewed; do not
automatically refresh the pin to hide a failure.

Run `chrome-oracle-pure-silence-webgpu-hybrid` with another fresh output name
for the existing silence control. Direct runner CLI supports opt-in controller
and cancellation checks; those are separate acceptance scopes, not implicit
parts of these two manifest cases. Other model families need their own adapter,
not reuse of Orukeet's oracle/model assumptions.

## Validation and remaining scope

19 automatic-runner, 6 browser-adapter and 24 manifest Python tests PASS.
Direct Chrome same-PCM repeated oracle token/text smoke PASS. New snapshot is
independently reviewed. Fresh manifest-driven JFK run exits 0, parent PASS,
`PASS_BROWSER_ORACLE_MATCH` and all 19 checks true:
`tools/data/results/orukeet/manifest-chrome-user-recovery-2026-10-08-01.json`,
SHA-256 `e9d36026eb8e163b90e011af5e119f457917062c1f291571ab2ac725efbbb4ab`.
Child receipt SHA-256:
`25005b29deaf1edd25c67a49a29dd59fd12cfa21f29e47e1c9556e47b69cafdb`.
Both actual outputs have 38 exact oracle tokens/text; current code identity,
official effective PCM and awaited cleanup pass. This is a fresh run independent
of the snapshot seed, not acceptance by replaying its own seed receipt.

Fresh silence manifest run also exits 0, parent PASS and all 19 checks true:
`tools/data/results/orukeet/manifest-chrome-user-recovery-silence-2026-10-08-01.json`,
SHA-256 `f6d9c312eb639912d8b1094fb5ebbc6f34ffc346db6bff5ef68178fb06b25f11`.
Child receipt SHA-256:
`4bc76f12bddde54d6ed350752de6717c8199e5b6f5a5470166aa32b6910c05fd`.
Two explicit empty token/text outputs match the official same-PCM silence
oracle; no missing-output shortcut. Disposal and browser closure pass.

These are real-browser output smokes, not full model acceptance. Strict
production feature numerical parity remains unresolved; fp16, native streaming,
held-out quality, model-backed VAD and GPU memory/soak gates are not certified.
Protected Nemotron/Qwen changes and the earlier VAD smoke work are not included
in this repair.

## Fresh user-requested recheck

Rechecked the existing system without replacing its inference runner. The HTML
entrypoint is tracked in `examples/demo/public/`, not the demo root; Vite serves
it at `/orukeet-smoke.html`. HTTP 200 confirmed the existing 8767 server before
reuse. A second startup rejected the occupied port; no server was stopped.

- `auto-browser-user-recheck-2026-10-08-01.json`: PASS, two exact 38-token
  file-oracle outputs, official effective PCM, actual WebGPU encoder/WASM
  decoder/joint, controller/timing checks, two real fake-device manual capture
  sessions, exact capture-to-utterance delivery and same-PCM ASR replay.
  Each capture has 32,000 samples and 11 tokens; exact float32 sidecars were
  exclusively saved. Model/controller/capture resources and Chrome closed.
  Receipt SHA-256: `8ab8124b4f7f94374fd7b8b6a84894988e014c4792950a4728bb3480842be7f9`.
- `manifest-chrome-user-recheck-2026-10-08-01.json`: PASS, all 19 adapter
  checks including the unchanged C83 delivered-code contract. The subprocess
  explicitly enabled Node TLS verification; no global setting changed.
- 25 automatic-runner, 7 adapter and 24 manifest unit tests: PASS.

Receipts are under `tools/data/results/orukeet/`. The direct run inherited a
Node TLS-verification-disabled warning; the manifest subprocess did not.
Console favicon/HMR connection warnings remain recorded; no page errors or
blocked inference requests occurred. No new model download or dependency
installation. This recheck does not run a new original-engine oracle on the
saved capture PCM and does not certify speech-detect ASR, native streaming,
stage numerical parity or memory/soak. Use the C91 recipe for an independent
capture oracle, and retain these narrower acceptance boundaries.
