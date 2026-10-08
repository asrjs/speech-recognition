# C88 — Capture PCM across asynchronous model startup

Status: bounded real-browser capture/model integration PASS. Whole-product goal
and original-model numerical acceptance remain open.

## Failure and shared fix

The existing offline Chrome runner now supports `--scenario ten-vad-capture`.
It uses actual getUserMedia, AudioContext, AudioWorklet, public microphone
controller/default starter and packaged TEN WASM worker. The WAV device is fake;
only delivery of the real worker INIT message is delayed by 250 ms.

Before the fix, `ten-vad-capture-controller-2026-10-08-02.json` saved FAIL:
32,000 captured frames but only 114 model hops, with incorrect initial offsets.
The controller delivered PCM while the adapter was initializing; the adapter
could not process the prefix. The initial ring-reset hypothesis was not proven.

`src/runtime/browser-controller.ts` now copies and buffers the startup prefix
before capture acquisition resolves, replays it in order only after full starter
readiness, and clears it on stop/dispose/error. The buffer is bounded by configured
ring duration at the requested 16 kHz capture rate. Overflow aborts initialization
and fails start rather than silently dropping PCM. Decoder logic, thresholds,
rough-gate boundaries and model mathematics are unchanged. This is shared capture
lifecycle behavior, not a TEN-specific runtime workaround.

## Verified evidence

Receipts under `tools/data/results/harness/` are preserved, never overwritten:

- `ten-vad-capture-controller-2026-10-08-01.json`: normal initialization PASS.
- `ten-vad-capture-controller-2026-10-08-02.json`: controlled-delay pre-fix FAIL.
- `ten-vad-capture-controller-2026-10-08-03.json`: post-fix PASS before probe formatting.
- `ten-vad-capture-controller-2026-10-08-04.json`: final current-hash PASS.

Final Chrome 154.0.8037.98 receipt: two independent starts, each 32,000 frames
and 125 actual model hops with complete contiguous coverage; one rough-fallback
utterance per run with PCM exactly matching its actual captured slice. Each
session loads one worklet, stops capture once, closes context and ends owned
tracks. Actual DISPOSE acknowledgement and one worker termination pass; no page
errors or rejected network requests; owned Chrome closes. Browser processing can
change WAV PCM, so captured hashes and rough boundaries are not fixed oracles.
Independent read-only review verified final input and served-probe hashes.

The existing file-PCM controller scenario also re-passes against the new build:
`ten-vad-controller-c88-regression-2026-10-08-01.json`, with identical 687-hop
speech probabilities/buffer values on two starts, four exact PCM utterances per
speech run and 62 silent hops with no utterances after restart.

Five added regression tests cover owned clones/order, stop, dispose, bounded
overflow and initialization rejection. Three discriminating tests failed before
the production fix. Afterward 25 focused controller/disposal tests pass; full
Vitest suite: 201 files PASS, 7 SKIP; 1,675 tests PASS, 34 SKIP. Source/examples
TypeScript checks, build, focused ESLint, probe syntax and diff checks pass.

## Reproduce and scope

From the repository root, with existing local dependencies and installed Chrome:

```powershell
node node_modules/typescript/bin/tsc -p tsconfig.json
node scripts/copy-runtime-assets.mjs
python tools/browser-validation/capture-lifecycle-smoke.py --scenario ten-vad-capture --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --audio tests/fixtures/jfk2.en.wav --output tools/data/results/harness/ten-capture-FRESH.json
```

No downloads, server or browser installation. Fresh receipts hash source,
runner/probe, assets and delivered modules. Build first: hashes alone do not
prove source-to-build equivalence.

TEN remains diagnostics-only; rough gate determines segments. This does not
certify ASR, VAD accuracy, original-engine numerical parity, WebGPU, physical
microphone, permission-dialog race, cleanup-failure injection or memory soak.
For automatic URL/model load/ASR output use the separate existing
`tests/smoke/orukeet-auto-browser.py` documented in `tools/browser-validation/README.md`.
If delivered code changes, the reviewed Orukeet manifest's code snapshot requires
separate review before refresh; never bypass or automatically regenerate its pin.
C89 verifies that this demo does not deliver the microphone controller changed
here: the base C83 pin remains byte-identical and its independent oracle run passes.
See [the current controller verification](chrome-current-controller-2026-10-08.md).
