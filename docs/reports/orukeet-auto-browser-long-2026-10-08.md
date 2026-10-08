# C54 — Automatic Chrome Orukeet long-input control

Status: PARTIAL PASS, G1/G5/G6. The active whole-product goal remains unfinished.
This report supports the existing PRODUCT_TASKS C54 record; it is not a new queue.

## Verified scope

The existing local-folder demo now automates real transcription, not merely
model loading/warm-up. A fresh installed Chrome 154.0.8037.98 context ran our
own packed fp32 encoder plus decoder/joint through the library, on NVIDIA
Blackwell WebGPU encoder + WASM decoder/joint with JS features. No secondary
decoder, model download, runtime install or tolerance change.

Official original: `oruk/orukeet`, revision
`b59c13a733fce5cf193230d0fa317ee7f145a108`, NeMo 3.0.0/Torch 2.6.0 CPU oracle.
Artifact digests are in the receipt and existing `oruk-orukeet.port.json`.
Local ORT package is 1.29.0; actual delivered dependency/runtime bodies, not
only that version string, are recorded in `served_code` (287 deliveries).
This tests a source-checkout consumer, not an installed npm package.

Fixture: `librivox-repeat8-16k-pcm16-2026-10-02.wav`, 149.7125 s,
2,395,400 valid mono samples at 16 kHz, no padding/resampling.
Exact effective f32le PCM SHA-256:
`caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46`.
It agrees with the existing original-engine long-form handoff; the WAV hash
alone was not used to establish effective input identity.

Final receipt: `tools/data/results/orukeet/auto-browser-long-repeat8-2026-10-08-06.json`.
SHA-256: `deaa5c01d509bdf6e2f8cb32d3619f78253b8c012e48ca23227651711f2b81c6`.
Runner source SHA-256:
`4602fd146173857d84dc142ad326a1c92516aeae93789108a8ff8c791679f288`.

| Check | Result |
| --- | --- |
| Two independent transcriptions in the loaded session | 722/722 IDs and full text exact each time |
| Effective PCM / valid length / rate / channels / padding | Exact reference hash, metadata validated |
| Component backends | WebGPU encoder; WASM decoder and joint; JS features |
| Encoder frame count / decode iterations | 1872 / 806 on both repeats |
| Delivered code/runtime / runner identity | Bound; capture errors absent |
| Model disposal / own Chrome closure | Awaited, successful |
| Transcription time / RTFx, first then second | 9193.86 ms / 16.284; 6581.665 ms / 22.7469 |
| Model loading | 19628.935 ms, separate from transcription |

These are single-fixture observed timings, not a matched performance improvement
claim, peak VRAM/RSS measurement or proof of GPU resource reclamation.

## Harness repairs and retained failures

- The earlier JFK flow passed output control twice with 38 IDs. Default canonical
  segment detail omitted token objects; supported `canonical+native` exposes
  native debug token IDs and canonical metrics without changing inference math.
- Long `-01` passed token/text output, but did not prove identical effective PCM.
- `-02` retained FAIL: AudioContext yielded PCM hash `226b70ff...`, not the
  official `caf86c6b...`; post-disposal inspector response capture also failed.
  No decoder or feature change was justified by matching text on different PCM.
- `-03-aborted` is explicitly a supervisor startup failure record: Vite could
  not resolve a bare package import. Only the identity-verified owned Python
  process and its Chrome descendants were stopped; the source-checkout demo
  now imports the public browser/media entrypoint source, like its loader.
- `-04` verified same PCM and two exact outputs but retained FAIL because ORT
  worker responses were unavailable/evicted from Chrome's inspector cache.
- `-05` captured all runtime bodies correctly but retained a false-negative
  code gate: Vite legitimately compiled type-only TS modules to empty ESM.
  The corrected gate binds those empty bytes and still requires nonempty demo
  entrypoint and ORT WASM; a focused regression covers both directions.
- `-06` passes unchanged original-output gates plus effective PCM, code binding,
  requested repeat count/component backends, and cleanup. Without an explicit
  reference PCM digest a run is visibly `PCM_UNBOUND`, not same-input parity.

Nine auto-runner regressions, nine browser-helper tests and 14 local-Parakeet
helper tests pass. Python compilation, JS syntax and targeted diff checks pass.
Independent read-only checks found no confirmed remaining issue; actual inference
is evidenced by the fresh receipt, not those mocked tests.

## Reproduction and open delivery gates

Start the existing demo at loopback port 8767, then use the exact long-control
command in `examples/demo/README.md`, with a fresh output filename. Installed
Python Playwright/Chrome are prerequisites. Original inputs and receipts are
never overwritten. Test commands:

```powershell
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
python -m unittest discover -s tests/smoke -p test_browser_demo_smoke.py
python -m unittest discover -s tests/smoke -p test_local_parakeet_smoke.py
```

Production feature conformance remains FAIL/deferred at unchanged gates. This
does not promote Orukeet overall or authorize fp16. Stage numerics, streaming
controller behavior, cancellation/reuse under interruption, broader quality,
held-out human-label evaluation, peak memory and soak still need their own gates.
The manifest's old companion warm-up case still proves warm-up only; it is not
silently upgraded by this standalone run. Next independent user journey: C14
classic `browser-demo` fixture file transcription via its actual app UI, then
separate app-level cancellation/disposal, preserving the Orukeet numerical gaps.
