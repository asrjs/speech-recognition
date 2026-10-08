# C77 — Fixture-aware Chrome oracle run from the port manifest

Status: G1/G5/G6 output-smoke integration PASS; full model acceptance incomplete.

## User-visible result and boundaries

One existing manifest command now opens installed Chrome, loads local Orukeet
fp32 graphs, transcribes the declared JFK fixture twice, compares original-engine
full token/text output and effective PCM, awaits model/browser cleanup and saves
distinct parent/child receipts. It invokes the existing library-backed runner;
there is no new inference or decoder implementation.

`port_manifest.py` owns digest checks, dispatch and output safety. The explicit
`browser-oracle` kind selects a fixed registered adapter; unknown adapters fail.
`reference/orukeet/browser_manifest.py` owns this model's fp32/NeMo TDT layout,
oracle shape, precision/backend, repeat and result checks. No core runtime,
model math, tolerance, integer precision or artifact changed.

The old `runner: browser` remains companion **warm-up only** with
`comparesOracle: false`. New `PASS_BROWSER_ORACLE_MATCH` is same-PCM repeated
token/text browser smoke, not feature/encoder/state numerical acceptance,
fp16 promotion, native streaming, human-label quality, package-consumer or soak
acceptance. Orukeet's numerical feature gap remains open.

## Identity and acceptance

Before execution, the manifest binds original weights, five declared model
files, fixture/oracle, native runtime configuration and browser code contract.
The adapter requires exact receipt identity for the four actually consumed
graph/tokenizer/sidecar files. Source/native config remain explicitly auxiliary,
not falsely described as browser-consumed. The fixture stores original effective
PCM SHA separately from source WAV SHA. Oracle metadata must match model/audio.

The existing runner is pinned by SHA; every HTML/JS/TS/WASM response is captured
at the fulfilled-byte boundary. The code contract pins the complete path set,
sizes and hashes; repeated deliveries must each match. HTML inline execution is
covered, not only external modules. The adapter records its own loaded/current
SHA and the exact child receipt digest. These are local delivered-byte snapshot
identities, not general source-to-build equivalence or portable build hashes.
Changes to source/dependencies/server output must fail this contract until a
reviewed, evidence-backed refresh; do not auto-update a baseline to hide failure.

Final receipt: `tools/data/results/orukeet/manifest-chrome-oracle-jfk-2026-10-08-03.json`
SHA: `a04c60bb0254533b8b594f98aeedaa024954f88a952ee650f63952dd44d2cb31`.
Child SHA: `628cfb05c2630caf414bc9f7381623d7d574a6b1b3e9e2dbec49d0bcba7d66a6`.
Adapter SHA: `179e99ab2100817bbed8f9c1684faae7018ce5f085a0acd78c416dc7cb1c9735`.
Runner SHA: `4e1f3af766fcf1108395693572cd3364cd77f6a1133bb6c94674911be7c1cff8`.
Code-contract SHA: `9c736db33165dfcf01accf3c547f1c0bbdeb4efa95232c5a41d9fa8cb3d3d8d9`.

Chrome 154.0.8037.98, WebGPU encoder + WASM decoder/joint, JS features,
fp32, two exact 38-token/text results, 288 deliveries / 281 unique code paths.
All 19 adapter acceptance checks passed, including disposal/browser closure.
The existing loopback server was reused without modification or termination.
TLS verification settings were secure and child-local; no global change.

Three real integration runs are retained: `-01` established dispatch/output,
`-02` added expected external-code binding, `-03` also bound HTML. Earlier PASS
labels have these narrower scopes; do not substitute them for the final gate.
No new weight/download/export or numerical sweep was run.

## Reproduction and regressions

Start the reviewed demo on 8767 per `examples/demo/README.md`, then run from root
with a fresh output filename:

```powershell
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-chrome-fresh-run.json
python -m unittest discover -s tests -p test_port_manifest.py
python -m unittest discover -s tests -p test_orukeet_browser_manifest.py
python -m unittest discover -s tests/smoke -p test_orukeet_auto_browser.py
```

24 manifest tests, five adapter tests (21 mismatch controls plus invalid-run and
contract preflight controls), and 19 existing-runner tests PASS: 48 total.
Python compilation, JSON parsing and targeted diff checks PASS. Independent
static review caught external-code and HTML binding omissions; both were fixed
and re-reviewed APPROVE before final handoff. Simulated receipt tests are not
the real-GPU evidence: the final receipt above provides that bounded evidence.

Next: extend declared fixtures/backend/lifecycle coordinates when their own
required evidence is available; keep long windowed quality and production
feature numerical failures separate. Do not copy this adapter's decoding/layout
rules to another model or call this a generic accepted model pipeline.
