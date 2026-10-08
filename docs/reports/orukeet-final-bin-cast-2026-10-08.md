# C74 — Production FFT output-component cast control

Status: diagnostic tested; numerical feature hypothesis rejected on JFK.
G1/G6, not production/model acceptance. No `src` or model weights changed.

The existing boundary probe gains opt-in `--production-final-bin-f32`.
Here “final bin” means the final complex components of **all** FFT output bins,
not only the Nyquist bin. Only those components are cast with Math.fround before
the unchanged binary64 squares/sum and existing float32 power-buffer assignment.
No FFT butterfly, scratch, preemphasis, window, mel or normalization changes.
No magnitude square-root/square roundtrip or reference-spectrum injection.

The isolated candidate retains pre-cast complex captures. The probe checks exact
production/candidate input-frame bytes, pre-cast real/imag and filterbank/bounds;
independent float32-component -> float64-power -> float32-assignment computation
must reproduce candidate power bytes. The candidate module, power, raw features,
normalized features, compiled processor, official capture and effective PCM are
SHA-bound. Optional execution does not mutate production or default computations.

## Observed first case

Original effective JFK PCM and actual official STFT reproduction PASS. Unchanged
tolerances remain atol=1e-5 / rtol=1e-4. Power and valid raw-mel tolerant checks
PASS; raw values are not bit-exact (68,056 differences).

| Feature control | Failures / 140,928 | Max absolute error | Mean absolute error | First index |
| --- | ---: | ---: | ---: | ---: |
| Unmodified production | 313 | 0.00034236907959 | 0.00000122680998 | 45 |
| Final-component f32 cast | 313 | 0.00034236907959 | 0.00000122697847 | 45 |

This rejects output-component casting as a feature-parity fix on this case; it
does not establish why the complete FFT/reduction chain differs or justify a
production patch. Exact input preservation is not feature/model acceptance.

## Validation and execution

Two new regressions RED before implementation; final 84 probe tests PASS, plus
8 actual-STFT and 3 captured-PCM tests PASS. Python syntax and scoped diff checks
PASS. Read-only independent review found no correctness issue. Tests are
weight-free and do not certify inference, held-out quality or memory behavior.

Budget: one numerical setting, five-minute compute window, no model/GPU/export.
First case completed exit 1 with failed comparisons, not hidden as PASS. Its raw
gzip receipt remains local/untracked:
`tools/data/results/orukeet/production-final-bin-f32-jfk-2026-10-08.json.gz`
(103,432,451 bytes), SHA256
`72a354550b82d4b956fa762a6556a7a98a15f077f4836347ee15b22c9ed8ad5e`.
Its diagnostic SHA is `ddb3332c1bcd225c606c29d8ef5876d1b33e8dca50b6385f9577ad6ec908f6ff`.
After that run, the new raw-mel comparison row was corrected to use its nested
`tolerant` metric, matching the flat result-gate schema. It changes reporting,
not candidate arithmetic. A single same-setting verification completed against
the corrected reporter, exit 1. Candidate/baseline metrics and candidate source,
power/raw/features hashes are identical to the first case. All final comparison
rows have boolean `allclose`; final diagnostic SHA matches the executed source.
The first receipt is preserved, not overwritten. Final receipt:
`tools/data/results/orukeet/production-final-bin-f32-jfk-final-2026-10-08.json.gz`
(103,432,410 bytes), SHA256
`85873af7d1fd7daa86bbae48156e3166e6b1dee95f95591ce544eb1808467168`.
Final diagnostic SHA256:
`1b679b7c7b8633607e277f6ae6de93b5df4807e6edba13a861942758180a0047`.
Both large raw receipts are retained locally, not committed to Git.

Identities shared by both runs:

- Effective PCM f32le: `ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
- Official capture: `4328840597d82b6942e640c2d9a4f810ca6c589588fb833009d02d8b14178508`.
- Compiled processor: `067a902ceea1f260ce811f2bc04a7112e1a3a0e0cc17b99c6da9aa15a1a6a32f`.
- Isolated candidate module: `4e6e3abe9eb1271bfa1574562d073ec162dc5973b716463627887372bad7c700`.
- Identical upstream f64 frames: `63ffee9a21735a9c82c8352228ee61eb3b81d778862c495f050970fb3245b413`.

Reproduce with a new output path:

```powershell
tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/compare_js_feature_boundaries.py --audio tools/data/fixtures/audio/jfk-short.wav --source-config models/orukeet/onnx/fp32-packed-2026-10-01/source-config.json --official-capture tools/data/results/orukeet/official-jfk-stft-layout-stages-2026-10-02.json.gz --sample-id jfk-short --captured-pcm --require-actual-stft --production-final-bin-f32 --summary-only --output tools/data/results/orukeet/final-component-cast-new.json.gz
```

Using the existing NeMo-overlay interpreter does not resume Nemotron work or
change reference dependencies. No install/download/sibling write occurred.
Do not repeat this rejected cast or C60 FFT product-rounding sweep. Continue
independent product work or a distinct discriminating numerical hypothesis;
Orukeet feature/model acceptance, companion cancellation/unload and Redux
original-runtime prerequisites remain open.
