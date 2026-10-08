# C60 — Two isolated FFT product-rounding controls

Status: MEASURED / rejected as a feature-parity fix. Tracks G1/G6.
No production change, relaxed tolerance, model promotion or model-queue change.

The existing boundary diagnostic measured two previously unrecorded one-factor
settings on the original-engine captured JFK waveform and actual STFT. Both
validated exact input frame bytes and actual STFT reproduction. Same current
diagnostic, compiled processor, helper and reference capture hashes apply to
both. The tracked compact record is
`tools/data/results/orukeet/fft-product-rounding-handoff-2026-10-08.json`.

| Packed f32 product setting | Feature failures / 140928 | Max absolute error |
| --- | ---: | ---: |
| Historical separate/separate | 215 | 0.000307888 |
| Butterfly single-round only | 179 | 0.000362605 |
| Recombination single-round only | 220 | 0.000307888 |
| Unchanged aligned-complex control, both new runs | 121 | 0.000252336 |
| Unchanged production, both new runs | 313 | 0.000342369 |

Feature tolerances remain atol=1e-5, rtol=1e-4. Neither new candidate passes.
The first reduces mismatch count versus the historical packed control but
increases worst error and still does worse than the unchanged aligned control.
The second has more feature failures even though power maximum error decreases.
Complex and power tensors both pass their existing tolerant comparisons;
that cannot substitute for the failing downstream feature gate.

Historical comparison is not represented as a fresh paired A/B: its diagnostic
and capture container differ. We verified identical official float32 frame
hash, raw-mel hash and compiled processor hash, and identical unchanged
production/aligned feature metrics. Full provenance is in the compact handoff.

This diagnostic uses binary64 product sums with a final float32 rounding,
not hardware FMA emulation. It cannot identify the original CPU FFT algorithm
or justify fixture-specific production corrections. No decoder/export/model
weights were modified, downloaded or loaded.

## Execution and retained evidence

Budget: exactly two sequential cases, completed in under five minutes total
compute window; no third combination or full ASR rerun. Two gzip receipts total
206,674,827 bytes and remain local/untracked to avoid adding large experimental
blobs to Git. Their SHA-256 hashes and exact results are tracked in the handoff.
Both CLI exit codes are 1, not a silently accepted diagnostic failure.
The existing NeMo-overlay environment only supplies the original processor;
using its directory does not resume Nemotron work.

98 targeted tests pass: 81 feature-probe, 8 actual-STFT, 3 captured-PCM, 6 packed
FFT tests. They verify the diagnostic; they do not turn its failed result into
model acceptance. No full regression/performance/browser test is claimed for
these diagnostic-only calculations.

Reproduce each with the existing tool:

```powershell
tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/compare_js_feature_boundaries.py --audio tools/data/fixtures/audio/jfk-short.wav --source-config models/orukeet/onnx/fp32-packed-2026-10-01/source-config.json --official-capture tools/data/results/orukeet/official-jfk-stft-layout-stages-2026-10-02.json.gz --sample-id jfk-short --captured-pcm --require-actual-stft --packed-butterfly-products single-round --packed-recombination-products separate --summary-only --output tools/data/results/orukeet/fft-butterfly-new.json.gz
```

Second case swaps only the two named settings to `separate` / `single-round`
and uses a fresh output path. Existing outputs are protected.

## Next boundary

Stop this product-rounding branch; do not combine the knobs or patch the shared
processor to chase fewer mismatches. A next numerical experiment must isolate
a distinct actual production complex/power boundary on exact captured frames,
without repeating existing packed-f32/thread/window/normalizer controls.
Independent agreed library/example work can continue alongside that open gate.
