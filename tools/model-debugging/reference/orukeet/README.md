# Orukeet official-source reference and port handoff

Run from the `speech-recognition` repository root. Original weights live under
ignored `models/orukeet/source/`; small provenance/summary receipts live under
`tools/data/results/orukeet/`. Full compressed numerical traces are ignored
local artifacts and reproducible with the command below.
Commands referencing historical receipt names document past runs. Reuse verified
local assets and choose fresh output names when reproducing; do not overwrite
original weights, captures or existing receipts.

Orukeet retains 627M parameters; it is a same-size Parakeet TDT v3 fine-tune,
not a smaller student. Original checkpoint size is not deployment graph size.
Source: [official model card](https://huggingface.co/oruk/orukeet).

## 1. Acquire and audit original assets

Reuse the existing local NeMo 3.0 environment. Do not change the user's global
Python/CUDA setup. Its Torch 2.6.0+cu118 does not support this machine's sm120
GPU, so the official numerical reference here runs on CPU. This does not
describe browser WebGPU support.

```powershell
$orukeetPython = 'tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe'
& $orukeetPython tools/model-debugging/reference/orukeet/acquire_source.py `
  --download-weights `
  --output tools/data/results/orukeet/source-provenance-2026-09-30.json
& $orukeetPython tools/model-debugging/reference/orukeet/audit_source.py `
  --v3-vocab N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx/vocab.txt `
  --output tools/data/results/orukeet/source-contract-2026-09-30.json
```

The downloader pins revision `b59c13a733fce5cf193230d0fa317ee7f145a108` and
verifies original `.nemo` SHA-256
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
The audit reads config and SentencePiece from that checkpoint, not merely
converted Hugging Face metadata. All 8,193 indexed tokens match local v3,
including blank 8192; configuration checks pass. This is contract compatibility,
not numerical/export/browser parity.

## 2. Capture the official decoder's own numerical execution

```powershell
& $orukeetPython tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py `
  --restore-path models/orukeet/source/orukeet-v0.1.0.nemo `
  --model oruk/orukeet `
  --model-revision b59c13a733fce5cf193230d0fa317ee7f145a108 `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --device cpu --cpu-threads 4 --sample-id jfk-short `
  --stage-output tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz `
  --output tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json
```

The existing official inspector now supports optional CPU greedy-batch TDT
capture. It hooks real features/encoder outputs and wraps NeMo's actual
`predict`, `project_encoder`, `project_prednet`, `joint_after_projection` calls;
NeMo retains ownership of search. Joint inputs are its projected boundaries,
not raw encoder frames. Final tokens, durations and frame timestamps remain
separate outputs. Every stage has full values, shape and dtype in processing
order, with per-call inputs/states and audio/source SHA-256 provenance.

The historical JFK run produced 464 stages, 40 predictor calls, 46 joint calls and 38 emitted
tokens. Baseline/captured/after transcription controls passed exact tokens/text;
baseline frames and post-capture durations also match. Limits and missing
stages fail instead of silently truncating the oracle. The capture is diagnostic,
not a speed benchmark. The 2026-10-01 speech/silence continuation produced
1,120 stages (71 predictor / 182 joint calls) with the same exact controls;
see `official-speech-silence-stage-summary-2026-10-01.json`. Pure silence and
long-audio remain open. Partial native export and public-library output evidence
are recorded below; numerical feature and Orukeet browser acceptance remain open.

The Windows restore stall was Numba probing an inaccessible environment cache
through Librosa's lazy JIT import. The inspector uses `.cache/nemo-reference`,
sets `NUMBA_CACHE_DIR` and calls `numba.config.reload_config()` because NeMo
already imported Numba. CPU-thread control, progress and timeout stack dumps
make this diagnosable without altering the installed environment.

Instrumentation simulation tests require only Torch and Python's standard
`unittest`, not an additional pytest install:

```powershell
& $orukeetPython tools/model-debugging/reference/hf-parakeet-port/test_nemo_tdt_stage_capture.py -v
```

## 3. Own fp32 export produced; numerical parity next

Follow `docs/PARAKEET_VARIANT_TASKS.md`. The reusable family exporter is now
`tools/nemo-tdt-onnx-export/export_nemo_fp32.py`; see its README for the exact
command. It restored the pinned original checkpoint and exported fp32 encoder
plus combined decoder/joint through NeMo's own preparation (no-extra-SOS,
raw joint scores). Both graph contracts and external tensor ranges pass audit.
Assets are in ignored `models/orukeet/onnx/fp32-nemo-2026-10-01/` with a detailed
`export-receipt.json`; the small tracked receipt is
`tools/data/results/orukeet/fp32-export-2026-10-01.json`.

The original export status is **EXPORTED_UNVALIDATED**. Its corrected audit
includes 295 external tensor files: the first audit missed a `Constant` node
attribute. `tools/repack_onnx_external_data.py` now consolidates all ranges,
byte-identically, into ignored `models/orukeet/onnx/fp32-packed-2026-10-01/`
without overwriting the source control. The data file is 2,475,088,896 bytes.
Do not proceed to fp16 until public-library fp32 token/text parity. The CPU official
joint captures are log-softmax scores across all 8,198 channels, whereas the
export emits raw scores: compare declared equivalent normalized boundaries,
not raw logits directly to log probabilities or weakened tolerances.
Use the existing stage comparator (JSON/gzip), library smoke, TDT simulations
and sibling browser runner; do not create another decoder as a testing oracle.

Native Python ORT CPU replay passed JFK and speech/silence: 916 numerical
encoder/length, fused-head score/state/length checks at `atol=1e-5, rtol=1e-4`,
with exact token/duration argmax for all 228 recorded joint calls. Receipt:
`tools/data/results/orukeet/native-fp32-stage-replay-2026-10-01.json`.
Replay uses captured official raw encoder frames and predictor input states;
it is not accumulated ASR decoding. Raw predictor output is not exposed by
the combined graph. Pure silence, long audio, JS processor parity, general
Node/Bun native support and Orukeet WebGPU still need acceptance. Native
output-only controls do not replace these gates.

[sherpa export instructions](https://github.com/Oruk-AI/orukeet/blob/main/export/onnx/README.md)
provide source-to-graph recipes and receipts; their INT8 CPU deployment is not
our WebGPU contract. [transcribe.cpp](https://huggingface.co/oruk/orukeet/blob/main/transcribe-cpp/README.md)
offers a distinct GGUF conversion and silence/cancellation/reuse checks as a
secondary cross-runtime control, not a substitute for original fp32 NeMo.

Stock v3 browser control was run with the existing sibling harness:

```powershell
# In ../webgpu-agent-test with its Vite server listening on 127.0.0.1:8765
$env:ASRJS_WEBGPU_HOST = '127.0.0.1'
node scripts/run-parakeet-tdt-webgpu.mjs --model=v3 --mode=wasm `
  --encoder=fp16 --repeat=2 --warmup=1 --cpu-threads=4
```

`--mode=wasm` selects the decoder; the encoder remains WebGPU. Two exact text
controls passed on the real NVIDIA adapter, with clean disposal. Receipt:
`parakeet-v3-browser-control-2026-09-30.json`. It is a stock v3 baseline, not
Orukeet support; candidate artifact routing and its official oracle must be
connected before P06 can pass.

## 4. Public-library native output controls and effective PCM

Packed fp32 artifacts and the explicit experimental `native-cpu` backend have
produced exact 38-token JFK output independently in Node and Bun. The old
speech/silence run recorded 69 official versus 76 candidate tokens, but used
linear resampling rather than official NeMo/Lhotse preparation. Preserve the
failed receipt; it does not establish a same-input decoder fault.

The official engine now captures `preprocessor_input_signal` before feature
extraction. `official-speech-silence-waveform-summary-2026-10-01.json` records
1,122 stages, all six unchanged-output controls, and the exact mono float32
16 kHz / 649,600-sample input with no right padding. The f32le SHA-256 is
`8f4293517daf07f433b307a4a971f7c17dc9ed348b5f72999945b217331ec3df`.
Do not roundtrip this input through PCM16 or another resampler.

The existing inspector can use this captured input explicitly:

```powershell
node tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs `
  --family nemo-tdt --model-id oruk/orukeet `
  --model-dir models/orukeet/onnx/fp32-packed-2026-10-01 `
  --config tools/model-debugging/reference/orukeet/runtime-config-jfk.json `
  --backend native-cpu --preprocessor-backend js `
  --encoder-quant fp32 --decoder-quant fp32 --no-grid-batching --cpu-threads 4 `
  --audio tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav `
  --oracle tools/data/results/orukeet/official-speech-silence-waveform-stages-2026-10-01.json.gz `
  --sample-id speech-silence --original-weights models/orukeet/source/orukeet-v0.1.0.nemo `
  --input-mode oracle-pcm `
  --output tools/data/results/orukeet/library-node-native-speech-silence-fresh.json
```

For independent Bun validation run the same inspector/arguments with `bun` and
a different fresh output name. Already recorded Node 26.2.0 and Bun 1.3.14 runs
both match all 69 tokens, frame indices, TDT durations and full text, with
awaited cleanup PASS. Receipts:
`library-node-native-fp32-speech-silence-same-pcm-2026-10-01.json` and
`library-bun-native-fp32-speech-silence-same-pcm-2026-10-01.json`.
Both are **PARTIAL_PASS_OUTPUT_ONLY**. Frozen-input controls do not accept
normal source decoding/resampling, feature numerics, all native providers or
Orukeet browser execution. Captured padded inputs are currently rejected by
this diagnostic; legacy oracles without effective PCM remain UNVERIFIED for
that input rung. The JFK config's English label is diagnostic fixture metadata,
not the full model's language catalog or a ready public preset.

### Public consumer example (Node/Bun)

`examples/nemo-tdt-native-local.mjs` is the small public API recipe, separate
from the diagnostic inspector. Build the checkout, then run it against the
audited local fp32 graphs and a WAV at the configured sample rate:

```powershell
npm run build
node examples/nemo-tdt-native-local.mjs `
  --model-dir models/orukeet/onnx/fp32-packed-2026-10-01 `
  --config tools/model-debugging/reference/orukeet/runtime-config-jfk.json `
  --model-id oruk/orukeet --audio tools/data/fixtures/audio/jfk-short.wav
```

Replace `node` with `bun` to exercise that host. Native CPU is explicitly
experimental; no automatic download, WASM fallback or implicit resampling occurs.
Progress is on stderr and canonical/native JSON is emitted after awaited cleanup.
Graph precision is an audited input requirement, not inferred from backend
capabilities. The shared pure WAV decoder preserves float32 input and avoids
AudioContext/DOM dependencies for this native recipe.

The existing smoke wrapper invokes this actual CLI and uses the existing
reference adapter for PCM and complete output comparisons. To check the official
speech/silence effective input without changing its resampling contract:

```powershell
node tests/smoke/nemo-tdt-native-example-smoke.mjs `
  --model-id oruk/orukeet --model-dir models/orukeet/onnx/fp32-packed-2026-10-01 `
  --config tools/model-debugging/reference/orukeet/runtime-config-jfk.json `
  --audio tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav `
  --oracle tools/data/results/orukeet/official-speech-silence-waveform-stages-2026-10-01.json.gz `
  --sample-id speech-silence --original-weights models/orukeet/source/orukeet-v0.1.0.nemo `
  --prepared-wav tools/data/results/harness/native-example-official-speech-silence-f32-2026-10-01.wav `
  --output tools/data/results/orukeet/public-example-node-speech-silence-fresh.json
```

Keep the wrapper on Node; add `--executable <path-to-bun>` and a fresh output
filename to validate the public CLI on Bun. Receipts refuse overwrites, bind the
original weights/source audio, include either graph's sidecar and snapshot the
actual compiled build. Derived IEEE-float WAV bytes must match the captured
official PCM; stale rate/length/padding metadata fails even with an equal hash.
This remains **PARTIAL_PASS_OUTPUT_ONLY**, not feature parity, source-resampling,
representative-quality, browser or fp16 acceptance. Preserve earlier failures.

## 5. Weight-free FFT boundary investigation

`compare_js_feature_boundaries.py` reproduces official features from the saved
source configuration without loading ASR weights. Its guarded isolated module
experiments never write the production processor. Forty-two probe tests include
independent direct-DFT checks, shape/masked-tail and complex/power layout,
rounding, source/cache immutability and fail-closed contract changes.
Build this checkout before running the probe tests: they exercise the actual
`dist/audio/js-mel.js`, not a replacement test implementation. The supplied
source config and official capture are separately hash-bound; these diagnostics
do not independently establish a config-to-original-weight provenance chain.

`library-feature-full-fft-control-jfk-2026-10-01.json` retains the existing
`atol=1e-5, rtol=1e-4`. The official processor control is exact. Full 512-point
FFT Float32 scratch leaves 129 normalized-feature mismatches; Float64 leaves
120, like the prior final-complex-rounding control. Both power allclose checks
pass, but feature parity FAILS. Production remains unchanged.

The next control is recorded in
`library-feature-complex-fft-f32le-jfk-2026-10-01.json`. Both JS controls now
capture `[1,1101,512]` windowed input as canonical f32le bytes, preserving `-0`
instead of trusting a lossy JSON number array. Their bytes and reconstructed
official inputs share SHA-256
`e6a1dc984525ae456e0adc92085442b147ff62427581b30012ade46f4f8aaf30`.
Official framed `torch.fft.rfft` matches the actual NeMo complex STFT exactly;
JS complex components pass allclose on the same frames, but downstream features
still FAIL at 129/120 mismatches. Neither component PASS nor improved capture
promotes the port. The earlier JSON-only complex receipt is retained to expose
the serialization defect, not as bit-identity proof.

The official-complex injection control is now recorded in
`library-feature-official-complex-injection-jfk-2026-10-01.json`: both isolated
JS power formulas using canonical official STFT components give zero normalized
feature mismatches, max error `3.5762786865234375e-6`, at the unchanged tolerance.
Power allclose passes but exact power equality fails (153,688 / 1,862 differing
values); production/full-FFT features remain FAIL at 313 / 129 / 120 mismatches.
This is **DIAGNOSTIC_NOT_PORT_ACCEPTANCE** and does not establish a production
FFT replacement or numerical model acceptance.

The raw/statistics decomposition is now recorded in
`library-feature-normalization-decomposition-jfk-2026-10-01.json`. An instance-
local wrapper captures raw mel before normalization/masking without changing
the real JS process path. The tool uses actual official float32 mean/std
(including `+1e-5`), scalar JS-order statistics and explicit arithmetic residuals.
All mismatch coordinates and per-mel statistics are retained. Production has
313 mismatches; using official statistics counterfactually gives 314. Full-FFT
Float32/Float64 remain 129/120, versus 131/121 under official statistics. Both
official-complex controls still have zero. Reconstruction checks pass and the
four-term closure residual is below `8.51e-16`; these are diagnostic controls,
not relaxed promotion gates. The statistics term includes engine arithmetic
differences as well as raw-dependent statistics. Production source/compiled
processor hashes are unchanged.

The signed filterbank-weighted attribution is now recorded in
`library-feature-weighted-power-jfk-2026-10-01.json`. All four isolated controls
reconstruct every valid raw-mel value exactly. At failed coordinates, the
power/log term dominates the reference projection-rounding residual; failures
include low and high bands. Full-FFT controls still fail at 129/120, and
production still fails at 313. This is diagnostic evidence, not port acceptance.

`library-feature-selected-dft-jfk-2026-10-01.json` now compares an independent
float64 sin/cos + `math.fsum` DFT at every active bin of both full-FFT controls'
failed coordinates. All 1,003 unique frame/bin pairs use identical canonical
windowed float32 bytes. The Float64 JS control's maximum component error to
this approximation is `1.91e-15`, versus official complex64 `8.22e-7` and JS
Float32 `5.14e-7`; every Float64 control bin matches the DFT after f32 rounding.
This weakens a wrong full-Float64 FFT algorithm hypothesis. It does not prove
every production FFT path correct, exact arithmetic, or model acceptance.
Production/full-FFT features still fail at 313/129/120; the command exits 1.

Reproduce with a fresh output, after building:

```powershell
& $orukeetPython tools/model-debugging/reference/hf-parakeet-port/compare_js_feature_boundaries.py `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --source-config models/orukeet/onnx/fp32-packed-2026-10-01/source-config.json `
  --official-capture tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz `
  --sample-id jfk-short --output tools/data/results/orukeet/feature-selected-dft-jfk-fresh.json
```

Nine additional weight-free regressions cover analytic DFT sign/bin/frame
values, cancellation, immutable canonical inputs, invalid inputs, support union,
engine-error separation, signed-zero byte differences, and unknown empty
support.

The packed follow-up `library-feature-packed-dft-jfk-2026-10-01.json` captures
raw complex components before rounding/power, preserving the existing aligned
control's power formula. Upper paired bins and the quarter bin are conjugated
for the negative-exponent DFT convention; power remains unchanged. Canonical
frame bytes match the full-FFT controls exactly. At all 1,003 selected pairs,
maximum component error to DFT is `2.47e-15` and every f32-rounded component
matches. The aligned packed control still fails at 120 feature coordinates;
unmodified production still fails at 313. Forty-four probe tests pass, including
signed complex agreement with full FFT and exact rounded-power reconstruction.
The first integration run stopped on a missing normalization entry before
writing a receipt; after reverting that edit, shared control-key lists now keep
normalization, attribution and DFT analysis synchronized. The successful run
exits 1 because feature acceptance still fails, not because the harness broke.
Next isolate unmodified production frame/product deltas. This is not a
production FFT replacement, a tolerance change, or accepted model parity.

All reference/diagnostic commands now require fresh receipt paths and reject
output/input aliases before work. Source acquisition updates only the new
receipt handle it owns; ordinary JSON and gzip writes use exclusive creation.
Run `tools/test_artifact_outputs.py -v` for these small safety contracts.

The unchanged production path is now observed in
`library-feature-production-frames-final-jfk-2026-10-01.json`. The temporary
module starts from the original source before preemphasis substitutions and
only records windowed inputs and packed complex components; original power
expressions remain intact. Canonical little-endian float64 frame bytes preserve
the actual window products (including signed zeros). Exact raw-valid-prefix,
normalized-feature and reconstructed-power controls pass. Raw full-buffer test
uses `computeRawMel`, since even `normalization: none` finalization masks the
invalid tail; confusing these boundaries would create a false regression.

Official f32 frame values differ from production f64 products at 435,823 of
563,712 values (max `9.44e-8`, mean `1.54e-9`). Explicitly casting production
frames to f32 still gives 365,586 bit differences; this is not same-input proof.
Independent NumPy FFT on the captured production f64 frames agrees with packed
components within `7.11e-15` real / `6.22e-15` imaginary across all 282,957 bins.
NumPy is a same-input arithmetic diagnostic, not a replacement official oracle.
Production features still fail at 313 coordinates, aligned packed features at
120. The successful diagnostic exits 1 because acceptance remains incomplete.
Forty-nine probe tests pass, including source/output immutability, exact original
power reconstruction, f64 byte/mirror checks and malformed shape/coercion
refusal. The cumulative controls below now separate preemphasis and window deltas.
No production code or tolerances
changed. Earlier preliminary receipts remain preserved, not overwritten.

### Cumulative arithmetic and original-weight pure silence

The final JFK matrix retains original packed power arithmetic and gives
313 → 200 → 139 → 121 failed feature coordinates as f32 preemphasis, official
window values and rounded products are introduced cumulatively. Final window
products equal the official FFT inputs exactly; features still FAIL.

Pure silence uses a fresh generated 1-second 16 kHz mono PCM16 exact-zero WAV:

```powershell
ffmpeg -hide_banner -loglevel error -f lavfi -i anullsrc=r=16000:cl=mono -t 1 -c:a pcm_s16le -n tools/data/results/orukeet/pure-silence-1s-16k-pcm16.wav
```

The WAV SHA-256 is `57d8dd674034eb744bd8dfa71344ee02e1a8e709563cef10108eda63663675c8`.
`official-pure-silence-1s-summary-2026-10-01.json` binds the original-weight
NeMo capture (empty transcript/tokens, 60 stages, six exact instrumentation
controls). Own fp32 native stage replay passes 50 numerical comparisons and
12 exact decoding decisions; accumulated library decoding and browser remain
UNTESTED by this replay.

All silence frame controls and raw mel match the reference exactly, but
normalization fails at 12,800 coordinates (max `0.1600533276796341`). For the
first constant raw row, reference mean is `-16.635530471801758` and guarded std
`1.1916957191715483e-5`, while JS mean is `-16.63553237915039` and guarded std
`1e-5`. Inspect official float32 masked reductions before proposing a shared
processor fix. No silence offset, tolerance adjustment or production patch is
justified by this diagnostic alone.

Reports accept fresh `.json.gz` paths with exclusive creation; `--summary-only`
only shortens console output, not evidence or exit criteria. Final local
silence/JFK compressed receipts preserve full diagnostics in approximately
1.53 MB / 0.72 MB. Both exit 1 because feature acceptance fails. 53 focused
tests pass. Hashes and next steps are in the frame-arithmetic-and-silence handoff.
