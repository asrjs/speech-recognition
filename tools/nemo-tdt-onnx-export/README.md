# Official NeMo TDT fp32 export

Family-owned tooling for offline TDT encoder + combined predictor/joint graphs.
The source model supplies dimensions, vocabulary and export preparation; no
Orukeet/Parakeet-specific size or weight path is hardcoded in the exporter.
Prompted/cache-aware RNNT families are rejected, not silently adapted.

Run from the library root using the existing NeMo environment:

```powershell
$tdtPython = 'tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe'
& $tdtPython tools/nemo-tdt-onnx-export/test_export_nemo_fp32.py -v
& $tdtPython tools/nemo-tdt-onnx-export/export_nemo_fp32.py `
  --nemo models/orukeet/source/orukeet-v0.1.0.nemo `
  --expected-sha256 031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56 `
  --model-id oruk/orukeet `
  --model-revision b59c13a733fce5cf193230d0fa317ee7f145a108 `
  --output-dir models/orukeet/onnx/fp32-nemo-2026-10-01 `
  --cpu-threads 4
```

The output directory must not already exist, even if empty. Use a new directory
for another run; no original weights or previous export files are overwritten.
Weights are hash-checked before creating output. Failed runs preserve partial
artifacts and a failure receipt; do not load them as a completed port.

The exporter uses official NeMo subnet `export()` with legacy Torch/opset 17,
fp32, dynamic input time/batch, small deterministic trace inputs and CPU. The
encoder trace has 256 frames by default; the predictor/joint has one frame and
one label instead of the joint's 8,192-frame default. NeMo preparation owns
`_rnnt_export` (no extra SOS), `log_softmax=False` and fused-loss disablement.
These mutations persist in the restored instance: discard it after exporting;
never reuse it as an unchanged original-engine oracle.

Output includes `encoder-model.onnx`, `decoder_joint-model.onnx`, colocated
external tensors where needed, source config, indexed vocabulary and
`export-receipt.json` (source hash/revision, environment, I/O, opset, graph and
external-data hashes). ONNX checking and contract audit do not prove numerical
parity, dynamic-shape execution, performance or browser support. Successful
status is deliberately `EXPORTED_UNVALIDATED`.

The first Orukeet export actually has 295 external tensors; the initial
initializer-only audit missed a Constant attribute. The shared tensor walker
now includes attributes, nested graphs, sparse tensors and functions. A new
candidate is consolidated using the model-neutral streaming repacker:

```powershell
& $tdtPython tools/repack_onnx_external_data.py `
  --graph models/orukeet/onnx/fp32-nemo-2026-10-01/encoder-model.onnx `
  --expected-sha256 325c9405923f670eea47405806b02f64167723f36b745f5ddf62e776259aa7b8 `
  --output-dir models/orukeet/onnx/fp32-packed-2026-10-01 `
  --copy-file decoder_joint-model.onnx --copy-file vocab.txt --copy-file source-config.json
```

This preserves original files, streams 8 MiB ranges, verifies source/destination
SHA-256 and writes a failure-safe receipt. All external tensors (not only
initializers) are repointed into one colocated `.onnx.data`; inline weights are
unchanged. Aliased source ranges are copied separately. Source checksum/basepath
metadata is removed because file identity changes; receipt hashes replace it.
Existing output directories are refused. Byte identity is not model parity.

`validate_stage_replay.py` replays saved official features and raw encoder
frames/labels/predictor states through native Python ORT CPU, with explicit
atol/rtol and exact recorded token/duration decisions. It does not contain a
search loop. Use `--help`; repeat matching `--capture`, `--summary`, `--audio`
arguments for each fixture. Output must be new; source weight/revision,
audio/capture/summary identity, controls and packed hashes are checked.
JFK and speech/silence passed (916 numerical comparisons / 228 joint calls);
see `tools/data/results/orukeet/native-fp32-stage-replay-2026-10-01.json`.
This is partial boundary evidence: predictor raw output is not exposed,
accumulated library tokens/text and other fixtures still require validation.
Do not create another greedy decoder as an oracle. Account explicitly for
CPU NeMo's normalized joint outputs versus exported raw scores.

Eight small stdlib `unittest` exporter safety checks require the existing ONNX/NumPy
dependencies but neither pytest nor model weights. They cover overwrite
refusal, content hashes, family exclusion, graph names/dtypes/ranks, sidecar
hashes, escape/missing sidecars and out-of-range external data. Their PASS is
tooling evidence only. Run `tools/test_onnx_artifacts.py` (9),
`tools/test_tensor_parity.py` (5) and `test_validate_stage_replay.py` (7) for
the additional small tooling contracts; they require no ASR weights and are
not real inference evidence. Follow `docs/PARAKEET_VARIANT_TASKS.md` for remaining
original-engine, native, WASM and real-browser acceptance.

## Opt-in one-token fp32 head candidate

`expand_single_step_lstm.py` is a family-owned post-export transformation,
not a change to the default exporter or a public model preset. It replaces
two supported default forward LSTMs with explicit IOFC fp32 equations while
preserving all other operators, weights and metadata. Hidden/input widths come
from the config and graph, not a brand or a 640 constant. Only `targets` axis 1
is fixed to one token. Fresh shape inference must prove each ONNX LSTM input
sequence axis 0 is consequently one; it does not assume these are the same axis.
Batch/encoder-grid input axes are unchanged, not automatically certified.

```powershell
python tools/nemo-tdt-onnx-export/test_expand_single_step_lstm.py -v
python tools/nemo-tdt-onnx-export/expand_single_step_lstm.py `
  --source models/orukeet/onnx/fp32-packed-2026-10-01/decoder_joint-model.onnx `
  --expected-sha256 89fb566d2f0cd339387582dec9753959c84d54a90272581e7f51ceecc5a8eb90 `
  --config models/orukeet/onnx/fp32-packed-2026-10-01/source-config.json `
  --expected-config-sha256 344297eaf9e28d7f849e099c06535fbb0df978926e1b5f753b6aa836d05acdb3 `
  --output-dir tools/data/results/orukeet/my-fresh-head-candidate
```

Both source buffers are hashed before parsing. Existing output directories,
including empty ones, are refused. Source files remain immutable; transpose
and bias splitting are byte-reversible. Nondefault activations/direction/layout,
clipping, coupled input/forget gates, sequence lengths/peepholes, missing states,
shared W/R/B consumers, unknown sequence axes, external weights, name collisions
and incompatible family I/O/configuration are rejected. Output includes only
the candidate head, unchanged config and `transformation-receipt.json` with
`CANDIDATE_GRAPH_CREATED_UNVALIDATED`. It is not a complete loadable model bundle.
Keep large candidate graphs/raw traces local, not in Git.

The existing `browser_head_replay.py` accepts `--candidate-dir` in addition to
its unchanged required original `--artifact-dir` and `--export-receipt` gates.
It validates the candidate source/config/tool identity, reproduces the entire
transformation in memory and requires an exact candidate graph hash. Do not
forge an export receipt for the transformed head. Add `--all-steps` to replay
every recorded binding twice; candidate receipts use a separate
`RECORDED_CANDIDATE_HEAD_COMPONENT_*` status. Numerical tolerances and exact
token/duration/length checks are unchanged. Recorded states are independent
oracle inputs, not a candidate-fed ASR search loop. Full library recurrence,
text/timestamps, real grid/batch execution per backend, features/encoder and
matched memory/performance remain distinct promotion gates. No fp16 promotion.

## Recorded browser head gate

`browser_head_replay.py` runs three captured first/middle/last joint calls twice
in headless installed Chrome, through the compiled library's internal ORT
session bridge. It reuses `replay_bindings` and the native score comparator;
there is no alternate ASR search loop. It uses a loopback-only, allowlisted
ephemeral server and does not edit/start sibling projects or download assets.
The system Python needs existing ONNX/NumPy/ORT and Playwright dependencies;
the older NeMo environment does not currently contain Playwright. Run `--help`
first. Do not install dependencies implicitly or reuse an output path.

```powershell
python tools/nemo-tdt-onnx-export/browser_head_replay.py `
  --capture tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz `
  --summary tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json `
  --audio tools/data/fixtures/audio/jfk-short.wav `
  --artifact-dir models/orukeet/onnx/fp32-packed-2026-10-01 `
  --export-receipt models/orukeet/onnx/fp32-nemo-2026-10-01/export-receipt.json `
  --backend webgpu --output tools/data/results/orukeet/browser-head-new-run.json
```

Use `--backend wasm` separately. Receipts include source/artifact/fixture/tool
hashes, actual browser/ORT versions, all returned tensor data and comparisons,
exact decisions, repeated calls, cleanup errors and failed browser requests.
WebGPU additionally requires compute dispatches during every graph call;
this is not proof that every graph node executes on GPU. Original CPU scores
are all-channel log probabilities; exported raw scores are normalized before
comparison at the existing `atol=1e-5, rtol=1e-4`. Predictor raw output is not
exposed. This does not certify features, encoder, accumulated sequence, public
API, application integration or performance. Seven weight-free tests validate
mapping, fail-closed controls/identity/outputs, repeats, cleanup and routes.

2026-10-02: one initial WebGPU run passed six numerical/decision checks, but
subsequent WASM and hash-bound WebGPU runs failed local asset fetches with
`ERR_CONNECTION_RESET`, before graph calls. Component browser acceptance is
therefore inconsistent/unaccepted, not a demonstrated unsupported operator.
The original pass predates complete tool-hash/raw-output receipt binding.
See `browser-head-handoff-2026-10-02.json`; investigate local delivery before
another model or precision experiment. No backend fallback or tolerance change.

Follow-up on the same date: HTTP/1.0 reproduced a standalone WASM reset even
though server byte/hash logging showed a complete write. An HTTP/1.1-only
child-process intervention passed nine exact asset fetches in three clean
Chrome cycles. The handler now uses HTTP/1.1 with Content-Length and records
each request's status, written byte count/hash and write errors. Four subsequent
actual selected-head runs passed (WASM/WebGPU, two fresh Chrome instances each,
six calls each) with full source/tool/output binding, no fetch or cleanup errors.
This supersedes the delivery-blocked selected-head status above, not full port
acceptance or a proof of the underlying OS/network-filter cause. Nine focused
tool tests include HTTP response framing and connection reuse. See
`browser-head-http11-handoff-2026-10-02.json`. Next cover all recorded calls and
additional official fixtures; retain feature/encoder/public API gates separately.

Use `--all-steps` to replay every validated chronological joint binding twice
instead of the default first/middle/last selection. Receipts name the selection
and total recorded joint-call count. This still feeds recorded encoder frames,
labels and recurrent states: it is not a state-propagating decode sequence.
The comparator rejects missing or reordered calls. Ten focused weight-free
tests cover both selection modes and missing-call rejection; they are not GPU
or model parity evidence. Raw result receipts grow with the complete call count.

For a bounded fresh-session diagnostic, repeat `--step-index N` in strictly
increasing order (zero-based recorded joint binding index). It is mutually
exclusive with `--all-steps`; empty, duplicate, reordered and out-of-range
selections fail before browser creation. The fixture still validates the full
capture's source/control/call contract, but compares only the named bindings.
Receipts label this `explicit-indices`, never complete-matrix acceptance.
Example: `--step-index 139 --step-index 173` isolates the two speech/silence
cell-state failures. Eleven weight-free browser-tool tests cover these rules.

## Diagnostic frozen encoder boundary

`frozen_encoder_replay.py` prepares a small **diagnostic fixture, not a model
export**. It preserves the own encoder graph's input/output contracts, but
replaces computation with two Constant nodes holding the exact official captured
encoder tensor and length. Inputs are deliberately ignored; no external model
weights are embedded. This prepares an isolated test of the library's real
recurrent decoder without copying its search loop or changing production code.
Generating this fixture does not run or certify that decoder.

```powershell
python tools/nemo-tdt-onnx-export/frozen_encoder_replay.py `
  --capture tools/data/results/orukeet/official-speech-silence-stft-stages-2026-10-02.json.gz `
  --summary tools/data/results/orukeet/official-speech-silence-stft-summary-2026-10-02.json `
  --audio tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav `
  --artifact-dir models/orukeet/onnx/fp32-packed-2026-10-01 `
  --export-receipt models/orukeet/onnx/fp32-nemo-2026-10-01/export-receipt.json `
  --nemo models/orukeet/source/orukeet-v0.1.0.nemo `
  --output-dir tools/data/results/orukeet/diagnostic-frozen-new-run
```

Use a fresh output directory. The detailed export receipt is required, not its
compact handoff summary. Original weight/export/packed graph/config hashes and
official instrumentation/source/audio/call guards must pass before writing.
Native ORT verifies both output tensors bitwise, including signed zero, first
with captured features and again with zero features. Receipts explicitly label
features and the real encoder bypassed/unproven, recurrence and browser untested,
and model acceptance not proven. No external-weight bytes are used by this tool;
it binds the source graph contract, not a new validation of its external weights.

Nine weight-free tests exercise deterministic serialization, source-contract
preservation, malformed contracts/shapes/lengths/data, and native exact delivery.
Next: use the fixture only in an explicitly conditional public-library browser
runner, exact captured PCM, audited candidate decoder, encoder WASM/decoder
WASM or WebGPU, grid batching disabled, awaited cleanup and full provenance.
Keep the production feature gate and full real-encoder ASR gate separate.
