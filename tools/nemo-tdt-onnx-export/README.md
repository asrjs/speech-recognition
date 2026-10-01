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
