"""Encoder-side accuracy budget for the live WebGPU encoder.

Reports what the recorded real-browser runs actually deliver, and the
empirical directional limit observed by scaling the recorded encoder delta.
No gain constant and no assumed tolerance form: the feed-error ceiling is
found by running the fp32 joint graph on official reference inputs perturbed
by scaled encoder deltas, comparing with the full unchanged
`atol + rtol*|ref|` gate.

Read-only: it reads existing receipts and writes nothing.
"""
import gzip
import json
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "tools")
sys.path.insert(0, "tools/nemo-tdt-onnx-export")
from validate_stage_replay import log_softmax
from stage_attribution import sweep_input_scale

ENCODER_RECEIPT = "tools/data/results/orukeet/browser-production-feature-encoder-jfk-2026-10-03.json"
HYBRID = "tools/data/results/orukeet/browser-real-public-hybrid-jfk-2026-10-03.json"
CAPTURE = "tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz"
GRAPH = "models/orukeet/onnx/fp32-packed-2026-10-01/decoder_joint-model.onnx"
CALL, FRAME, PRED = 20, 69, "predictor/0015"

ATOL, RTOL = 1e-5, 1e-4

receipt = json.load(open(ENCODER_RECEIPT))
hybrid = json.load(open(HYBRID))
result = receipt["result"]

print("== what the recorded real WebGPU encoder runs deliver ==")
comparison = result["feature_comparison"]
print(f"  encoder gate, official features   : {result['official_encoder_pass']}")
print(f"  encoder gate, production-JS feats : {result['production_encoder_pass']}")
print(f"  upstream feature gate             : allclose={comparison['allclose']} "
      f"max_abs_error={comparison['max_abs_error']:.6e} "
      f"mismatch={comparison['mismatch_count']}/{comparison['count']}")

runs = result["runs"]
by_source = {}
for run in runs:
    by_source.setdefault(run["featureSource"], []).append(run["outputs"]["outputs"])
print()
print("== repeat determinism on an identical feature source ==")
for source, outputs in by_source.items():
    arrays = [np.asarray(o["data"], dtype=np.float32).reshape(o["shape"]) for o in outputs]
    delta = np.abs(arrays[0] - arrays[1])
    print(f"  {source:>13}: maxAbsError={delta.max():.6e} "
          f"bitwise_exact={arrays[0].tobytes() == arrays[1].tobytes()}")

# Official-feature WebGPU encoder output against the official encoder tensor.
print()
print("== official-feature WebGPU encoder vs official captured encoder tensor ==")
cap = json.load(gzip.open(CAPTURE))
official = np.asarray(cap["samples"][0]["stages"]["encoder"]["data"], dtype=np.float32)
official = official.reshape(cap["samples"][0]["stages"]["encoder"]["shape"])
webgpu = np.asarray(by_source["official"][0]["data"], dtype=np.float32).reshape(
    by_source["official"][0]["shape"])
if webgpu.shape == official.shape:
    err = np.abs(webgpu - official)
    covered = err <= ATOL + RTOL * np.abs(official)
    print(f"  shape={official.shape} full-tensor maxAbsError={err.max():.6e} "
          f"meanAbsError={err.mean():.6e}")
    print(f"  coordinates failing the unchanged gate: {(~covered).sum()}/{err.size}")
else:
    print(f"  shape mismatch: webgpu={webgpu.shape} official={official.shape}")
    print("  encoder output shape is not directly comparable to the official tensor")

print()
print("== observed live feed error at the decoder input boundary ==")
comparisons = hybrid["live_head"]["comparisons"]
enc = np.array([c["metrics"]["input/encoder_outputs"]["max_abs_error"] for c in comparisons])
passing = sum(1 for c in comparisons if c["metrics"]["input/encoder_outputs"]["allclose"])
print(f"  calls={enc.size} min={enc.min():.6e} median={np.median(enc):.6e} max={enc.max():.6e}")
print(f"  calls passing the unchanged encoder gate: {passing}/{enc.size}")

# Empirical directional limit: scale the recorded call-20 encoder delta and
# apply the full per-coordinate gate to the resulting score error.
print()
print("== empirical feed-error ceiling at call 20 (full atol+rtol*|ref| gate) ==")
stages = cap["samples"][0]["stages"]
official_scores = np.asarray(
    stages[f"joint/{CALL:04d}/output"]["data"], dtype=np.float32
).reshape(stages[f"joint/{CALL:04d}/output"]["shape"])
t = lambda v: np.asarray(v["data"], dtype=v["dtype"]).reshape(v["shape"])
recorded = {k: t(v) for k, v in hybrid["result"]["runs"][0]["headCalls"][CALL]["feeds"].items()}
reference_feeds = {
    "encoder_outputs": np.ascontiguousarray(t(stages["encoder"])[:, :, FRAME:FRAME + 1]),
    "targets": t(stages[PRED + "/labels"]).astype(np.int32),
    "target_length": np.ones(1, dtype=np.int32),
    "input_states_1": t(stages[PRED + "/input_state_h"]),
    "input_states_2": t(stages[PRED + "/input_state_c"]),
}
delta = (recorded["encoder_outputs"] - reference_feeds["encoder_outputs"]).astype(np.float32)
options = ort.SessionOptions()
options.intra_op_num_threads = 4
sess = ort.InferenceSession(GRAPH, sess_options=options, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
print(f"  actual providers: {sess.get_providers()}")
print(f"  recorded encoder delta maxAbs={np.abs(delta).max():.6e}")


def run_head(feeds):
    """Vary only encoder_outputs; every other feed stays on the reference."""
    complete = {**reference_feeds, **feeds}
    outputs = dict(zip(names, sess.run(None, complete)))
    return {"normalized_scores": log_softmax(outputs["outputs"])}


# The sweep uses the shared model-neutral control rather than a local copy.
# Only encoder_outputs is varied, matching the original measurement. The
# oracle is supplied explicitly so each row is gated against the recorded
# official scores, which keeps the native execution residual visible at scale
# 0 instead of collapsing it to zero.
sweep = sweep_input_scale(
    {"encoder_outputs": reference_feeds["encoder_outputs"]},
    {"encoder_outputs": recorded["encoder_outputs"]},
    run_head,
    scales=[0.0, 0.05, 0.1, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    atol=ATOL,
    rtol=RTOL,
    output_names=["normalized_scores"],
    compare_against={"normalized_scores": official_scores},
)
print("  scale   feedMaxAbs     scoreMaxAbsError  mismatch")
for scale, row in zip(sweep["scales"], sweep["rows"]):
    feed = float(np.abs(delta * scale).max())
    metrics = row["max_abs_error"]["normalized_scores"]
    count = row["mismatch_count"]["normalized_scores"]
    print(f"  {scale:5.3f}  {feed:.6e}  {metrics:.6e}       {count:3d}")
print()
print(f"  largest passing scale: {sweep['largest_passing_scale']}")
print(f"  observed live feed error (median/max): {np.median(enc):.6e} / {enc.max():.6e}")
print(f"  {sweep['limitations']}")