"""Isolate whether head call 20's score gap is input-driven or a head defect.

Uses immutable official captured stages as the reference and the recorded
browser feeds. Feeds the fp32 joint graph (native CPU) the reference inputs,
the recorded browser inputs, and reference inputs perturbed by exactly the
recorded input deltas. No tolerance, normalization or acceptance change.

Read-only diagnostic: it loads existing artifacts and writes nothing.
"""
import gzip
import json
import sys

import numpy as np
import onnxruntime as ort
sys.path.insert(0, "tools/nemo-tdt-onnx-export")
from validate_stage_replay import log_softmax
CALL = 20
FRAME = 69  # joint/0020 binds predictor/0015 at encoder frame 69

CAP = "tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz"
RUN = "tools/data/results/orukeet/browser-real-public-hybrid-jfk-2026-10-03.json"
GRAPH = "models/orukeet/onnx/fp32-packed-2026-10-01/decoder_joint-model.onnx"
PRED = "predictor/0015"

with gzip.open(CAP) as fh:
    cap = json.load(fh)
run = json.load(open(RUN))
stages = cap["samples"][0]["stages"]
calls = run["result"]["runs"][0]["headCalls"]

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(GRAPH, sess_options=opts, providers=["CPUExecutionProvider"])
names = [o.name for o in sess.get_outputs()]
print("actual providers:", sess.get_providers())
print("graph outputs:", names)


def t(value):
    return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])


def run_head(feeds):
    return dict(zip(names, sess.run(None, feeds)))


official_ls = t(stages[f"joint/{CALL:04d}/output"])
ref_feeds = {
    "encoder_outputs": np.ascontiguousarray(t(stages["encoder"])[:, :, FRAME:FRAME + 1]),
    "targets": t(stages[PRED + "/labels"]).astype(np.int32),
    "target_length": np.ones(1, dtype=np.int32),
    "input_states_1": t(stages[PRED + "/input_state_h"]),
    "input_states_2": t(stages[PRED + "/input_state_c"]),
}
call = calls[CALL]
recorded_feeds = {name: t(value) for name, value in call["feeds"].items()}

deltas = {}
print("== recorded browser inputs vs official reference inputs ==")
for name, reference in ref_feeds.items():
    candidate = recorded_feeds[name]
    if reference.shape == candidate.shape:
        deltas[name] = candidate.astype(np.float32) - reference.astype(np.float32)
        print(f"  {name}: shape={reference.shape} delta maxAbs={np.abs(deltas[name]).max():.6e}")

ref_out = run_head(ref_feeds)
print()
print("== like-for-like against the official captured normalized scores ==")


def report(label, scores):
    ls = log_softmax(scores)
    err = np.abs(official_ls - ls)
    bad = np.flatnonzero(err > 1e-5 + 1e-4 * np.abs(official_ls))
    print(f"  {label}: maxAbsError={err.max():.6e} meanAbsError={err.mean():.6e} "
          f"mismatch={bad.size} first={bad[0] if bad.size else None}")
    return err


report("official reference inputs, native CPU", ref_out["outputs"])
report("recorded browser inputs, native CPU", run_head(recorded_feeds)["outputs"])

encoder_only = dict(ref_feeds)
encoder_only["encoder_outputs"] = np.ascontiguousarray(
    ref_feeds["encoder_outputs"] + deltas["encoder_outputs"]
)
report("reference + recorded encoder delta only", run_head(encoder_only)["outputs"])

states_only = dict(ref_feeds)
states_only["input_states_1"] = np.ascontiguousarray(
    ref_feeds["input_states_1"] + deltas["input_states_1"]
)
states_only["input_states_2"] = np.ascontiguousarray(
    ref_feeds["input_states_2"] + deltas["input_states_2"]
)
report("reference + recorded predictor-state deltas only", run_head(states_only)["outputs"])

all_perturbed = {}
for name, reference in ref_feeds.items():
    if name in deltas:
        perturbed = reference + deltas[name]
    else:
        perturbed = reference
    all_perturbed[name] = np.ascontiguousarray(perturbed, dtype=reference.dtype)
report("reference + all recorded input deltas", run_head(all_perturbed)["outputs"])

# State the decision impact of the reference-input run.
reference_scores = ref_out["outputs"]
blank = 8193
print()
print("== decision check on the reference-input run ==")
print("  token argmax:", int(np.argmax(reference_scores[..., :blank + 1])),
      "official:", int(np.argmax(official_ls[..., :blank + 1])))
print("  duration argmax:", int(np.argmax(reference_scores[..., blank + 1:])),
      "official:", int(np.argmax(official_ls[..., blank + 1:])))