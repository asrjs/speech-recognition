"""Apply the shared attribution control to the recorded Orukeet head case.

Demonstrates that tools/stage_attribution.py localizes a real failing
boundary, not only synthetic runners. Uses the immutable official JFK capture
as reference inputs and the recorded browser feeds as candidate inputs,
against the real fp32 joint graph on native CPU.

Three binding gates run before any attribution is reported:

1. The reference-input replay must reproduce the recorded official scores.
2. The native replay of the recorded feeds must reproduce the browser's own
   head output. The browser receipt stores raw joint logits, so they are
   normalized with the same contract first.
3. The failing coordinate set must equal the recorded failing set exactly.
   A non-empty mismatch count alone does not establish reproduction.

If any gate fails this exits UNRESOLVED instead of attributing anything.

Read-only: reads existing receipts and writes nothing.
"""
import gzip
import json
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, "tools")
sys.path.insert(0, "tools/nemo-tdt-onnx-export")

from stage_attribution import (  # noqa: E402
    attribute_input_source,
    compare_stage_outputs,
    sweep_input_scale,
)
from validate_stage_replay import log_softmax  # noqa: E402

CAPTURE = "tools/data/results/orukeet/official-jfk-stages-2026-09-30.json.gz"
HYBRID = "tools/data/results/orukeet/browser-real-public-hybrid-jfk-2026-10-03.json"
GRAPH = "models/orukeet/onnx/fp32-packed-2026-10-01/decoder_joint-model.onnx"
CALL, FRAME, PRED = 20, 69, "predictor/0015"
ATOL, RTOL = 1e-5, 1e-4


def as_array(value):
    return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])


def failing_indices(reference_value, candidate_value):
    """Indices failing the full per-coordinate gate, as sorted channel ids."""
    err = np.abs(reference_value - candidate_value)
    bad = np.flatnonzero(err > ATOL + RTOL * np.abs(reference_value))
    return sorted(int(i) for i in bad)


cap = json.load(gzip.open(CAPTURE))
hybrid = json.load(open(HYBRID))
stages = cap["samples"][0]["stages"]
calls = hybrid["result"]["runs"][0]["headCalls"]

official_record = stages[f"joint/{CALL:04d}/output"]
official_scores = np.asarray(official_record["data"], dtype=np.float32).reshape(
    official_record["shape"]
)

options = ort.SessionOptions()
options.intra_op_num_threads = 4
session = ort.InferenceSession(
    GRAPH, sess_options=options, providers=["CPUExecutionProvider"]
)
names = [o.name for o in session.get_outputs()]


def run_head(feeds):
    """Run the joint graph and normalize outputs per the replay contract.

    targets and target_length are required by the graph and are identical on
    both sides for this binding, so they come from the reference and stay out
    of the perturbed set.
    """
    complete = {
        **feeds,
        "targets": as_array(stages[PRED + "/labels"]).astype(np.int32),
        "target_length": np.ones(1, dtype=np.int32),
    }
    outputs = dict(zip(names, session.run(None, complete)))
    outputs["normalized_scores"] = log_softmax(outputs["outputs"])
    return outputs


reference_feeds = {
    "encoder_outputs": np.ascontiguousarray(
        as_array(stages["encoder"])[:, :, FRAME:FRAME + 1]
    ),
    "input_states_1": np.ascontiguousarray(as_array(stages[PRED + "/input_state_h"])),
    "input_states_2": np.ascontiguousarray(as_array(stages[PRED + "/input_state_c"])),
}
candidate_feeds = {
    "encoder_outputs": np.ascontiguousarray(
        as_array(calls[CALL]["feeds"]["encoder_outputs"])
    ),
    "input_states_1": np.ascontiguousarray(as_array(calls[CALL]["feeds"]["input_states_1"])),
    "input_states_2": np.ascontiguousarray(as_array(calls[CALL]["feeds"]["input_states_2"])),
}

print("actual providers:", session.get_providers())

reference_replay = run_head(dict(reference_feeds))
candidate_replay = run_head(dict(candidate_feeds))
browser_scores = log_softmax(as_array(calls[CALL]["outputs"]["outputs"]))


def metrics(reference_value, candidate_value):
    return compare_stage_outputs(
        {"normalized_scores": reference_value},
        {"normalized_scores": candidate_value},
        atol=ATOL,
        rtol=RTOL,
    )["outputs"]["normalized_scores"]


baseline = metrics(official_scores, reference_replay["normalized_scores"])
candidate_vs_official = metrics(official_scores, candidate_replay["normalized_scores"])
browser_agreement = metrics(browser_scores, candidate_replay["normalized_scores"])

reproduced = failing_indices(official_scores, candidate_replay["normalized_scores"])
recorded_failure = failing_indices(official_scores, browser_scores)

print()
print("== binding gates before any attribution ==")
print(f"  1 reference replay vs official scores : maxAbs={baseline['max_abs_error']:.6e} "
      f"mismatch={baseline['mismatch_count']}")
print(f"  2 native replay vs browser head output: maxAbs={browser_agreement['max_abs_error']:.6e} "
      f"mismatch={browser_agreement['mismatch_count']}")
print(f"  3 candidate replay vs official        : maxAbs={candidate_vs_official['max_abs_error']:.6e} "
      f"mismatch={candidate_vs_official['mismatch_count']}")
print(f"    failing indices, native replay: {reproduced}")
print(f"    failing indices, browser run  : {recorded_failure}")

if not baseline["allclose"]:
    raise SystemExit(
        "UNRESOLVED: the reference-input replay does not reproduce the recorded "
        "official scores; this control cannot attribute this case."
    )
if not browser_agreement["allclose"]:
    raise SystemExit(
        "UNRESOLVED: the native replay of the recorded feeds does not reproduce "
        "the browser's own head output; this control cannot attribute this case."
    )
if not reproduced or reproduced != recorded_failure:
    raise SystemExit(
        f"UNRESOLVED: failing coordinates {reproduced} do not match the recorded "
        f"failure {recorded_failure}."
    )
print("  all three binding gates PASS")

result = attribute_input_source(
    reference_feeds,
    candidate_feeds,
    run_head,
    atol=ATOL,
    rtol=RTOL,
    output_names=["normalized_scores"],
)

print()
print("== shared attribution control on the real recorded failure ==")
print(f"  status: {result['status']}")
print(f"  resolved: {result['resolved']}")
print(f"  inputs tested: {result['inputs_tested']}")
print(f"  outputs compared: {result['outputs_tested']}")
print(f"  implicated: {result['implicated_inputs']}")
for name, row in result["per_input"].items():
    observed = row["comparison"]["outputs"]["normalized_scores"]
    print(
        f"    {name:16s} deltaMaxAbs={row['input_delta_max_abs_error']:.6e} "
        f"failsGate={row['fails_gate_against_reference']} "
        f"reproducesOutput={row['reproduces_observed_output']} "
        f"mismatch={observed['mismatch_count']} "
        f"first={observed['first_divergent_index']}"
    )

sweep = sweep_input_scale(
    reference_feeds,
    candidate_feeds,
    run_head,
    scales=[0.0, 0.25, 0.5, 1.0],
    atol=ATOL,
    rtol=RTOL,
    output_names=["normalized_scores"],
)
print()
print("== directional sweep along the recorded feed delta ==")
for row in sweep["rows"]:
    print(
        f"  scale {row['scale']:.2f}  maxAbs={row['max_abs_error']['normalized_scores']:.6e}  "
        f"allclose={row['allclose']}  "
        f"first={row['first_divergent_index']['normalized_scores']}"
    )
print(f"  largest passing scale: {sweep['largest_passing_scale']}")
print(f"  {sweep['limitations']}")