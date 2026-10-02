"""Compare observed library head calls, never drive a decoder or oracle loop."""
import numpy as np

from tensor_parity import compare_tensors
from validate_stage_replay import log_softmax


def tensor(value):
    dtype, shape, data = value["dtype"], value["shape"], value["data"]
    if dtype not in {"float32", "int32"} or not shape or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Trace tensor dtype/shape differs")
    if not isinstance(data, list) or len(data) != np.prod(shape):
        raise ValueError("Trace tensor count differs")
    if dtype == "int32":
        if any(type(n) is not int or not -(2**31) <= n < 2**31 for n in data):
            raise ValueError("Trace integer coercion/range differs")
    elif any(type(n) not in (int, float) or not np.isfinite(n) or np.float32(n) != n for n in data):
        raise ValueError("Trace requires finite exact float32 values")
    return np.asarray(data, dtype=dtype).reshape(shape)


def compare_live_head(result, head):
    runs = result["runs"]
    if (len(runs) != 2 or runs[0].get("traceEnabled") is not True
            or runs[1].get("traceEnabled") is not False or runs[1].get("headCalls") != []
            or any(run.get("observerRestored") is not True for run in runs)):
        raise ValueError("Traced/uninstrumented observation control missing")
    calls = runs[0]["headCalls"]
    if not isinstance(calls, list) or not calls or len(calls) != len(head["steps"]):
        raise ValueError("Live/official call count differs")
    comparisons, good = [], True
    for index, (call, expected) in enumerate(zip(calls, head["steps"])):
        if type(call["index"]) is not int or call["index"] != index:
            raise ValueError("Trace call chronology differs")
        if set(call["feeds"]) != set(expected["feeds"]) or set(call["outputs"]) != {
                "outputs", "output_states_1", "output_states_2", "prednet_lengths"}:
            raise ValueError("Missing/unexpected live head fields")
        entry = {"index": index, "binding": expected["binding"], "metrics": {}}
        for name, reference in expected["feeds"].items():
            left, right = tensor(reference), tensor(call["feeds"][name])
            metrics = compare_tensors(left, right, atol=1e-5, rtol=1e-4)
            # Frozen frames, targets and lengths must be exact, not allclose.
            if name in {"encoder_outputs", "targets", "target_length"}:
                metrics["bitwise_exact"] = left.tobytes() == right.tobytes()
                good = metrics["bitwise_exact"] and good
            entry["metrics"]["input/" + name] = metrics
            good = metrics["allclose"] and good
        scores = tensor(call["outputs"]["outputs"])
        reference_scores = tensor(expected["expected"]["normalized_scores"])
        metrics = compare_tensors(reference_scores, log_softmax(scores), atol=1e-5, rtol=1e-4)
        entry["metrics"]["normalized_scores"] = metrics
        good = metrics["allclose"] and good
        blank = head["blank_id"]
        entry["token_duration_exact"] = (
            int(np.argmax(scores[..., :blank+1])) == int(np.argmax(reference_scores[..., :blank+1]))
            and int(np.argmax(scores[..., blank+1:])) == int(np.argmax(reference_scores[..., blank+1:])))
        good = entry["token_duration_exact"] and good
        for name in ("output_states_1", "output_states_2", "prednet_lengths"):
            metrics = compare_tensors(tensor(expected["expected"][name]), tensor(call["outputs"][name]), atol=1e-5, rtol=1e-4)
            entry["metrics"][name] = metrics
            good = metrics["allclose"] and good
        comparisons.append(entry)
    control = {"output_exact": all(runs[0][name] == runs[1][name] for name in
                                  ("tokenIds", "frameIndices", "tdtSteps", "text")), "state": {}}
    for key in ("h", "c"):
        left, right = tensor(runs[0]["decoderState"][key]), tensor(runs[1]["decoderState"][key])
        metrics = compare_tensors(left, right, atol=1e-5, rtol=1e-4)
        metrics["bitwise_exact"] = left.tobytes() == right.tobytes()
        control["state"][key] = metrics
    good = control["output_exact"] and all(m["bitwise_exact"] for m in control["state"].values()) and good
    return {"status": "LIVE_HEAD_NUMERICAL_PASS" if good else "LIVE_HEAD_NUMERICAL_FAIL",
            "observed_calls": len(calls), "comparisons": comparisons, "observation_control": control,
            "boundary": "PROSPECTIVE_CALL_OUTPUTS_NOT_ALL_COMMITTED", "allclose": bool(good)}
