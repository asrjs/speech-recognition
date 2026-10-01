"""Replay recorded official CPU NeMo TDT calls through native ORT, not a decoder.

Encoder uses captured features. Fused heads use captured raw encoder frames and
predictor labels/input states, isolating head export from accumulated ASR search.
This certifies only recorded boundaries/decisions, not end-to-end tokens/text.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repack_onnx_external_data import range_hash
from tensor_parity import compare_tensors
from export_nemo_fp32 import audit_graph, ENCODER_INPUTS, ENCODER_OUTPUTS, DECODER_INPUTS, DECODER_OUTPUTS

CONTROL_KEYS = {"baseline_tokens_match", "baseline_text_match", "baseline_frames_match",
                "after_tokens_match", "after_text_match", "after_durations_match"}


def read_json(path: Path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")) as stream:
        return json.load(stream)


def stage(stages, name, dtype=None, shape=None):
    value = stages[name]
    if value["dtype"] not in {"float32", "int64"} or (dtype and value["dtype"] != dtype):
        raise ValueError(f"Unexpected reference dtype: {name}")
    dims = value["shape"]
    if not dims or any(type(dim) is not int or dim < 1 for dim in dims):
        raise ValueError(f"Empty/invalid reference shape: {name}")
    if len(value["data"]) != np.prod(dims) or (shape and list(shape) != dims):
        raise ValueError(f"Reference shape/data count mismatch: {name}")
    if value["dtype"] == "int64" and any(type(item) is not int for item in value["data"]):
        raise ValueError(f"Noninteger int64 reference data: {name}")
    result = np.asarray(value["data"], dtype=value["dtype"]).reshape(dims)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite reference stage: {name}")
    return result


def log_softmax(value):
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Finite fp32 scores required")
    precise = value.astype(np.float64)
    shifted = precise - precise.max(axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float32)


def unique_frame(projected, row):
    if projected.ndim != 3 or projected.shape[0] != 1 or row.shape != (1, 1, projected.shape[2]):
        raise ValueError("Unexpected projected frame contract")
    matches = np.flatnonzero(np.all(projected[0] == row[0, 0], axis=1))
    if matches.size != 1:
        raise ValueError("Joint frame does not uniquely match a captured encoder projection")
    return int(matches[0])


def replay_bindings(sample, cfg):
    stages = sample["stages"]
    raw = stage(stages, "encoder", "float32")
    if raw.ndim != 3 or raw.shape[:2] != (1, cfg["joint"]["jointnet"]["encoder_hidden"]):
        raise ValueError("Unexpected raw encoder contract")
    projected = stage(stages, "encoder_projection/0000/output", "float32",
                      (1, raw.shape[2], cfg["joint"]["jointnet"]["joint_hidden"]))
    if not np.array_equal(raw.transpose(0, 2, 1), stage(stages, "encoder_projection/0000/input_encoder_output", "float32")):
        raise ValueError("Captured encoder/projection input differs")
    blank = cfg["decoder"]["vocab_size"]
    state_shape = (cfg["decoder"]["prednet"]["pred_rnn_layers"], 1, cfg["decoder"]["prednet"]["pred_hidden"])
    latest = latest_projection = None
    counts = {kind: 0 for kind in ("encoder_projection", "predictor", "predictor_projection", "joint")}
    bindings, previous_frame = [], -1
    for call in sample["calls"]:
        kind, index, prefix = call["kind"], call["index"], call["stage_prefix"]
        if kind not in counts or index != counts[kind] or prefix != f"{kind}/{index:04d}":
            raise ValueError("Noncanonical/duplicate call chronology")
        counts[kind] += 1
        if kind == "encoder_projection":
            if index != 0 or latest is not None:
                raise ValueError("Only one offline encoder projection is supported")
        elif kind == "predictor":
            if not counts["encoder_projection"] or call["labels_missing"] or call["state_missing"] or call["add_sos"]:
                raise ValueError("Unsupported predictor SOS/state contract")
            labels = stage(stages, prefix + "/labels", "int64", (1, 1))
            if not np.all((labels >= 0) & (labels <= blank)) or blank > np.iinfo(np.int32).max:
                raise ValueError("Label outside blank/vocabulary or int32 range")
            latest, latest_projection = prefix, None
            for suffix in ("input_state_h", "input_state_c", "output_state_h", "output_state_c"):
                stage(stages, prefix + "/" + suffix, "float32", state_shape)
        elif kind == "predictor_projection":
            if latest != f"predictor/{index:04d}" or latest_projection is not None:
                raise ValueError("Predictor projection/call association differs")
            if not np.array_equal(stage(stages, latest + "/output", "float32"),
                                  stage(stages, prefix + "/input_prednet_output", "float32")):
                raise ValueError("Captured predictor/projection input differs")
            latest_projection = prefix
        else:
            if latest is None or latest_projection is None:
                raise ValueError("Joint has no preceding predictor/projection")
            if not np.array_equal(stage(stages, prefix + "/input_g", "float32"),
                                  stage(stages, latest_projection + "/output", "float32")):
                raise ValueError("Joint/predictor projection input differs")
            frame = unique_frame(projected, stage(stages, prefix + "/input_f", "float32"))
            if frame < previous_frame:
                raise ValueError("Joint frames are not monotonic")
            previous_frame = frame
            bindings.append({"joint": prefix, "predictor": latest, "frame": frame})
    if not bindings or counts["predictor"] != counts["predictor_projection"] or counts["encoder_projection"] != 1:
        raise ValueError("Incomplete call capture")
    return bindings, counts


def validate_case(capture, summary, audio, cfg, identity, sessions, atol, rtol):
    reference, control = read_json(capture), read_json(summary)
    if reference["schema_version"] != 1 or len(reference["samples"]) != 1:
        raise ValueError("Only single-item schema-v1 captures supported")
    for key in ("model", "model_revision", "weight_sha256"):
        if reference["meta"][key] != identity[key] or control["meta"][key] != identity[key]:
            raise ValueError(f"Reference source identity mismatch: {key}")
    for item in (reference, control["stage_capture"]):
        flags = item["instrumentation_control"]
        if set(flags) != CONTROL_KEYS or any(value is not True for value in flags.values()):
            raise ValueError("Official instrumentation controls did not all pass")
    sample = reference["samples"][0]
    audio_hash = range_hash(audio)
    if any(digest != audio_hash for digest in (reference["meta"]["audio_sha256"], control["meta"]["audio_sha256"], sample["audio"]["sha256"])):
        raise ValueError("Reference audio identity mismatch")
    if reference["meta"]["device"] != "cpu" or reference["meta"]["batch_size"] != 1:
        raise ValueError("CPU batch-one capture required")
    if (reference["stage_order"] != list(sample["stages"]) or control["stage_capture"]["stage_count"] != len(sample["stages"])
            or sample["tokens"] != control["result"]["token_ids"] or sample["transcript"] != control["result"]["text"]
            or sample["token_frames"] != control["result"]["timestamp"] or sample["token_duration"] != control["result"]["token_duration"]):
        raise ValueError("Capture/summary structure or official outputs differ")
    bindings, counts = replay_bindings(sample, cfg)
    if counts != control["stage_capture"]["call_counts"]:
        raise ValueError("Capture call counts differ")
    stages = sample["stages"]
    result = {"sample_id": sample["sample_id"], "audio_sha256": audio_hash,
              "capture_sha256": range_hash(capture), "status": "REPLAY_IN_PROGRESS", "comparisons": {}}

    def compare(name, left, right):
        result["comparisons"][name] = compare_tensors(left, right, atol=atol, rtol=rtol)
        return result["comparisons"][name]["allclose"]

    outputs = sessions[0].run(None, {"audio_signal": stage(stages, "features", "float32"),
                                     "length": stage(stages, "features_length", "int64", (1,))})
    encoder = dict(zip([item.name for item in sessions[0].get_outputs()], outputs))
    good = compare("encoder", stage(stages, "encoder", "float32"), encoder["outputs"])
    good = compare("encoder_length", stage(stages, "encoder_length", "int64", (1,)), encoder["encoded_lengths"]) and good
    if not good:
        result["status"] = "ENCODER_NUMERICAL_FAIL_HEAD_UNTESTED"
        return result
    channels = cfg["joint"]["num_classes"] + 1 + cfg["joint"]["num_extra_outputs"]
    blank = cfg["decoder"]["vocab_size"]
    for binding in bindings:
        joint, predictor, frame = binding["joint"], binding["predictor"], binding["frame"]
        feeds = {"encoder_outputs": np.ascontiguousarray(stage(stages, "encoder", "float32")[:, :, frame:frame+1]),
                 "targets": stage(stages, predictor + "/labels", "int64", (1, 1)).astype(np.int32),
                 "target_length": np.ones(1, dtype=np.int32),
                 "input_states_1": stage(stages, predictor + "/input_state_h", "float32"),
                 "input_states_2": stage(stages, predictor + "/input_state_c", "float32")}
        output = dict(zip([item.name for item in sessions[1].get_outputs()], sessions[1].run(None, feeds)))
        reference_scores = stage(stages, joint + "/output", "float32", (1, 1, 1, channels))
        if output["outputs"].shape != reference_scores.shape or not np.allclose(log_softmax(reference_scores), reference_scores, atol=1e-5, rtol=0):
            raise ValueError("Joint channels/official all-channel score normalization differs")
        scores = log_softmax(output["outputs"])
        good = compare(joint + "/normalized_scores", reference_scores, scores) and good
        for number, suffix in ((1, "h"), (2, "c")):
            good = compare(joint + "/predictor_state_" + suffix, stage(stages, predictor + "/output_state_" + suffix, "float32"), output[f"output_states_{number}"]) and good
        good = compare(joint + "/predictor_length", np.ones(1, dtype=np.int32), output["prednet_lengths"]) and good
        decisions_equal = all(np.array_equal(np.argmax(reference_scores[..., section], axis=-1), np.argmax(scores[..., section], axis=-1))
                              for section in (slice(0, blank+1), slice(blank+1, channels)))
        result["comparisons"][joint + "/token_duration_argmax"] = {"exact": decisions_equal, "frame": frame, "predictor": predictor}
        good = decisions_equal and good
    result.update({"status": "RECORDED_BOUNDARIES_PASS" if good else "HEAD_NUMERICAL_FAIL",
                   "joint_calls_replayed": len(bindings), "predictor_calls": counts["predictor"],
                   "predictor_raw_output": "NOT_EXPOSED", "end_to_end_tokens_text": "UNTESTED"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-dir", "source-export-receipt", "nemo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model-id", "model-revision", "expected-weight-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("capture", "summary", "audio"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.cpu_threads < 1 or not len(args.capture) == len(args.summary) == len(args.audio):
        raise ValueError("Positive threads and matching capture/summary/audio lists required")
    receipt = {"schema_version": 1, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "status": "NATIVE_REPLAY_IN_PROGRESS", "engine": "onnxruntime-python", "ort_version": ort.__version__,
               "providers": ["CPUExecutionProvider"], "cpu_threads": args.cpu_threads,
               "atol": args.atol, "rtol": args.rtol, "cases": [], "library_parity": "UNTESTED", "browser_parity": "UNTESTED"}
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            artifacts = args.artifact_dir.resolve()
            packed = read_json(artifacts / "repack-receipt.json")
            exported = read_json(args.source_export_receipt)
            identity = {"model": args.model_id, "model_revision": args.model_revision, "weight_sha256": args.expected_weight_sha256}
            if (exported["source"]["sha256"] != args.expected_weight_sha256 or range_hash(args.nemo) != args.expected_weight_sha256
                    or exported["model_id"] != args.model_id or exported["model_revision"] != args.model_revision
                    or packed["status"] != "REPACKED_BYTE_IDENTICAL_UNVALIDATED"
                    or packed["source_graph"]["sha256"] != exported["encoder"]["sha256"]):
                raise ValueError("Original weight/export/repack source identity mismatch")
            for item in [packed["graph"], packed["external_data"], *packed["companions"]]:
                name = item["name"]
                path = (artifacts / name).resolve()
                if Path(name).name != name or path.parent != artifacts or path.stat().st_size != item["bytes"] or range_hash(path) != item["sha256"]:
                    raise ValueError("Packed artifact identity mismatch")
            cfg = read_json(artifacts / "source-config.json")
            if (cfg["loss"]["loss_name"] != "tdt" or not cfg["decoder"]["blank_as_pad"]
                    or cfg["decoder"]["vocab_size"] != cfg["joint"]["num_classes"]
                    or cfg["joint"]["log_softmax"] is not None or cfg["joint"]["num_extra_outputs"] != len(cfg["decoding"]["durations"])):
                raise ValueError("Unsupported source TDT/CPU normalized-score contract")
            receipt.update({"identity": identity, "score_boundary": "official CPU log_softmax(null); exported raw; normalize across all token+blank+duration channels",
                            "repack_receipt_sha256": range_hash(artifacts / "repack-receipt.json")})
            receipt["encoder_artifact"] = audit_graph(artifacts / "encoder-model.onnx", ENCODER_INPUTS, ENCODER_OUTPUTS)
            receipt["decoder_artifact"] = audit_graph(artifacts / "decoder_joint-model.onnx", DECODER_INPUTS, DECODER_OUTPUTS)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = args.cpu_threads, 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessions = [ort.InferenceSession(str(artifacts / name), sess_options=options, providers=["CPUExecutionProvider"])
                        for name in ("encoder-model.onnx", "decoder_joint-model.onnx")]
            if any(session.get_providers() != ["CPUExecutionProvider"] for session in sessions):
                raise ValueError("Unexpected actual native providers")
            for capture, summary, audio in zip(args.capture, args.summary, args.audio):
                print(f"Replaying official captured calls: {capture.name}", flush=True)
                result = validate_case(capture, summary, audio, cfg, identity, sessions, args.atol, args.rtol)
                receipt["cases"].append(result)
                print(json.dumps({key: result[key] for key in ("sample_id", "status")}), flush=True)
            receipt["status"] = "RECORDED_BOUNDARIES_PASS" if all(case["status"] == "RECORDED_BOUNDARIES_PASS" for case in receipt["cases"]) else "NUMERICAL_FAIL"
            if receipt["status"] == "NUMERICAL_FAIL":
                raise RuntimeError("Native numerical mismatch; see first divergent metrics in receipt")
        except Exception as error:
            if receipt["status"] != "NUMERICAL_FAIL":
                receipt["status"] = "NATIVE_REPLAY_FAILED"
            receipt["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
