"""Prepare a diagnostic frozen encoder boundary, never a model export.

This lets a later library run isolate recurrent decoding without duplicating its
decoder or trusting the currently failing feature path. Inputs are deliberately
ignored. Native checks below certify only exact frozen tensor delivery, NOT the
real encoder, features, decoder, browser, or end-to-end ASR.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from browser_head_replay import make_fixture
from export_nemo_fp32 import ENCODER_INPUTS, ENCODER_OUTPUTS, sha256
from validate_stage_replay import read_json, stage

SCOPE = "DIAGNOSTIC_FROZEN_ENCODER_BOUNDARY_ONLY"


def require_contract(values, expected):
    if len(values) != len(expected) or {value.name for value in values} != set(expected):
        raise ValueError("Encoder boundary names differ")
    for value in values:
        tensor = value.type.tensor_type
        dtype, rank = expected[value.name]
        if onnx.TensorProto.DataType.Name(tensor.elem_type) != dtype or len(tensor.shape.dim) != rank:
            raise ValueError("Encoder boundary dtype/rank differs")


def require_array(value, contract):
    tensor = contract.type.tensor_type
    expected_dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type))
    if value.dtype != expected_dtype or value.ndim != len(tensor.shape.dim) or not np.isfinite(value).all():
        raise ValueError("Frozen boundary tensor dtype/rank/finiteness differs")
    for size, dim in zip(value.shape, tensor.shape.dim):
        if size < 1 or (dim.HasField("dim_value") and dim.dim_value != size):
            raise ValueError("Frozen boundary tensor static shape differs")


def make_graph(source, stages, cfg, sample_id):
    require_contract(source.graph.input, ENCODER_INPUTS)
    require_contract(source.graph.output, ENCODER_OUTPUTS)
    features = stage(stages, "features", "float32")
    feature_length = stage(stages, "features_length", "int64", (1,))
    encoder = stage(stages, "encoder", "float32")
    encoded_length = stage(stages, "encoder_length", "int64", (1,))
    if (features.ndim != 3 or features.shape[:2] != (1, cfg["preprocessor"]["features"])
            or encoder.ndim != 3 or encoder.shape[:2] != (1, cfg["joint"]["jointnet"]["encoder_hidden"])
            or not 0 < feature_length[0] <= features.shape[2]
            or not 0 < encoded_length[0] <= encoder.shape[2]
            or not isinstance(sample_id, str) or not sample_id):
        raise ValueError("Only nonempty batch-one offline BDT captures are supported")
    feeds = {"audio_signal": features, "length": feature_length}
    outputs = {"outputs": encoder, "encoded_lengths": encoded_length}
    for contract in source.graph.input:
        require_array(feeds[contract.name], contract)
    for contract in source.graph.output:
        require_array(outputs[contract.name], contract)
    nodes = [helper.make_node("Constant", [], [name],
                             value=numpy_helper.from_array(value, name="frozen_" + name))
             for name, value in outputs.items()]
    graph = helper.make_graph(nodes, "diagnostic_frozen_encoder_not_model",
                              copy.deepcopy(list(source.graph.input)), copy.deepcopy(list(source.graph.output)))
    model = helper.make_model(graph, producer_name="asrjs-diagnostic-frozen-boundary",
                              opset_imports=copy.deepcopy(list(source.opset_import)))
    model.ir_version = source.ir_version
    helper.set_model_props(model, {"asrjs.scope": SCOPE, "asrjs.sample_id": sample_id,
                                  "asrjs.inputs": "IGNORED", "asrjs.model_acceptance": "NOT_PROVEN"})
    onnx.checker.check_model(model)
    return model, feeds, outputs


def verify_delivery(model_bytes, feeds, expected):
    """Run real ORT twice with different feature values; require bitwise delivery."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
    names = [value.name for value in session.get_outputs()]
    checks = []
    for label, inputs in (("captured_features", feeds),
                          ("zero_features", {**feeds, "audio_signal": np.zeros_like(feeds["audio_signal"])})):
        actual = dict(zip(names, session.run(None, inputs)))
        check = {"input": label, "outputs": {name: {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "bitwise_exact": (value.dtype == expected[name].dtype and value.shape == expected[name].shape
                              and value.tobytes() == expected[name].tobytes()),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            for name, value in actual.items()}}
        if set(actual) != set(expected) or not all(item["bitwise_exact"] for item in check["outputs"].values()):
            raise ValueError("Frozen boundary native delivery differs")
        checks.append(check)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "summary", "audio", "artifact-dir", "export-receipt", "nemo", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    exported, packed = read_json(args.export_receipt), read_json(args.artifact_dir / "repack-receipt.json")
    graph_path, config_path = args.artifact_dir / "encoder-model.onnx", args.artifact_dir / "source-config.json"
    config_binding = next(item for item in packed["companions"] if item["name"] == config_path.name)
    if (packed["status"] != "REPACKED_BYTE_IDENTICAL_UNVALIDATED"
            or packed["source_graph"]["sha256"] != exported["encoder"]["sha256"]
            or sha256(graph_path) != packed["graph"]["sha256"]
            or sha256(config_path) != config_binding["sha256"]
            or config_path.stat().st_size != config_binding["bytes"]
            or sha256(args.nemo) != exported["source"]["sha256"]):
        raise ValueError("Original weight/export/packed graph/config lineage differs")
    reference, summary, cfg = read_json(args.capture), read_json(args.summary), read_json(config_path)
    identity = {"model": exported["model_id"], "model_revision": exported["model_revision"],
                "weight_sha256": exported["source"]["sha256"]}
    fixture = make_fixture(reference, summary, cfg, identity, sha256(args.audio), all_steps=True)
    sample = reference["samples"][0]
    source = onnx.load(str(graph_path), load_external_data=False)
    model, feeds, expected = make_graph(source, sample["stages"], cfg, fixture["sample_id"])
    payload = model.SerializeToString()
    checks = verify_delivery(payload, feeds, expected)
    receipt = {"schema_version": 1, "scope": SCOPE, "status": "FROZEN_BOUNDARY_NATIVE_DELIVERY_PASS",
               "sample_id": fixture["sample_id"], "identity": identity, "inputs_ignored": True,
               "real_features": "BYPASSED_UNPROVEN", "real_encoder": "BYPASSED_UNPROVEN",
               "decoder_recurrence": "UNTESTED", "browser": "UNTESTED", "model_acceptance": "NOT_PROVEN",
               "graph_sha256": hashlib.sha256(payload).hexdigest(), "graph_bytes": len(payload),
               "source_graph_sha256": sha256(graph_path), "config_sha256": sha256(config_path),
               "capture_sha256": sha256(args.capture), "summary_sha256": sha256(args.summary),
               "audio_sha256": sha256(args.audio), "export_receipt_sha256": sha256(args.export_receipt),
               "repack_receipt_sha256": sha256(args.artifact_dir / "repack-receipt.json"),
               "tool_sha256": sha256(Path(__file__)), "numpy": np.__version__, "onnx": onnx.__version__,
               "onnxruntime": ort.__version__, "checks": checks}
    # Prerequisites and native controls all pass before any artifact is written.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "diagnostic-frozen-encoder.onnx").open("xb") as output:
        output.write(payload)
    with (args.output_dir / "frozen-boundary-receipt.json").open("x", encoding="utf-8", newline="\n") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
