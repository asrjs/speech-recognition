#!/usr/bin/env python
"""Build an isolated Nemotron encoder bundle with conditional first-window drop."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import onnx
from onnx import helper, numpy_helper


BUNDLE_FORMAT = "asrjs.nemotron-rnnt"
BUNDLE_SCHEMA_VERSION = 1
FIRST_CHUNK_FRAMES = 49
SHIFT_FRAMES = 56
LEFT_CONTEXT_FRAMES = 9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_constant(model: onnx.ModelProto, value_name: str) -> Any:
    for initializer in model.graph.initializer:
        if initializer.name == value_name:
            value = numpy_helper.to_array(initializer)
            return value.item() if value.ndim == 0 else value.tolist()
    for node in model.graph.node:
        if node.op_type != "Constant" or value_name not in node.output:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                value = numpy_helper.to_array(attribute.t)
                return value.item() if value.ndim == 0 else value.tolist()
            if attribute.name == "value_int":
                return attribute.i
            if attribute.name == "value_ints":
                return list(attribute.ints)
    raise ValueError(f"expected {value_name!r} to be a static integer constant")


def _target_node(model: onnx.ModelProto, name: str, op_type: str, output: str) -> onnx.NodeProto:
    matches = [
        node
        for node in model.graph.node
        if node.name == name and node.op_type == op_type and output in node.output
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {op_type} node named {name!r} producing {output!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _external_locations(model: onnx.ModelProto) -> list[str]:
    locations = {
        item.value
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        for item in initializer.external_data
        if item.key == "location"
    }
    return sorted(locations)


def _validate_external_location(location: str) -> Path:
    relative = PurePosixPath(location)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe ONNX external-data location: {location!r}")
    return Path(*relative.parts)


def _verify_only_target_inputs_changed(
    original: onnx.ModelProto,
    candidate: onnx.ModelProto,
    slice_name: str,
    sub_name: str,
) -> None:
    if [value.name for value in original.graph.input] != [value.name for value in candidate.graph.input]:
        raise ValueError("candidate changed graph input names")
    if [value.name for value in original.graph.output] != [value.name for value in candidate.graph.output]:
        raise ValueError("candidate changed graph output names")

    candidate_nodes = [node for node in candidate.graph.node if not node.name.startswith("asrjs_conditional_drop_")]
    if len(candidate_nodes) != len(original.graph.node):
        raise ValueError("candidate changed the number of original graph nodes")
    for before, after in zip(original.graph.node, candidate_nodes, strict=True):
        if before.name in {slice_name, sub_name}:
            expected = onnx.NodeProto()
            expected.CopyFrom(before)
            expected.input[1] = after.input[1]
            if expected.SerializeToString() != after.SerializeToString():
                raise ValueError(f"candidate changed more than the selected input of {before.name!r}")
        elif before.SerializeToString() != after.SerializeToString():
            raise ValueError(f"candidate modified unrelated ONNX node {before.name!r}")

    candidate_initializers = {item.name: item for item in candidate.graph.initializer}
    for before in original.graph.initializer:
        after = candidate_initializers.get(before.name)
        if after is None or before.SerializeToString() != after.SerializeToString():
            raise ValueError(f"candidate changed original initializer {before.name!r}")


def build_candidate(input_encoder: Path, output_dir: Path) -> dict[str, Any]:
    source_graph = input_encoder.resolve(strict=True)
    destination = output_dir.resolve()
    if destination == source_graph.parent or source_graph.is_relative_to(destination):
        raise ValueError("output directory must be separate from the source model directory")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate directory: {destination}")

    source_graph_sha = sha256(source_graph)
    model = onnx.load(str(source_graph), load_external_data=False)
    original = onnx.ModelProto()
    original.CopyFrom(model)

    graph_inputs = {value.name for value in model.graph.input}
    if "cache_last_channel_len" not in graph_inputs:
        raise ValueError("source encoder has no cache_last_channel_len input")

    slice_node = _target_node(model, "/Slice", "Slice", "/Slice_output_0")
    if not slice_node.input or slice_node.input[0] != "/pre_encode/out/Add_output_0":
        raise ValueError("source /Slice is not bound to the expected pre-encoded tensor")
    if _as_constant(model, slice_node.input[1]) != [2]:
        raise ValueError("source /Slice start is no longer the expected static [2]")

    sub_node = _target_node(model, "/Sub", "Sub", "/Sub_output_0")
    if _as_constant(model, sub_node.input[1]) != 2:
        raise ValueError("source encoded_lengths subtraction is no longer static 2")

    generated_names = {
        "asrjs_conditional_drop_zero_scalar",
        "asrjs_conditional_drop_zero",
        "asrjs_conditional_drop_two",
        "asrjs_conditional_drop_cache_empty",
        "asrjs_conditional_drop_value",
        "asrjs_conditional_drop_cache_test",
        "asrjs_conditional_drop_select",
    }
    existing_names = {
        value
        for node in model.graph.node
        for value in (*node.input, *node.output, node.name)
        if value
    } | {initializer.name for initializer in model.graph.initializer}
    collision = generated_names & existing_names
    if collision:
        raise ValueError(f"source graph already uses surgery identifiers: {sorted(collision)}")

    slice_node.input[1] = "asrjs_conditional_drop_value"
    sub_node.input[1] = "asrjs_conditional_drop_value"
    model.graph.initializer.extend(
        [
            helper.make_tensor("asrjs_conditional_drop_zero_scalar", onnx.TensorProto.INT64, [], [0]),
            helper.make_tensor("asrjs_conditional_drop_zero", onnx.TensorProto.INT64, [1], [0]),
            helper.make_tensor("asrjs_conditional_drop_two", onnx.TensorProto.INT64, [1], [2]),
        ]
    )
    dynamic_nodes = [
        helper.make_node(
            "Equal",
            ["cache_last_channel_len", "asrjs_conditional_drop_zero_scalar"],
            ["asrjs_conditional_drop_cache_empty"],
            name="asrjs_conditional_drop_cache_test",
        ),
        helper.make_node(
            "Where",
            [
                "asrjs_conditional_drop_cache_empty",
                "asrjs_conditional_drop_zero",
                "asrjs_conditional_drop_two",
            ],
            ["asrjs_conditional_drop_value"],
            name="asrjs_conditional_drop_select",
        ),
    ]
    original_nodes = list(model.graph.node)
    model.graph.ClearField("node")
    model.graph.node.extend([*dynamic_nodes, *original_nodes])
    _verify_only_target_inputs_changed(original, model, "/Slice", "/Sub")

    external_files: list[dict[str, Any]] = []
    for location in _external_locations(model):
        relative = _validate_external_location(location)
        source_data = source_graph.parent / relative
        if not source_data.is_file():
            raise FileNotFoundError(f"missing source ONNX external data: {source_data}")
        external_files.append(
            {
                "location": relative,
                "source": source_data,
                "sha256": sha256(source_data),
                "sizeBytes": source_data.stat().st_size,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    candidate_graph = destination / "encoder.onnx"
    onnx.save_model(model, str(candidate_graph))
    copied_external_files = []
    for external in external_files:
        target = destination / external["location"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(external["source"], target)
        copied_sha = sha256(target)
        if copied_sha != external["sha256"]:
            raise ValueError(f"copied external data hash mismatch for {external['location']}")
        copied_external_files.append(
            {
                "location": external["location"].as_posix(),
                "sourceSha256": external["sha256"],
                "candidateSha256": copied_sha,
                "sizeBytes": external["sizeBytes"],
                "byteIdentical": True,
            }
        )

    onnx.checker.check_model(str(candidate_graph), full_check=False)
    if sha256(source_graph) != source_graph_sha:
        raise ValueError("source graph changed during candidate generation")
    _verify_only_target_inputs_changed(original, onnx.load(str(candidate_graph), load_external_data=False), "/Slice", "/Sub")

    bundle_config = {
        "format": BUNDLE_FORMAT,
        "schemaVersion": BUNDLE_SCHEMA_VERSION,
        "family": "nemotron-rnnt",
        "encoderWindowing": {
            "firstChunkFrames": FIRST_CHUNK_FRAMES,
            "shiftFrames": SHIFT_FRAMES,
            "leftContextFrames": LEFT_CONTEXT_FRAMES,
        },
    }
    config_path = destination / "config.json"
    config_path.write_text(json.dumps(bundle_config, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_GRAPH_CHECKED_NO_NUMERICAL_CLAIM",
        "sourceGraph": str(source_graph),
        "sourceGraphSha256": source_graph_sha,
        "candidateGraph": str(candidate_graph),
        "candidateGraphSha256": sha256(candidate_graph),
        "externalData": copied_external_files,
        "conditionalDrop": "cache_last_channel_len == 0 => [0], otherwise [2]",
        "modifiedOriginalInputs": {
            "/Slice": {"input": 1, "from": [2], "to": "asrjs_conditional_drop_value"},
            "/Sub": {"input": 1, "from": 2, "to": "asrjs_conditional_drop_value"},
        },
        "otherOriginalNodesModified": False,
        "originalInitializersBytePreserved": True,
        "onnxChecker": "PASS_PATH_CHECK",
        "bundleConfig": str(config_path),
        "scope": "isolated offline candidate only; source model graph and external weights are never overwritten",
    }
    (destination / "build-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-encoder", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_candidate(args.input_encoder, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
