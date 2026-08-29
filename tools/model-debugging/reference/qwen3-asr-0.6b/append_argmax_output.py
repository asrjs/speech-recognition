"""Append a greedy-token output to an explicit-KV Qwen decoder graph.

The graph still computes logits internally, but an ArgMax output lets the
runtime fetch one token id instead of downloading a 151,936-wide logits row.
The input graph and its external-data file are never modified. By default the
logits output is retained for an A/B control; pass ``--remove-logits`` for the
scalar-output candidate used by the browser benchmark.

This is a diagnostic graph-surgery tool, not an assertion that ArgMax is
supported or faster on every ONNX Runtime execution provider. Always compare
the candidate with the original graph for exact tokens, load time, latency,
memory, and output-disposal behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original decoder .onnx graph")
    parser.add_argument("--output", type=Path, required=True, help="Candidate graph path")
    parser.add_argument(
        "--remove-logits",
        action="store_true",
        help="Remove the logits graph output after wiring ArgMax",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON surgery manifest")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symbolic_shape(value_info, *, argmax: bool = False) -> list[int | str | None] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            shape.append(None)
    if argmax and shape:
        shape[-1] = 1
    return shape


def external_locations(model) -> list[str]:
    locations: list[str] = []
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location" and entry.value not in locations:
                locations.append(entry.value)
    return locations


def append_argmax(input_path: Path, output_path: Path, remove_logits: bool) -> dict:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load_model(str(input_path), load_external_data=False)
    graph_outputs = {output.name: output for output in model.graph.output}
    logits = graph_outputs.get("logits")
    if logits is None:
        raise ValueError(f"Expected a graph output named 'logits' in {input_path}")
    if "next_token_id" in graph_outputs:
        raise ValueError(f"Graph already contains a 'next_token_id' output: {input_path}")

    output_name = "next_token_id"
    node_name = "qwen_greedy_argmax"
    model.graph.node.append(
        helper.make_node(
            "ArgMax",
            inputs=["logits"],
            outputs=[output_name],
            name=node_name,
            axis=-1,
            keepdims=1,
        )
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            output_name,
            TensorProto.INT64,
            symbolic_shape(logits, argmax=True),
        )
    )
    if remove_logits:
        kept_outputs = [output for output in model.graph.output if output.name != "logits"]
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # load_external_data=False plus save_as_external_data=False preserves the
    # original external-data locations without copying or rewriting the shard.
    onnx.save_model(model, str(output_path), save_as_external_data=False)
    # Check the serialized file so ONNX can resolve any co-located external
    # shard references. This validates the protobuf without loading the large
    # initializer payload into Python memory.
    onnx.checker.check_model(str(output_path), full_check=False)

    result = {
        "schema": "asrjs.qwen.append-argmax.v1",
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
        "input_size_bytes": input_path.stat().st_size,
        "output_size_bytes": output_path.stat().st_size,
        "remove_logits": remove_logits,
        "node": {"name": node_name, "op_type": "ArgMax", "axis": -1, "keepdims": 1},
        "external_data_locations": external_locations(model),
        "graph_outputs": [output.name for output in model.graph.output],
        "opset_import": [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in model.opset_import
        ],
    }
    return result


def main() -> None:
    args = parse_args()
    result = append_argmax(args.input, args.output, args.remove_logits)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
