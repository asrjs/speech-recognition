#!/usr/bin/env python
"""Split selected wide ONNX Concat nodes into ordered bounded-input trees.

This is an offline candidate generator. It never overwrites the source graph;
external initializer data must be copied beside the destination first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import helper


def split_variadic_concat_nodes(
    model: onnx.ModelProto,
    *,
    node_names: tuple[str, ...],
    max_inputs: int = 6,
) -> dict[str, int]:
    """Rewrite named Concat nodes, preserving input order and final output names."""
    if max_inputs < 2:
        raise ValueError("max_inputs must be at least 2")
    if not node_names or len(set(node_names)) != len(node_names):
        raise ValueError("provide one or more unique Concat node names")
    wanted = set(node_names)
    found: set[str] = set()
    used_values = {
        value
        for node in model.graph.node
        for value in (*node.input, *node.output)
        if value
    }
    rewritten: list[onnx.NodeProto] = []
    counts: dict[str, int] = {}

    for node in model.graph.node:
        if node.name not in wanted:
            rewritten.append(node)
            continue
        if node.op_type != "Concat":
            raise ValueError(f"target {node.name!r} is {node.op_type}, not Concat")
        if node.name in found:
            raise ValueError(f"duplicate target node name {node.name!r}")
        found.add(node.name)
        if len(node.output) != 1:
            raise ValueError(f"target {node.name!r} must have exactly one output")
        if len(node.input) <= max_inputs:
            raise ValueError(
                f"target {node.name!r} has {len(node.input)} inputs; expected > {max_inputs}"
            )

        attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        axis = attrs.get("axis")
        if not isinstance(axis, int):
            raise ValueError(f"target {node.name!r} has no integer axis attribute")

        current = list(node.input)
        level = 0
        generated: list[onnx.NodeProto] = []
        while len(current) > max_inputs:
            next_level: list[str] = []
            for group_index, start in enumerate(range(0, len(current), max_inputs)):
                group = current[start : start + max_inputs]
                if len(group) == 1:
                    next_level.append(group[0])
                    continue
                output_name = f"{node.name}__tree_{level}_{group_index}"
                while output_name in used_values:
                    output_name += "_"
                used_values.add(output_name)
                generated.append(
                    helper.make_node(
                        "Concat",
                        group,
                        [output_name],
                        name=f"{node.name}__tree_node_{level}_{group_index}",
                        axis=axis,
                        domain=node.domain,
                    )
                )
                next_level.append(output_name)
            current = next_level
            level += 1

        generated.append(
            helper.make_node(
                "Concat",
                current,
                list(node.output),
                name=node.name,
                axis=axis,
                domain=node.domain,
            )
        )
        generated[-1].doc_string = node.doc_string
        rewritten.extend(generated)
        counts[node.name] = len(generated)

    missing = wanted - found
    if missing:
        raise ValueError(f"target Concat nodes not found: {', '.join(sorted(missing))}")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return counts


def move_late_producers_before_consumers(model: onnx.ModelProto) -> list[str]:
    """Repair only top-level producer-order violations, preserving other order."""
    nodes = list(model.graph.node)
    moved: list[str] = []
    for _ in range(len(nodes) + 1):
        producer: dict[str, int] = {}
        for index, node in enumerate(nodes):
            for output in node.output:
                if output in producer:
                    raise ValueError(f"multiple graph producers for value {output!r}")
                producer[output] = index
        violations = [
            (consumer_index, producer[name], name)
            for consumer_index, node in enumerate(nodes)
            for name in node.input
            if name and name in producer and producer[name] > consumer_index
        ]
        if not violations:
            del model.graph.node[:]
            model.graph.node.extend(nodes)
            return moved
        consumer_index, producer_index, value_name = min(violations)
        producer_node = nodes.pop(producer_index)
        nodes.insert(consumer_index, producer_node)
        moved.append(f"{producer_node.name}:{value_name}")
    raise ValueError("could not repair graph topological order without a dependency cycle")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--node", action="append", dest="nodes", required=True)
    parser.add_argument("--max-inputs", type=int, default=6)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("refusing to overwrite the input graph")
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing candidate graph")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.input), load_external_data=False)
    counts = split_variadic_concat_nodes(
        model,
        node_names=tuple(args.nodes),
        max_inputs=args.max_inputs,
    )
    moved_nodes = move_late_producers_before_consumers(model)
    onnx.save_model(model, str(args.output))
    # Check by path so the checker resolves unchanged external tensor data
    # beside the copied candidate graph, not relative to the caller's cwd.
    onnx.checker.check_model(str(args.output))
    print(json.dumps({
        "output": str(args.output),
        "splitNodes": counts,
        "maxConcatInputs": max(
            len(node.input) for node in model.graph.node if node.op_type == "Concat"
        ),
        "movedLateProducers": moved_nodes,
    }))


if __name__ == "__main__":
    main()
