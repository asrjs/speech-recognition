"""Restore symbolic axes that the float16 conversion materialised.

ORT's float16 converter runs shape inference, and on this Qwen3-ASR decoder it
replaced the symbolic past-length axis with the concrete length it inferred.
The session then refuses any other cache length: the step graph is fed a
zero-length past on the first call and an arbitrary length afterwards, so a
frozen axis breaks the decode loop even though the arithmetic is unaffected.

The dtype conversion is the only intended change, so this copies the symbolic
dimension names back from the fp32 source for every matching input and output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx


def _symbolic_dims(value_info) -> dict:
    """Map axis index to dim_param for a sparse tensor type."""
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return {}
    return {
        axis: dim.dim_param
        for axis, dim in enumerate(tensor_type.shape.dim)
        if dim.HasField("dim_param") and dim.dim_param
    }


def _copy_symbolic(source, target) -> list[str]:
    source_dims = {value.name: _symbolic_dims(value) for value in source}
    restored: list[str] = []
    for value in target:
        symbolic = source_dims.get(value.name)
        if not symbolic:
            continue
        shape = value.type.tensor_type.shape
        for axis, name in symbolic.items():
            if axis >= len(shape.dim):
                continue
            dim = shape.dim[axis]
            if dim.HasField("dim_value") or dim.dim_param != name:
                dim.ClearField("dim_value")
                dim.dim_param = name
                restored.append(f"{value.name}[{axis}]={name}")
    return restored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="the fp32 graph")
    parser.add_argument("--converted", required=True, type=Path, help="the float16 graph to repair")
    args = parser.parse_args(argv)

    source = onnx.load(str(args.source), load_external_data=False)
    converted = onnx.load(str(args.converted), load_external_data=False)

    restored: list[str] = []
    for source_section, target_section in (
        (source.graph.input, converted.graph.input),
        (source.graph.output, converted.graph.output),
        (source.graph.value_info, converted.graph.value_info),
    ):
        restored.extend(_copy_symbolic(source_section, target_section))

    if not restored:
        print(f"{args.converted.name}: nothing to restore")
        return 0

    temporary = args.converted.with_suffix(".onnx.repaired")
    onnx.save_model(converted, str(temporary), save_as_external_data=False)
    temporary.replace(args.converted)
    print(f"{args.converted.name}: restored {len(restored)} axes")
    for entry in restored:
        print(f"  {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
