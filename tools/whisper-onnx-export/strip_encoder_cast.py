"""
Remove the final Cast(f16→f32) from the encoder ONNX graph.

The fp16_iofp32 encoder (keep_io_types=True) has a Cast node at the output
that converts internal fp16 last_hidden_state_fp16 → fp32 last_hidden_state.
By removing this Cast and exposing the fp16 output directly, the encoder
outputs fp16 on GPU — compatible with the original decoder_init (fp16 input)
with zero CPU casts and zero cross-session synchronization.

Usage:
    python strip_encoder_cast.py <encoder_model.onnx> [output_path]
"""

import sys
import os
import shutil
from pathlib import Path
from typing import Optional

import onnx
from onnx import TensorProto, ValueInfoProto, TypeProto


def strip_encoder_output_cast(input_path: str, output_path: Optional[str] = None):
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_suffix("").with_suffix(".fp16out.onnx"))

    print(f"Loading: {input_path}")
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    # Find the Cast node that produces last_hidden_state
    cast_node = None
    for n in graph.node:
        if n.op_type == "Cast" and "last_hidden_state" in n.output:
            cast_node = n
            break

    if cast_node is None:
        print("ERROR: Could not find Cast node producing last_hidden_state")
        return False

    cast_input = list(cast_node.input)[0]   # last_hidden_state_fp16
    cast_output = list(cast_node.output)[0]  # last_hidden_state
    print(f"Found Cast: {cast_input} → {cast_output}")

    # Verify the Cast is f16→f32
    for attr in cast_node.attribute:
        if attr.name == "to":
            print(f"  Cast to type: {attr.i} ({'float32' if attr.i == 1 else 'float16' if attr.i == 10 else '?'})")

    # Remove the Cast node
    nodes = [n for n in graph.node if n != cast_node]
    graph.ClearField("node")
    graph.node.extend(nodes)
    print(f"  Removed Cast node. Nodes: {len(graph.node)} (was {len(graph.node) + 1})")

    # Change the graph output from last_hidden_state (fp32) → cast_input (fp16)
    found = False
    for o in graph.output:
        if o.name == cast_output:
            # Build new output type
            new_type = TypeProto()
            new_type.tensor_type.elem_type = TensorProto.FLOAT16
            for old_dim in o.type.tensor_type.shape.dim:
                new_dim = new_type.tensor_type.shape.dim.add()
                if old_dim.dim_param:
                    new_dim.dim_param = old_dim.dim_param
                elif old_dim.dim_value:
                    new_dim.dim_value = old_dim.dim_value

            new_output = ValueInfoProto()
            new_output.name = cast_input
            new_output.type.CopyFrom(new_type)

            outputs = list(graph.output)
            idx = next(i for i, out in enumerate(outputs) if out.name == cast_output)
            outputs[idx] = new_output
            graph.ClearField("output")
            graph.output.extend(outputs)

            print(f"  Changed output: {cast_output}(fp32) → {cast_input}(fp16)")
            found = True
            break

    if not found:
        print(f"ERROR: Output '{cast_output}' not found in graph outputs")
        return False

    # Also redirect any internal node that references cast_output → cast_input
    ref_count = 0
    for n in graph.node:
        new_ins = list(n.input)
        changed = False
        for j, inp in enumerate(new_ins):
            if inp == cast_output:
                new_ins[j] = cast_input
                ref_count += 1
                changed = True
        if changed:
            n.ClearField("input")
            n.input.extend(new_ins)
    if ref_count > 0:
        print(f"  Redirected {ref_count} internal refs")

    # Save (graph only, preserve external data reference)
    onnx.save(model, output_path)

    # Copy original .data file
    src_data = input_path + ".data"
    dst_data = output_path + ".data"
    if os.path.exists(src_data) and not os.path.exists(dst_data):
        shutil.copy2(src_data, dst_data)
        print(f"  Copied data: {src_data} → {dst_data}")

    # Verify
    verify = onnx.load(output_path, load_external_data=False)
    for o in verify.graph.output:
        print(f"  ✅ Output: {o.name} type={o.type.tensor_type.elem_type}")

    try:
        onnx.checker.check_model(output_path)
        print(f"  ✅ onnx.checker: PASSED")
    except Exception as e:
        print(f"  ⚠️  onnx.checker: {e}")

    print(f"\nSaved: {output_path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    strip_encoder_output_cast(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
