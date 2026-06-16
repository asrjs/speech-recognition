"""
Inject a native ONNX Cast node at the entry of decoder_init.onnx.

Converts the encoder_hidden_states input from FLOAT16 to FLOAT (fp32),
then inserts a Cast(f32→f16) node so the rest of the graph operates
on fp16 as before. This offloads the f32→f16 conversion from CPU
(JS float32ToFloat16Bits loop) to GPU hardware.

Usage:
    python inject_decoder_init_cast.py <decoder_init.onnx> [output_path]

If output_path is omitted, saves as <input>.cast.onnx alongside the original.
The original file is NEVER modified. External data is re-packed.
"""

import sys
import os
from pathlib import Path
from typing import Optional

import onnx
from onnx import helper, TensorProto, ValueInfoProto, TypeProto


ONNX_TYPE_NAMES = {
    1: "float32",
    7: "int64",
    10: "float16",
}


def inject_cast(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    input_name: str = "encoder_hidden_states",
    src_type: int = TensorProto.FLOAT,       # float32
    dst_type: int = TensorProto.FLOAT16,     # float16
):
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_suffix("").with_suffix(".cast.onnx"))

    print(f"Loading: {input_path}")
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    # ------------------------------------------------------------------
    # 1. Find and validate the target input
    # ------------------------------------------------------------------
    orig_input = None
    for inp in graph.input:
        if inp.name == input_name:
            orig_input = inp
            break

    if orig_input is None:
        raise ValueError(f"Input '{input_name}' not found in graph.")

    orig_type = orig_input.type.tensor_type.elem_type
    print(f"  Original {input_name}: type={orig_type} "
          f"({ONNX_TYPE_NAMES.get(orig_type, '?')}), "
          f"dims={[(d.dim_value, d.dim_param) for d in orig_input.type.tensor_type.shape.dim]}")

    if orig_type != dst_type:
        print(f"  WARNING: Expected dst_type={dst_type}, but input is already {orig_type}. "
              f"Proceeding anyway.")

    # ------------------------------------------------------------------
    # 2. Change the graph input type to src_type (fp32)
    # ------------------------------------------------------------------
    new_type = TypeProto()
    new_type.tensor_type.elem_type = src_type
    for old_dim in orig_input.type.tensor_type.shape.dim:
        new_dim = new_type.tensor_type.shape.dim.add()
        if old_dim.dim_param:
            new_dim.dim_param = old_dim.dim_param
        elif old_dim.dim_value:
            new_dim.dim_value = old_dim.dim_value
        # else: unspecified dim — leave empty

    new_input = ValueInfoProto()
    new_input.name = input_name
    new_input.type.CopyFrom(new_type)

    # Replace in graph input list
    input_idx = next(i for i, inp in enumerate(graph.input) if inp.name == input_name)
    inputs = list(graph.input)
    inputs[input_idx] = new_input
    graph.ClearField("input")
    graph.input.extend(inputs)

    print(f"  Changed input type: {orig_type} → {src_type} "
          f"({ONNX_TYPE_NAMES.get(src_type, '?')})")
    print(f"  Preserved dims: {[(d.dim_value, d.dim_param) for d in new_type.tensor_type.shape.dim]}")

    # ------------------------------------------------------------------
    # 3. Remove any pre-existing Cast on this input (idempotent)
    # ------------------------------------------------------------------
    existing_cast_idx = None
    existing_cast_output = None
    for i, node in enumerate(graph.node):
        if node.op_type == "Cast" and list(node.input) == [input_name]:
            existing_cast_idx = i
            existing_cast_output = list(node.output)[0]
            break

    if existing_cast_idx is not None:
        nodes = list(graph.node)
        del nodes[existing_cast_idx]
        # Revert downstream refs back to the original input name
        for node in nodes:
            new_ins = list(node.input)
            changed = False
            for j, inp in enumerate(new_ins):
                if inp == existing_cast_output:
                    new_ins[j] = input_name
                    changed = True
            if changed:
                node.ClearField("input")
                node.input.extend(new_ins)
        graph.ClearField("node")
        graph.node.extend(nodes)
        print(f"  Removed existing Cast node (output={existing_cast_output})")

    # ------------------------------------------------------------------
    # 4. Create and insert the Cast node at position 0
    # ------------------------------------------------------------------
    cast_output = f"{input_name}_f16"
    cast_node = helper.make_node(
        "Cast",
        name=f"Cast_{input_name}_to_f16",
        inputs=[input_name],
        outputs=[cast_output],
        to=int(dst_type),
    )

    # ------------------------------------------------------------------
    # 5. Redirect ALL downstream references from input_name → cast_output
    # ------------------------------------------------------------------
    ref_count = 0
    for node in graph.node:
        if node.name == cast_node.name:
            continue
        new_ins = list(node.input)
        changed = False
        for j, inp in enumerate(new_ins):
            if inp == input_name:
                new_ins[j] = cast_output
                ref_count += 1
                changed = True
        if changed:
            node.ClearField("input")
            node.input.extend(new_ins)

    # Insert Cast at front
    all_nodes = [cast_node] + list(graph.node)
    graph.ClearField("node")
    graph.node.extend(all_nodes)

    print(f"  Cast inserted: {input_name} → {cast_output} (Cast to {ONNX_TYPE_NAMES.get(dst_type, str(dst_type))})")
    print(f"  Downstream refs redirected: {ref_count}")

    # Verify no node still references the original input name (except Cast)
    for node in graph.node:
        if node.name == cast_node.name:
            continue
        if input_name in node.input:
            print(f"  WARNING: Node '{node.name}' still references '{input_name}'")

    # ------------------------------------------------------------------
    # 6. Save (graph only — external data reference preserved from original)
    # ------------------------------------------------------------------
    # We loaded with load_external_data=False, so initializers retain their
    # original external_data.location fields. Save without re-packing so
    # the .onnx file is just the modified graph (~110KB), still pointing
    # to the original .data file.
    onnx.save(model, output_path)

    # Copy the original .data file alongside if not already present
    output_data = output_path + ".data"
    input_data = input_path + ".data"
    if os.path.exists(input_data) and not os.path.exists(output_data):
        import shutil
        shutil.copy2(input_data, output_data)
        print(f"  Copied data: {input_data} → {output_data}")

    # ------------------------------------------------------------------
    # 7. Verify
    # ------------------------------------------------------------------
    verify = onnx.load(output_path, load_external_data=False)
    for inp in verify.graph.input:
        if inp.name == input_name:
            dims = [(d.dim_value, d.dim_param) for d in inp.type.tensor_type.shape.dim]
            print(f"  ✅ Verified input: type={inp.type.tensor_type.elem_type} "
                  f"({ONNX_TYPE_NAMES.get(inp.type.tensor_type.elem_type, '?')}), "
                  f"dims={dims}")

    has_cast = any(
        n.op_type == "Cast"
        and list(n.input) == [input_name]
        and list(n.output) == [cast_output]
        for n in verify.graph.node
    )
    print(f"  ✅ Cast node: {'present' if has_cast else 'MISSING!'}")

    # Quick onnx.checker
    try:
        onnx.checker.check_model(output_path)
        print(f"  ✅ onnx.checker: PASSED")
    except Exception as e:
        print(f"  ⚠️  onnx.checker: {e}")

    # Node count
    print(f"  Nodes: {len(graph.node)} (was {len(graph.node) - 1} + 1 Cast)")
    print(f"\nSaved: {output_path}")
    print(f"Data:  {output_path}.data")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inject_cast(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
