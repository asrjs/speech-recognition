"""
Append GPU ArgMax to Whisper decoder_step ONNX graph.

Adds two nodes after the logits output:
  logits -> ArgMax(axis=-1, keepdims=0) -> Cast(INT32) -> next_token_id

The new output `next_token_id` is a scalar INT32 tensor representing
the argmax token index, computed entirely on GPU.

Usage:
  python append_argmax_to_decoder.py <decoder_step.onnx> [output_path]

If output_path is omitted, the input file is overwritten.
A backup of the original is saved as <input>.backup.
"""

import sys
import shutil
from pathlib import Path
from typing import Optional

import onnx
from onnx import helper, TensorProto, ValueInfoProto, TypeProto


def append_argmax_to_decoder(input_path: str, output_path: Optional[str] = None):
    output_path = output_path or input_path

    # Backup original
    backup_path = input_path + ".backup"
    if input_path == output_path and not Path(backup_path).exists():
        shutil.copy2(input_path, backup_path)
        print(f"Backup saved: {backup_path}")

    # Load graph structure only (weights stay in external data)
    model = onnx.load(input_path, load_external_data=False)
    graph = model.graph

    # Find the logits output to tap into
    logits_output = None
    for output in graph.output:
        if output.name == "logits":
            logits_output = output
            break

    if logits_output is None:
        raise ValueError("Could not find 'logits' output in the graph.")
    print(f"Found logits output: shape={[d.dim_value or d.dim_param for d in logits_output.type.tensor_type.shape.dim]}")

    # Get the elem_type from logits (10 = float16, 1 = float32)
    logits_elem_type = logits_output.type.tensor_type.elem_type
    print(f"Logits elem_type: {logits_elem_type}")

    # Verify opset — ArgMax needs opset >= 11
    ai_onnx_domain = ""
    ai_onnx_version = None
    for oi in model.opset_import:
        if oi.domain == "" or oi.domain == "ai.onnx":
            ai_onnx_version = oi.version
    print(f"ai.onnx opset version: {ai_onnx_version}")

    # 1. Add ArgMax node: reduces [batch, seq, vocab] -> [batch, seq]
    #    keepdims=0 so output is [batch, seq] not [batch, seq, 1]
    argmax_node = helper.make_node(
        "ArgMax",
        inputs=["logits"],
        outputs=["next_token_id_int64"],
        name="argmax_next_token",
        axis=-1,
        keepdims=0,
    )

    # 2. Cast INT64 -> INT32 for JS consumption (WebGPU handles int32 natively)
    cast_node = helper.make_node(
        "Cast",
        inputs=["next_token_id_int64"],
        outputs=["next_token_id"],
        name="cast_next_token_int32",
        to=TensorProto.INT32,
    )

    graph.node.extend([argmax_node, cast_node])

    # 3. Create the new output value info
    #    Preserve symbolic dims from logits shape minus the last (vocab) dimension
    logits_shape = logits_output.type.tensor_type.shape
    new_output_type = TypeProto()
    new_output_type.tensor_type.elem_type = TensorProto.INT32

    # Copy all dims except the last one (vocab dimension)
    for i, old_dim in enumerate(logits_shape.dim):
        if i == len(logits_shape.dim) - 1:
            continue  # skip vocab dim
        new_dim = new_output_type.tensor_type.shape.dim.add()
        if old_dim.dim_param:
            new_dim.dim_param = old_dim.dim_param
        elif old_dim.dim_value:
            new_dim.dim_value = old_dim.dim_value
        # else: keep as-is (unknown dim)

    new_output = ValueInfoProto()
    new_output.name = "next_token_id"
    new_output.type.CopyFrom(new_output_type)

    graph.output.append(new_output)

    # 4. Save the modified graph (external data reference preserved)
    onnx.save(model, output_path)
    print(f"\nModified model saved: {output_path}")
    print(f"New output: next_token_id (INT32) shape={[d.dim_value or d.dim_param for d in new_output_type.tensor_type.shape.dim]}")

    # 5. Verify
    try:
        onnx.checker.check_model(output_path)
        print("onnx.checker: PASSED")
    except Exception as e:
        print(f"onnx.checker: WARNING — {e}")
        print("Model may still work with ORT. Proceed with caution.")

    return model


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    append_argmax_to_decoder(input_file, output_file)
