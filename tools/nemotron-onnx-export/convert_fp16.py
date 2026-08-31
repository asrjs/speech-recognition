#!/usr/bin/env python
"""Convert a fp32 ONNX graph to fp16 weights while keeping fp32 graph IO.

Uses ORT's transformers float16 converter (handles large external-data
models; onnxconverter-common returned an empty graph on the encoder) with
keep_io_types so the executor contract (fp32 tensors in/out) is unchanged.
Softmax stays fp32, matching the upstream fp16 WebGPU export choice.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.transformers import float16 as ort_float16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--block-list", default="Softmax", help="comma-separated ops kept in fp32")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(src), load_external_data=True)
    op_block_list = {op.strip() for op in args.block_list.split(",") if op.strip()}
    # disable_shape_infer: onnx.shape_inference silently returns an empty
    # model on this 1.2 GB graph; the export already has static shapes.
    converted = ort_float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=op_block_list,
        disable_shape_infer=True,
    )
    data_name = dst.name + ".data"
    onnx.save_model(converted, str(dst), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_name)
    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB graph + {data_name})", flush=True)


if __name__ == "__main__":
    main()
