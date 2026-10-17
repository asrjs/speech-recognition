#!/usr/bin/env python
"""Convert a fp32 ONNX graph to fp16 weights while keeping fp32 graph IO.

Uses ORT's transformers float16 converter (handles large external-data
models; onnxconverter-common returned an empty graph on the encoder) with
keep_io_types so the executor contract (fp32 tensors in/out) is unchanged.
Softmax stays fp32, matching the upstream fp16 WebGPU export choice.

When ``--model-dir`` is given instead of ``--input/--output``, every graph
under the directory is converted in place: the root decoder/joint and every
``windows/<ms>/encoder.onnx`` variant (each output keeps its own .data
sibling). ``--embed-dir`` embeds the external data of every graph under a
directory into a single self-contained file, the layout the Node WASM
benchmark loads.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.transformers import float16 as ort_float16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="single fp32 graph to convert (with --output)")
    p.add_argument("--output", help="destination for a single conversion")
    p.add_argument("--model-dir", help="convert every fp32 graph under this dir in place")
    p.add_argument("--embed-dir", help="embed every external-data graph under this dir into single files")
    p.add_argument(
        "--src-dir",
        help="convert every fp32 graph under this dir into --dst-dir",
    )
    p.add_argument("--dst-dir", help="destination directory for --src-dir conversion")
    p.add_argument("--embed", action="store_true", help="with --src-dir: embed weights instead of external data")
    p.add_argument("--block-list", default="Softmax", help="comma-separated ops kept in fp32")
    return p.parse_args()

def convert_graph(src: Path, dst: Path, op_block_list: set[str], embed: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src), load_external_data=True)
    # disable_shape_infer: onnx.shape_inference silently returns an empty
    # model on the 1.2 GB encoder; the export already has static shapes.
    converted = ort_float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=op_block_list,
        disable_shape_infer=True,
    )
    data_name = dst.name + ".data"
    onnx.save_model(
        converted,
        str(dst),
        save_as_external_data=not embed,
        all_tensors_to_one_file=True,
        location=data_name,
    )
    suffix = "" if embed else f" + {data_name}"
    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB{suffix})", flush=True)

def main() -> None:
    args = parse_args()
    op_block_list = {op.strip() for op in args.block_list.split(",") if op.strip()}

    if args.src_dir:
        if not args.dst_dir:
            raise SystemExit("--src-dir requires --dst-dir")
        src_root = Path(args.src_dir)
        dst_root = Path(args.dst_dir)
        graphs = sorted(src_root.glob("*.onnx")) + sorted(
            src_root.glob("windows/*/encoder.onnx")
        )
        for graph in graphs:
            rel = graph.relative_to(src_root)
            convert_graph(graph, dst_root / rel, op_block_list, embed=args.embed)
        vocab = src_root / "vocab.txt"
        if vocab.exists():
            (dst_root / "vocab.txt").write_bytes(vocab.read_bytes())
            print(f"Copied {dst_root / 'vocab.txt'}", flush=True)
        return

    if args.embed_dir:
        root = Path(args.embed_dir)
        for graph in sorted(root.glob("*.onnx")) + sorted(root.glob("windows/*/encoder.onnx")):
            model = onnx.load(str(graph), load_external_data=True)
            onnx.save_model(model, str(graph), save_as_external_data=False)
            print(f"Embedded {graph} ({graph.stat().st_size / 1e6:.1f} MB)", flush=True)
        return

    if args.model_dir:
        root = Path(args.model_dir)
        for graph in sorted(root.glob("*.onnx")):
            if graph.name == "encoder.onnx":
                continue  # default variant lives under windows/<ms>/; converted below
            convert_graph(graph, graph, op_block_list)
        for variant in sorted(root.glob("windows/*/encoder.onnx")):
            convert_graph(variant, variant, op_block_list)
        return

    if not args.input or not args.output:
        raise SystemExit("Provide --input/--output, --model-dir, --embed-dir, or --src-dir/--dst-dir")
    convert_graph(Path(args.input), Path(args.output), op_block_list)


if __name__ == "__main__":
    main()
