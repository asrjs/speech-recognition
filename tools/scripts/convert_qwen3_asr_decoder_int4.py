#!/usr/bin/env python
"""Create INT4 (MatMulNBits) weight-only Qwen3-ASR 0.6B decoder graphs.

The official decoder-step/prefill graphs keep fp32 weights in a shared
external data file, so every single-token decode step re-reads ~3 GB of
weights. Weight-only INT4 shrinks the per-step weight traffic roughly 4x
for the transformer blocks while the float graph contract (inputs, KV
cache tensors, logits) stays unchanged. The lm_head MatMul is excluded by
default to limit logits drift; exact-token parity is still gated in the
browser before any promotion.

ORT Web 1.29 ships the MatMulNBits kernel in its JSEP/WebGPU bundle, which
makes this the shelf-standard fix for memory-bound single-token decoding
(same idea as WebLLM/MLC and llama.cpp weight-only quantization).
"""

from __future__ import annotations

import argparse
import pathlib
import time

from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer


DEFAULT_SOURCE = "N:/models/onnx/qwen3-asr-0.6b-official/decoder-step.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--block-size", type=int, default=128, choices=(16, 32, 64, 128, 256))
    parser.add_argument("--include-lm-head", action="store_true",
                        help="Also quantize /lm_head/MatMul (higher drift risk; gate carefully)")
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output) if args.output else source.with_name(
        source.stem + ".int4.onnx"
    )
    exclude = [] if args.include_lm_head else ["/lm_head/MatMul"]
    print(f"source: {source}")
    print(f"output: {output}")
    print(f"block_size: {args.block_size}")
    print(f"excluded nodes: {exclude}")
    started = time.time()
    quantizer = MatMul4BitsQuantizer(
        source.as_posix(),
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=0,
        nodes_to_exclude=exclude or None,
    )
    quantizer.process()
    output.parent.mkdir(parents=True, exist_ok=True)
    quantizer.model.save_model_to_file(output.as_posix(), use_external_data_format=True)
    data_path = output.with_name(output.name + ".data")
    total = output.stat().st_size + (data_path.stat().st_size if data_path.exists() else 0)
    print(f"saved {total / 1e6:.1f} MB total in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
