#!/usr/bin/env python
"""Create a dynamic-INT8 GigaAM v3 RNN-T encoder probe.

The input/output contract remains float32.  ONNX Runtime inserts dynamic
activation quantization around weight-bearing MatMul/Conv nodes while the
original fp32 source stays untouched.  This is a benchmark artifact, not a
production default: callers must run numerical parity and browser provider
checks before promoting it.
"""

from __future__ import annotations

import argparse
import pathlib
import time

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="N:/models/onnx/gigaam/v3-e2e-rnnt/v3_e2e_rnnt_encoder.onnx",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--ops",
        default="MatMul",
        help="Comma-separated operator types to quantize (default: MatMul)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel int8 weight scales (slower conversion, usually better accuracy)",
    )
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output) if args.output else source.with_name(
        source.stem + ".int8.onnx"
    )
    op_types = [value.strip() for value in args.ops.split(",") if value.strip()]
    if not op_types:
        raise SystemExit("--ops must contain at least one operator type")

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"source: {source}")
    print(f"output: {output}")
    print(f"ops: {','.join(op_types)}")
    print(f"per_channel: {args.per_channel}")
    started = time.time()
    quantize_dynamic(
        model_input=source.as_posix(),
        model_output=output.as_posix(),
        op_types_to_quantize=op_types,
        per_channel=args.per_channel,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    print(f"saved {output.stat().st_size / 1e6:.1f} MB in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
