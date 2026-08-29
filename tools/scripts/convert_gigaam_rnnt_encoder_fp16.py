#!/usr/bin/env python
"""Convert the GigaAM v3 E2E RNN-T fp32 encoder to fp16.

Outputs v3_e2e_rnnt_encoder.fp16.onnx next to the source artifact. Uses the
onnxruntime.transformers.float16 converter with keep_io_types=True so the
graph contract (float32 audio_signal in, float32 encoder outputs out) is
unchanged; casts are inserted internally. Run parity checks before use:
tools/scripts/check_gigaam_rnnt_fp16_parity.py.
"""
import argparse
import pathlib
import time

from onnxruntime.transformers import float16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=r"N:\models\onnx\gigaam\v3-e2e-rnnt\v3_e2e_rnnt_encoder.onnx",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = pathlib.Path(args.source)
    output = pathlib.Path(args.output) if args.output else source.with_name(
        source.stem + ".fp16.onnx"
    )

    print(f"source: {source}")
    print(f"output: {output}")
    started = time.time()
    converted = float16.convert_float_to_float16(
        source.as_posix(),
        keep_io_types=True,
        disable_shape_infer=True,
    )
    import onnx

    onnx.save_model(converted, output.as_posix())
    print(f"saved {output.stat().st_size / 1e6:.1f} MB in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

