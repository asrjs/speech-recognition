"""Decode `moondream/parakeet-redux` safetensors into a plain fp32 state dict.

The archive holds, per quantized module, a `qweight` block of base-3 packed
trits and a `scales` block; everything else is already dense. Expansion is exact:
every weight is a small integer times a stored scale, so float32 reproduces the
quantized model without adding error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ternary import TernaryLayout, decode_matrix  # noqa: E402

_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    # The packed trit blocks are stored as raw bytes.
    "U8": np.uint8,
}


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))

    return 8 + length, header


def load(path: Path) -> dict[str, np.ndarray]:
    start, header = read_header(path)
    tensors: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        handle.seek(start)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES[entry["dtype"]]
            count = int(np.prod(entry["shape"])) if entry["shape"] else 1
            handle.seek(entry["data_offsets"][0])
            raw = handle.read(count * np.dtype(dtype).itemsize)
            tensors[name] = np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])
    return tensors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(".cache/parakeet-redux/model.safetensors"))
    parser.add_argument("--manifest", type=Path, default=Path(".cache/parakeet-candidates/parakeet-redux-ternary.json"))
    parser.add_argument("--output", type=Path, default=Path(".cache/parakeet-redux/decoded"))
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    group_size = manifest["quant"]["group_size"]
    tensors = load(args.archive)

    quantized = [entry for entry in manifest["quantized_modules"]]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    total_params = 0
    for entry in quantized:
        module = entry["name"]
        out_features, in_features = entry["out_features"], entry["in_features"]
        layout = TernaryLayout(
            in_features=in_features, out_features=out_features, group_size=group_size
        )
        qweight = tensors[f"{module}.qweight"]
        scales = tensors[f"{module}.scales"]
        packed = qweight.reshape(-1).tobytes()
        if packed.__len__() != layout.row_bytes * out_features:
            raise SystemExit(
                f"{module}: packed block is {len(packed)} bytes, layout wants "
                f"{layout.row_bytes * out_features}"
            )
        if tuple(scales.shape) != (out_features, layout.groups):
            raise SystemExit(
                f"{module}: scales shape {tuple(scales.shape)} does not match "
                f"{(out_features, layout.groups)}"
            )
        weights = decode_matrix(packed, scales.astype(np.float32), layout)
        total_params += out_features * in_features
        rows.append((module, weights))

    saved = 0
    for module, weights in rows:
        destination = args.output / f"{module}.npy"
        weights.tofile(destination.with_suffix(".npy"))
        saved += weights.size

    receipt = {
        "archive": str(args.archive),
        "archiveBytes": args.archive.stat().st_size,
        "archiveSha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "quantizedModules": len(rows),
        "decodedParams": saved,
        "manifestParams": manifest["n_quantized_params"],
        "paramsMatch": saved == manifest["n_quantized_params"],
        "groupSize": group_size,
        "format": manifest["format"],
        "exact": "weights are scales[row, col // group] * (code - 1); float32 adds no error",
        "outputDir": str(args.output),
    }
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["paramsMatch"] else 1


if __name__ == "__main__":
    sys.exit(main())