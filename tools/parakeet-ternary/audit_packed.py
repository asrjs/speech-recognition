"""Offline, SHA-bound thrush-ternary-v2 preflight; never expands model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from decode_safetensors import load_snapshot

WEIGHT_RULE = "w = scales[row, col // group_size] * (code - 1) with code in {0,1,2}"
DIGIT_ORDER = (
    "element i of a row is base-3 digit i%5 of byte i//5, "
    "least significant digit first; pad digits 0"
)


def read_verified(path: Path, expected: str) -> tuple[bytes, str]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"{path}: recorded SHA-256 is required")
    snapshot = path.read_bytes()
    actual = hashlib.sha256(snapshot).hexdigest()
    if actual != expected.lower():
        raise ValueError(f"{path}: SHA-256 mismatch")
    return snapshot, actual


def check_identity(path: Path, expected: str) -> str:
    return read_verified(path, expected)[1]


def audit(manifest: dict, config: dict, tensors: dict[str, np.ndarray]) -> dict:
    """Validate a manifest's packed tensors without claiming inference parity."""
    packing = {
        "base": 3, "code_offset": 1, "elements_per_byte": 5,
        "row_bytes": "ceil(in/5)", "row_major": True, "digit_order": DIGIT_ORDER,
    }
    if manifest.get("format") != "thrush-ternary-v2" or manifest.get("names") != "hf":
        raise ValueError("unsupported manifest format/names")
    if any(manifest.get("packing", {}).get(k) != v for k, v in packing.items()):
        raise ValueError("unsupported packing contract")
    if manifest.get("weight_rule") != WEIGHT_RULE:
        raise ValueError("unsupported weight rule")
    quant = manifest.get("quant", {})
    group = quant.get("group_size")
    if quant.get("mode") != "ternary" or type(group) is not int or group <= 0:
        raise ValueError("invalid quantization group")
    if config.get("ternary_group_size") != group:
        raise ValueError("config/manifest group mismatch")
    entries = manifest.get("quantized_modules", [])
    names = [entry["name"] for entry in entries]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate module list")
    configured = config.get("ternary_modules", [])
    if len(configured) != len(set(configured)) or set(configured) != set(names):
        raise ValueError("config/manifest module mismatch")
    expected_keys = {name + suffix for name in names for suffix in (".qweight", ".scales")}
    actual_keys = {name for name in tensors if name.endswith((".qweight", ".scales"))}
    if actual_keys != expected_keys:
        raise ValueError("missing or unlisted packed tensors")
    params = qbytes = sbytes = two_bit_bytes = 0
    for entry in entries:
        name = entry["name"]
        rows, cols = entry["out_features"], entry["in_features"]
        if any(type(n) is not int or n <= 0 for n in (rows, cols)):
            raise ValueError(f"{name}: invalid matrix dimensions")
        if entry["group_size"] != group:
            raise ValueError(f"{name}: group mismatch")
        qweight, scales = tensors[name + ".qweight"], tensors[name + ".scales"]
        if qweight.dtype != np.uint8 or qweight.shape != (rows, (cols + 4) // 5):
            raise ValueError(f"{name}: invalid qweight dtype/shape")
        if scales.dtype != np.float16 or scales.shape != (rows, (cols + group - 1) // group):
            raise ValueError(f"{name}: invalid scales dtype/shape")
        if np.any(qweight >= 243):
            raise ValueError(f"{name}: invalid packed ternary byte")
        if cols % 5 and np.any(qweight[:, -1] // (3 ** (cols % 5))):
            raise ValueError(f"{name}: nonzero padding digits")
        if not np.isfinite(scales).all():
            raise ValueError(f"{name}: nonfinite scales")
        params += rows * cols
        qbytes += qweight.nbytes
        sbytes += scales.nbytes
        two_bit_bytes += rows * ((cols + 3) // 4)
    if manifest.get("n_quantized_params") != params:
        raise ValueError("manifest quantized parameter count mismatch")
    return {
        "status": "PASS_PACKED_FORMAT_ONLY", "quantizedModules": len(entries),
        "quantizedParams": params, "qweightBytes": qbytes, "scaleBytes": sbytes,
        "estimatedTwoBitCodeBytes": two_bit_bytes,
        "denseStoredElementsIncludingBuffersAndVAD": int(sum(
            tensor.size for name, tensor in tensors.items() if name not in expected_keys
        )),
        "scope": "SHA/layout/range/padding/finite-scale checks only; no repack, kernel, oracle, parity or speed acceptance.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("archive", "manifest", "config"):
        parser.add_argument("--" + role, type=Path, required=True)
        parser.add_argument("--" + role + "-sha256", required=True)
    args = parser.parse_args()
    verified = {
        role: read_verified(
            getattr(args, role), getattr(args, role + "_sha256")
        ) for role in ("archive", "manifest", "config")
    }
    identities = {
        role: {"path": str(getattr(args, role)), "sha256": verified[role][1]}
        for role in verified
    }
    result = audit(
        json.loads(verified["manifest"][0]), json.loads(verified["config"][0]),
        load_snapshot(verified["archive"][0]),
    )
    result["inputs"] = identities
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
