"""Write redux weights into a copy of the validated teacher encoder graph.

The teacher graph is a surgically renamed export whose transposed linear weights
are folded into anonymous constants, so this reuses its proven structure and
replaces only tensor values. Coverage is reported rather than assumed: a tensor
that cannot be matched keeps the teacher value, and the receipt says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_safetensors import load  # noqa: E402
from match_teacher_weights import load_teacher  # noqa: E402
from ternary import TernaryLayout, decode_matrix  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("models/orukeet/onnx/fp32-packed-2026-10-01/encoder-model.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("models/orukeet/onnx/fp32-packed-2026-10-01"))
    parser.add_argument("--archive", type=Path, default=Path(".cache/parakeet-redux/model.safetensors"))
    parser.add_argument("--manifest", type=Path, default=Path(".cache/parakeet-candidates/parakeet-redux-ternary.json"))
    parser.add_argument("--matmul-map", type=Path, default=Path("tools/data/results/model-candidates/redux-weight-map.json"))
    parser.add_argument("--conv-map", type=Path, default=Path("tools/data/results/model-candidates/redux-weight-map.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    group_size = manifest["quant"]["group_size"]
    specs = {entry["name"]: entry for entry in manifest["quantized_modules"]}
    raw = load(args.archive)

    pool: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        pool[name] = decode_matrix(
            raw[f"{name}.qweight"].tobytes(),
            raw[f"{name}.scales"].astype(np.float32),
            TernaryLayout(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                group_size=group_size,
            ),
        )
    for name, tensor in raw.items():
        if name.endswith(".scales") or name.endswith(".qweight"):
            continue
        pool.setdefault(name, tensor.astype(np.float32))

    mapping: dict[str, tuple[str, bool]] = {}
    for path in (args.matmul_map, args.conv_map):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            if row["match"] and row.get("full_correlation") and row["full_correlation"] > 0.5:
                mapping[row["tensor"]] = (row["match"], bool(row.get("transposed")))

    teacher = load_teacher(args.graph, args.data_dir)
    # Parse the graph once: reloading per initializer would re-parse the model
    # hundreds of times.
    model = onnx.load(str(args.graph), load_external_data=False)
    offsets = {}
    for entry in model.graph.initializer:
        if entry.external_data:
            meta = {item.key: item.value for item in entry.external_data}
            offsets[entry.name] = (int(meta["offset"]), int(meta["length"]))
        else:
            offsets[entry.name] = (None, None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_name = "encoder-model.onnx.data"
    source_data = args.data_dir / data_name
    shutil.copy(args.graph, args.output_dir / "encoder-model.onnx")
    for extra in ("repack-receipt.json", "source-config.json", "vocab.txt"):
        candidate = args.data_dir / extra
        if candidate.exists():
            shutil.copy(candidate, args.output_dir / extra)

    replaced, named_replaced = 0, 0
    kept: dict[str, str] = {}
    with source_data.open("rb") as source, (args.output_dir / data_name).open("wb") as target:
        payload = bytearray(source.read())
        for name, current in teacher.items():
            value = None
            hit = mapping.get(name)
            if hit is not None:
                candidate = pool.get(hit[0])
                if candidate is not None and candidate.size == current.size:
                    value = (candidate.T if hit[1] else candidate).astype(np.float32)
            if value is None and not name.startswith("onnx::"):
                # Teacher names omit the "encoder." prefix that redux uses.
                candidate = pool.get(f"encoder.{name}")
                if candidate is not None:
                    value = candidate.astype(np.float32)
                    named_replaced += 1
            offset, length = offsets[name]
            if offset is None:
                kept[name] = "inline initializer, not rewritten"
                continue
            if value is None:
                kept[name] = "no match above the correlation threshold"
                continue
            if value.size * 4 != length:
                kept[name] = f"size mismatch {value.size * 4} vs {length}"
                continue
            payload[offset : offset + length] = value.tobytes()
            replaced += 1
        target.write(bytes(payload))

    receipt = {
        "model": "moondream/parakeet-redux",
        "graph": "copy of the validated teacher encoder export",
        "tensorsInGraph": len(teacher),
        "replacedFromAnonymousConstants": replaced,
        "replacedByName": named_replaced,
        "keptTeacherValues": len(kept),
        "keptReasons": sorted({reason for reason in kept.values()}),
        "keptExamples": sorted(kept)[:8],
        "coverage": round((replaced) / max(len(teacher), 1), 3),
        "outputDir": str(args.output_dir),
        "dataSha256": hashlib.sha256((args.output_dir / data_name).read_bytes()).hexdigest(),
        "warning": "Coverage counts tensor replacement, not numerical validation. An oracle gate is still required.",
    }
    (args.output_dir / "redux-substitution-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())