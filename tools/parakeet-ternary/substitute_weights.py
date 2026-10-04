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
    parser.add_argument(
        "--stage",
        choices=("norms", "matmul", "full"),
        default="full",
        help="ladder stage: norms only, norms plus matched constants, or everything",
    )
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
    model = onnx.load(str(args.graph), load_external_data=False)
    entries = {entry.name: entry for entry in model.graph.initializer}
    offsets = {}
    for name, entry in entries.items():
        if entry.external_data:
            meta = {item.key: item.value for item in entry.external_data}
            offsets[name] = (int(meta["offset"]), int(meta["length"]))
        else:
            offsets[name] = (None, None)

    stage = args.stage

    def resolve(name: str):
        """Find the redux tensor for a graph initializer, in either orientation."""
        anonymous = name.startswith("onnx::")
        if stage == "norms" and anonymous:
            return None
        if stage == "matmul" and anonymous and not name.startswith("onnx::MatMul"):
            return None
        hit = mapping.get(name)
        if hit is not None:
            candidate = pool.get(hit[0])
            if candidate is not None and candidate.size == teacher[name].size:
                return (candidate.T if hit[1] else candidate).astype(np.float32)
        if not name.startswith("onnx::"):
            # Teacher names omit the "encoder." prefix that redux uses, and a
            # quantized module is stored without the ".weight" suffix.
            keys = [f"encoder.{name}"]
            if name.endswith(".weight"):
                keys.append(f"encoder.{name[: -len('.weight')]}")
            for key in keys:
                candidate = pool.get(key)
                if candidate is not None and candidate.size == teacher[name].size:
                    return candidate.astype(np.float32).reshape(teacher[name].shape)
        return None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_name = "encoder-model.onnx.data"
    replaced_external, replaced_inline, kept = 0, 0, {}

    # The source external data is copied verbatim, so the protobuf's offsets stay
    # valid. Re-saving with save_as_external_data would re-lay out every tensor
    # and cannot write bytes for external tensors the proto does not carry.
    shutil.copy(args.data_dir / data_name, args.output_dir / data_name)
    replaced_external, replaced_inline, kept = 0, 0, {}
    payload = bytearray()
    with (args.output_dir / data_name).open("r+b") as handle:
        for name, entry in entries.items():
            offset, length = offsets[name]
            value = resolve(name)
            if offset is not None:
                if value is None:
                    kept[name] = "no redux tensor above the correlation threshold"
                    continue
                if value.size * 4 != length:
                    kept[name] = f"size mismatch {value.size * 4} vs {length}"
                    continue
                handle.seek(offset)
                handle.write(value.tobytes())
                replaced_external += 1
            else:
                if value is None:
                    kept[name] = "inline initializer with no redux match"
                    continue
                if value.size * 4 != len(entry.raw_data):
                    kept[name] = f"inline size mismatch {value.size * 4} vs {len(entry.raw_data)}"
                    continue
                entry.ClearField("float_data")
                entry.ClearField("int32_data")
                entry.raw_data = value.tobytes()
                replaced_inline += 1

    graph_path = args.output_dir / "encoder-model.onnx"
    onnx.save(model, str(graph_path))

    for extra in ("repack-receipt.json", "source-config.json", "vocab.txt"):
        candidate = args.data_dir / extra
        if candidate.exists():
            shutil.copy(candidate, args.output_dir / extra)

    receipt = {
        "stage": stage,
        "model": "moondream/parakeet-redux",
        "graph": "copy of the validated teacher encoder export",
        "tensorsInGraph": len(teacher),
        "replacedExternal": replaced_external,
        "replacedInline": replaced_inline,
        "keptTeacherValues": len(kept),
        "keptReasons": sorted({reason for reason in kept.values()}),
        "keptExamples": sorted(kept)[:8],
        "coverage": round((replaced_external + replaced_inline) / max(len(teacher), 1), 3),
        "outputDir": str(args.output_dir),
        "dataSha256": hashlib.sha256((args.output_dir / data_name).read_bytes()).hexdigest(),
        "dataBytes": (args.output_dir / data_name).stat().st_size,
        "warning": "Coverage counts tensor replacement, not numerical validation. An oracle gate is still required.",
    }
    (args.output_dir / "redux-substitution-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())