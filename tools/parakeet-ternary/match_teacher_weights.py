"""Map the validated teacher encoder graph's anonymous MatMul constants onto redux modules.

The teacher graph folds every transposed linear weight into an unnamed
`onnx::MatMul_N` constant, so names cannot carry the mapping. Shapes are too
ambiguous on their own: 120 of the 217 tensors are `[1024, 1024]`.

Correlation does carry it. redux is a ternary repack of the same checkpoint, so
each teacher tensor is the high-precision original of exactly one redux module
and correlates far more strongly with it than with any other candidate. A cheap
subsample picks candidates; the full tensor confirms.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_safetensors import load  # noqa: E402
from ternary import TernaryLayout, decode_matrix  # noqa: E402

SAMPLE = 2048
SEED = 20261004


def load_teacher(graph_path: Path, data_dir: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(graph_path), load_external_data=False)
    out: dict[str, np.ndarray] = {}
    for entry in model.graph.initializer:
        if entry.external_data:
            meta = {item.key: item.value for item in entry.external_data}
            with (data_dir / meta["location"]).open("rb") as handle:
                handle.seek(int(meta["offset"]))
                out[entry.name] = np.frombuffer(
                    handle.read(int(meta["length"])), dtype=np.float32
                ).reshape(list(entry.dims))
        else:
            out[entry.name] = numpy_helper.to_array(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("models/orukeet/onnx/fp32-packed-2026-10-01/encoder-model.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("models/orukeet/onnx/fp32-packed-2026-10-01"))
    parser.add_argument("--archive", type=Path, default=Path(".cache/parakeet-redux/model.safetensors"))
    parser.add_argument("--manifest", type=Path, default=Path(".cache/parakeet-candidates/parakeet-redux-ternary.json"))
    parser.add_argument("--limit", type=int, default=0, help="stop after N tensors, 0 for all")
    parser.add_argument("--output", type=Path, default=Path("tools/data/results/model-candidates/redux-weight-map.json"))
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    group_size = manifest["quant"]["group_size"]
    specs = {entry["name"]: entry for entry in manifest["quantized_modules"]}
    raw = load(Path(args.archive))

    decoded: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        decoded[name] = decode_matrix(
            raw[f"{name}.qweight"].tobytes(),
            raw[f"{name}.scales"].astype(np.float32),
            TernaryLayout(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                group_size=group_size,
            ),
        )

    teacher = load_teacher(args.graph, args.data_dir)
    by_shape: dict[tuple, list[str]] = {}
    for name, tensor in decoded.items():
        by_shape.setdefault(tuple(tensor.shape), []).append(name)

    rng = np.random.default_rng(SEED)
    targets = [n for n in teacher if n.startswith("onnx::MatMul")]
    targets.sort(key=lambda n: int(n.rsplit("_", 1)[1]))
    if args.limit:
        targets = targets[: args.limit]

    rows = []
    for name in targets:
        reference = teacher[name]
        shape = tuple(reference.shape)
        size = reference.size
        pick = rng.choice(size, size=min(SAMPLE, size), replace=False)
        base = reference.ravel()[pick]
        base_norm = float(np.linalg.norm(base - base.mean()))

        best_name, best_corr, best_flip = None, -2.0, False
        # A folded MatMul constant may hold either orientation of the weight, so
        # both are tried; picking one up front mis-pairs the feed-forward blocks.
        for orientation in (False, True):
            pool = by_shape.get(tuple(reference.T.shape) if orientation else shape)
            if not pool:
                continue
            for candidate in pool:
                array = decoded[candidate]
                flat = (array.T if orientation else array).ravel()
                sample = flat[pick]
                norm = float(np.linalg.norm(sample - sample.mean()))
                if base_norm == 0 or norm == 0:
                    continue
                value = float(
                    np.dot(base - base.mean(), sample - sample.mean()) / (base_norm * norm)
                )
                if value > best_corr:
                    best_corr, best_name, best_flip = value, candidate, orientation

        full = None
        if best_name is not None:
            array = decoded[best_name]
            full = float(np.corrcoef(reference.ravel(), (array.T if best_flip else array).ravel())[0, 1])
        rows.append(
            {
                "tensor": name,
                "shape": list(reference.shape),
                "match": best_name,
                "transposed": best_flip,
                "sample_correlation": round(best_corr, 4),
                "full_correlation": None if full is None else round(full, 4),
            }
        )
        print(f"{name} {list(reference.shape)} -> {best_name} sample={best_corr:.4f} full={full}")

    confident = [r for r in rows if r["full_correlation"] and r["full_correlation"] > 0.5]
    payload = {
        "graph": str(args.graph),
        "tensors": len(rows),
        "confidentMatches": len(confident),
        "threshold": 0.5,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())