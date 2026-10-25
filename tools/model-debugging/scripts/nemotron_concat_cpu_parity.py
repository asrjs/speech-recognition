from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort


def make_feed(
    session: ort.InferenceSession,
    features: np.ndarray,
    cache_state: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    cache_state = cache_state or {}
    for item in session.get_inputs():
        name = item.name
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in item.shape]
        if name in cache_state:
            feed[name] = cache_state[name]
        elif name == "audio_signal":
            feed[name] = features
        elif name == "length":
            feed[name] = np.asarray([features.shape[1]], dtype=np.int64)
        elif name == "cache_last_channel_len":
            feed[name] = np.zeros(shape, dtype=np.int64)
        elif name == "lang_id":
            feed[name] = np.zeros(shape, dtype=np.int64)
        elif name == "cache_last_channel":
            # Match the shipped Nemotron executor config. The graph labels this
            # dimension "batch", but the active runtime feeds 24 cache layers here.
            feed[name] = np.zeros((1, 24, 56, 1024), dtype=np.float32)
        elif name == "cache_last_time":
            feed[name] = np.zeros((1, 24, 1024, 8), dtype=np.float32)
        elif "int64" in item.type:
            feed[name] = np.zeros(shape, dtype=np.int64)
        else:
            feed[name] = np.zeros(shape, dtype=np.float32)
    return feed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"unexpected execution providers: {session.get_providers()}")
    return session


def compare_outputs(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, object]:
    if baseline.keys() != candidate.keys():
        raise RuntimeError(
            f"output names differ: baseline={sorted(baseline)} candidate={sorted(candidate)}"
        )
    records = []
    for name in baseline:
        ref = np.asarray(baseline[name])
        got = np.asarray(candidate[name])
        if ref.shape != got.shape:
            records.append({"name": name, "baselineShape": list(ref.shape), "candidateShape": list(got.shape), "sameShape": False})
            continue
        delta = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        records.append(
            {
                "name": name,
                "shape": list(ref.shape),
                "dtype": str(ref.dtype),
                "baselineMaxAbs": float(np.abs(ref).max(initial=0)),
                "candidateMaxAbs": float(np.abs(got).max(initial=0)),
                "baselineNonzeroCount": int(np.count_nonzero(ref)),
                "candidateNonzeroCount": int(np.count_nonzero(got)),
                "maxAbs": float(delta.max(initial=0)),
                "meanAbs": float(delta.mean()) if delta.size else 0.0,
                "arrayEqual": bool(np.array_equal(ref, got)),
                "allcloseAtol1e6Rtol1e6": bool(np.allclose(ref, got, atol=1e-6, rtol=1e-6)),
            }
        )
    return {
        "outputCount": len(records),
        "allShapesEqual": all(item.get("sameShape", True) for item in records),
        "allArrayEqual": all(item.get("arrayEqual", False) for item in records),
        "allcloseAtol1e6Rtol1e6": all(item.get("allcloseAtol1e6Rtol1e6", False) for item in records),
        "outputs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--feature-json", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.original, args.candidate, args.feature_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.result.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.result}")

    record: dict[str, object] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU-only Nemotron encoder/cache graph equivalence; not ASR transcript/model parity",
        "providersRequested": ["CPUExecutionProvider"],
        "threads": {"intraOp": 2, "interOp": 2},
        "optimizationForParity": "ORT_DISABLE_ALL",
        "original": str(args.original.resolve()),
        "candidate": str(args.candidate.resolve()),
        "frozenFeatureFile": str(args.feature_json.resolve()),
        "artifacts": {
            "originalGraphSha256": sha256_file(args.original),
            "candidateGraphSha256": sha256_file(args.candidate),
            "frozenFeatureFileSha256": sha256_file(args.feature_json),
            "onnxruntimeVersion": ort.__version__,
        },
    }

    source = json.loads(args.feature_json.read_text(encoding="utf-8"))
    feature_bins_major = np.asarray(source["feature"], dtype=np.float32)
    if feature_bins_major.ndim != 2 or feature_bins_major.shape[0] != 128:
        raise ValueError(f"expected frozen features [128,T], got {feature_bins_major.shape}")
    frame_major = feature_bins_major.T
    chunk_frames = 65
    if frame_major.shape[0] < 2 * chunk_frames:
        raise ValueError(f"frozen feature file has only {frame_major.shape[0]} frames")
    features_first = frame_major[:chunk_frames][None, :, :].copy()
    features_cont = frame_major[chunk_frames : 2 * chunk_frames][None, :, :].copy()
    record["featureInput"] = {
        "sourceFixture": source.get("fixture"),
        "sourceFeatureShapeMetadata": source.get("featureShape"),
        "actualFeatureShapeBinsMajor": list(feature_bins_major.shape),
        "feedShape": list(features_first.shape),
        "frames": chunk_frames,
        "continuationFrames": chunk_frames,
        "cacheState": "zero initialized using current executor cache dimensions [1,24,56,1024] and [1,24,1024,8]",
        "promptId": 0,
    }

    original_session = create_cpu_session(args.original)
    candidate_session = create_cpu_session(args.candidate)
    original_feed = make_feed(original_session, features_first)
    candidate_feed = make_feed(candidate_session, features_first)
    if original_feed.keys() != candidate_feed.keys():
        raise RuntimeError("input names differ between original and candidate")
    for name in original_feed:
        if original_feed[name].shape != candidate_feed[name].shape:
            raise RuntimeError(f"input shape differs for {name}")
    record["inputNames"] = list(original_feed)
    record["feedShapes"] = {name: list(value.shape) for name, value in original_feed.items()}
    names = [item.name for item in original_session.get_outputs()]
    candidate_names = [item.name for item in candidate_session.get_outputs()]
    if names != candidate_names:
        raise RuntimeError(f"output names/order differ: {names} versus {candidate_names}")
    original_first = dict(zip(names, original_session.run(None, original_feed)))
    candidate_first = dict(zip(candidate_names, candidate_session.run(None, candidate_feed)))
    first_comparison = compare_outputs(original_first, candidate_first)

    original_cache = {
        "cache_last_channel": original_first["cache_last_channel_next"],
        "cache_last_time": original_first["cache_last_time_next"],
        "cache_last_channel_len": original_first["cache_last_channel_len_next"],
    }
    candidate_cache = {
        "cache_last_channel": candidate_first["cache_last_channel_next"],
        "cache_last_time": candidate_first["cache_last_time_next"],
        "cache_last_channel_len": candidate_first["cache_last_channel_len_next"],
    }
    original_cont_feed = make_feed(original_session, features_cont, original_cache)
    candidate_cont_feed = make_feed(candidate_session, features_cont, candidate_cache)
    continuation_comparison = compare_outputs(
        dict(zip(names, original_session.run(None, original_cont_feed))),
        dict(zip(names, candidate_session.run(None, candidate_cont_feed))),
    )
    record["comparison"] = {
        "firstChunk": first_comparison,
        "continuationWithReturnedCaches": continuation_comparison,
        "allArrayEqual": first_comparison["allArrayEqual"]
        and continuation_comparison["allArrayEqual"],
    }
    record["status"] = (
        "PASS_GRAPH_EQUIVALENCE_ONLY"
        if record["comparison"]["allArrayEqual"]  # type: ignore[index]
        else "FAIL_GRAPH_EQUIVALENCE"
    )

    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
