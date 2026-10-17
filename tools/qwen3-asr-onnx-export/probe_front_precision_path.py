"""Show where ORT CPU executes the fp16 front, and score both fp16 paths against fp32.

The checkpoint is bf16, so loading it as fp32 reproduces the original weights
exactly; fp32 compute on those weights is the original-model oracle. Native
PyTorch fp16 and the ONNX fp16 front are each compared with that oracle, which
separates "ONNX differs from one fp16 rounding path" from "ONNX is less
accurate". The optimized ORT graph is saved to count inserted precision casts.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from export_qwen3_asr_fp16 import _digest  # noqa: E402
from tensor_parity import compare_tensors  # noqa: E402

ATOL, RTOL = 1e-4, 1e-3
LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}


def embeds(model, features, mask, cu_seqlens, dtype) -> np.ndarray:
    with torch.no_grad():
        hidden = model.model.audio_tower(features.to(dtype), mask, cu_seqlens=cu_seqlens).last_hidden_state
        return model.model.multi_modal_projector(hidden).float().numpy()


def cast_summary(path: Path) -> dict:
    graph = onnx.load(str(path), load_external_data=False).graph
    ops = collections.Counter(node.op_type for node in graph.node)
    inserted = [node for node in graph.node if node.op_type == "Cast" and "InsertedPrecisionFreeCast" in node.name]
    produced_by = {out: node for node in graph.node for out in node.output}
    fp32_islands = collections.Counter()
    for node in graph.node:
        if node.op_type == "Cast":
            continue
        sources = [produced_by.get(name) for name in node.input if name]
        if any(s is not None and s in inserted for s in sources):
            fp32_islands[node.op_type] += 1
    return {
        "node_count": len(graph.node),
        "op_counts": dict(ops.most_common()),
        "inserted_precision_casts": len(inserted),
        "ops_fed_by_inserted_casts": dict(fp32_islands.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[100, 500, 1100])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source = args.artifact_dir / "audio_front_model.onnx"
    report = {
        "status": "PRECISION_PATH_MEASURED_NOT_NUMERICAL_ACCEPTANCE",
        "comparator": f"tools/tensor_parity.py::compare_tensors(reference, candidate, atol={ATOL}, rtol={RTOL})",
        "oracle": "original bf16 checkpoint weights loaded as fp32, fp32 compute, eager attention",
        "artifact": {"path": str(source), "sha256": _digest(source)},
        "source_sha256": _digest(Path(__file__)),
        "versions": {"onnxruntime": ort.__version__, "torch": torch.__version__, "transformers": transformers.__version__},
        "optimized_graphs": {},
        "lengths": {},
    }
    sessions = {}
    with tempfile.TemporaryDirectory() as directory:
        for level, value in LEVELS.items():
            options = ort.SessionOptions()
            options.graph_optimization_level = value
            optimized = Path(directory) / f"front-{level}.onnx"
            options.optimized_model_filepath = str(optimized)
            options.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name", optimized.name + ".data")
            options.add_session_config_entry(
                "session.optimized_model_external_initializers_min_size_in_bytes", "1024")
            sessions[level] = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
            report["optimized_graphs"][level] = cast_summary(optimized)

    inputs = {}
    for frames in args.frames:
        inputs[frames] = (
            torch.from_numpy(np.load(args.artifact_dir / f"features-{frames}.npy")),
            torch.from_numpy(np.load(args.artifact_dir / f"mask-{frames}.npy")),
            torch.from_numpy(np.fromfile(args.artifact_dir / f"schedule-{frames}.bin", dtype=np.int32)),
        )

    results = {frames: {} for frames in inputs}
    for dtype in (torch.float32, torch.float16):
        model = transformers.AutoModelForMultimodalLM.from_pretrained(
            args.model_dir, dtype=dtype, attn_implementation="eager", local_files_only=True).eval()
        for frames, (features, mask, schedule) in inputs.items():
            results[frames][str(dtype)] = embeds(model, features, mask, schedule, dtype)
        del model

    for frames, (features, mask, schedule) in inputs.items():
        oracle = results[frames]["torch.float32"]
        native = results[frames]["torch.float16"]
        feed = {"input_features": features.numpy(), "input_features_mask": mask.numpy(), "cu_seqlens": schedule.numpy()}
        row = {"native_fp16_vs_fp32_oracle": compare_tensors(oracle, native, atol=ATOL, rtol=RTOL)}
        for level, session in sessions.items():
            produced = session.run(None, feed)[0]
            row[f"onnx_{level}_vs_fp32_oracle"] = compare_tensors(oracle, produced, atol=ATOL, rtol=RTOL)
            row[f"onnx_{level}_vs_native_fp16"] = compare_tensors(native, produced, atol=ATOL, rtol=RTOL)
        report["lengths"][str(frames)] = row
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
