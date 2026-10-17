"""Locate where the fp16 Qwen3-ASR front ONNX graph departs from native PyTorch fp16.

Every encoder sub-block output is exposed as an extra graph output in a temporary
copy of the graph that shares the original external-data file. Each tap is
compared twice against native fp16 PyTorch with the receipt comparator:

- cumulative: the full native forward against the full ONNX forward;
- isolated: the native sub-block applied to the ONNX inputs of that sub-block,
  so drift inherited from earlier layers cannot explain a local difference.

At atol=1e-4 / rtol=1e-3 a one-ULP fp16 difference always passes and two ULPs
usually fail, so ULP distance is reported next to the comparator. The script
also counts pre-clamp values beyond the fp16-only clamp in native fp16 and in
the fp32-arithmetic control, which skips that clamp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers.models.qwen3_asr.modeling_qwen3_asr import eager_attention_forward

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from export_qwen3_asr_fp16 import _digest, load  # noqa: E402
from tensor_parity import compare_tensors  # noqa: E402

ATOL, RTOL = 1e-4, 1e-3
CLAMP = float(torch.finfo(torch.float16).max) - 1000
LEVELS = {
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
}
NODE_TAPS = {
    "attn_norm": "self_attn_layer_norm/LayerNormalization",
    "q": "self_attn/q_proj/Gemm",
    "k": "self_attn/k_proj/Gemm",
    "v": "self_attn/v_proj/Gemm",
    "attn": "self_attn/out_proj/Gemm",
    "attn_residual": "Add",
    "mlp_norm": "final_layer_norm/LayerNormalization",
    "fc1": "fc1/Gemm",
    "fc2": "fc2/Gemm",
    "pre_clamp": "Add_1",
    "out": "Clip",
}


def _ordered(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float16).view(np.uint16).astype(np.int32)
    return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)


def metric(reference, candidate) -> dict:
    reference = np.asarray(reference, dtype=np.float16)
    candidate = np.asarray(candidate, dtype=np.float16)
    row = compare_tensors(
        reference.astype(np.float32), candidate.astype(np.float32), atol=ATOL, rtol=RTOL
    )
    ulp = np.abs(_ordered(reference) - _ordered(candidate))
    keep = ("allclose", "max_abs_error", "mean_abs_error", "mismatch_count", "first_divergent_index", "count")
    summary = {key: row[key] for key in keep}
    summary.update(max_ulp=int(ulp.max()), elements_over_1_ulp=int((ulp > 1).sum()))
    return summary


def tap_names(graph: onnx.GraphProto, layers: int) -> dict:
    by_name = {node.name: node for node in graph.node}
    taps = {"stem": by_name["/tower/layers.0/self_attn_layer_norm/LayerNormalization"].input[0]}
    for index in range(layers):
        prefix = f"/tower/layers.{index}/"
        for key, suffix in NODE_TAPS.items():
            taps[f"{index}.{key}"] = by_name[prefix + suffix].output[0]
        taps[f"{index}.context"] = by_name[prefix + "self_attn/out_proj/Gemm"].input[0]
        taps[f"{index}.act"] = by_name[prefix + "fc2/Gemm"].input[0]
    taps["ln_post"] = by_name["/tower/ln_post/LayerNormalization"].output[0]
    return taps


def tapped_graph(source: Path, directory: Path, taps: dict) -> Path:
    model = onnx.load(str(source), load_external_data=False)
    locations = {
        entry.value
        for tensor in model.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    for location in locations:
        os.link(source.parent / location, directory / location)
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {info.name: info.type.tensor_type.elem_type for info in inferred.graph.value_info}
    existing = {output.name for output in model.graph.output}
    for name in taps.values():
        if name not in existing:
            elem = types.get(name) or onnx.TensorProto.FLOAT16
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, elem, None))
            existing.add(name)
    path = directory / source.name
    onnx.save(model, str(path))
    return path


def session(path: Path, level: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = LEVELS[level]
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def window_attention(module, query, key, value, cu_seqlens) -> torch.Tensor:
    length = query.shape[0]
    shaped = [t.reshape(length, module.num_heads, -1).transpose(0, 1).unsqueeze(0) for t in (query, key, value)]
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    splits = [torch.split(t, lengths, dim=2) for t in shaped]
    outputs = [
        eager_attention_forward(module, q, k, v, attention_mask=None, scaling=module.scaling, dropout=0.0)[0]
        for q, k, v in zip(*splits)
    ]
    return torch.cat(outputs, dim=1).reshape(length, -1).contiguous()


def native_taps(tower, features, mask, cu_seqlens) -> tuple[dict, dict]:
    """Full native forward with every sub-block output captured."""
    captured, handles = {}, []

    def keep(name, use_input=False):
        def hook(module, args, output):
            value = args[0] if use_input else (output[0] if isinstance(output, tuple) else output)
            captured[name] = value.detach().clone()
        return hook

    for index, layer in enumerate(tower.layers):
        p = f"{index}."
        handles += [
            layer.register_forward_hook(keep(p + "input", use_input=True)),
            layer.register_forward_hook(keep(p + "out")),
            layer.self_attn_layer_norm.register_forward_hook(keep(p + "attn_norm")),
            layer.self_attn.q_proj.register_forward_hook(keep(p + "q")),
            layer.self_attn.k_proj.register_forward_hook(keep(p + "k")),
            layer.self_attn.v_proj.register_forward_hook(keep(p + "v")),
            layer.self_attn.out_proj.register_forward_hook(keep(p + "context", use_input=True)),
            layer.self_attn.out_proj.register_forward_hook(keep(p + "attn")),
            layer.final_layer_norm.register_forward_hook(keep(p + "mlp_norm")),
            layer.fc1.register_forward_hook(keep(p + "fc1")),
            layer.fc2.register_forward_hook(keep(p + "act", use_input=True)),
            layer.fc2.register_forward_hook(keep(p + "fc2")),
        ]
    handles.append(tower.ln_post.register_forward_hook(keep("ln_post")))
    try:
        with torch.no_grad():
            tower(features, mask, cu_seqlens=cu_seqlens)
    finally:
        for handle in handles:
            handle.remove()
    clamp = {}
    for index in range(len(tower.layers)):
        p = f"{index}."
        residual = captured[p + "input"] + captured[p + "attn"]
        captured[p + "attn_residual"] = residual
        captured[p + "pre_clamp"] = residual + captured[p + "fc2"]
        pre = captured[p + "pre_clamp"].float()
        clamp[index] = {"over_clamp": int((pre.abs() > CLAMP).sum()), "max_abs": float(pre.abs().max())}
    captured["stem"] = captured["0.input"]
    return captured, clamp


def isolated(layer, onnx_out: dict, index: int, cu_seqlens) -> dict:
    """Apply each native sub-block to the ONNX tensors that feed it."""
    t = lambda key: torch.from_numpy(onnx_out[f"{index}.{key}"])  # noqa: E731
    previous = torch.from_numpy(onnx_out["stem"] if index == 0 else onnx_out[f"{index - 1}.out"])
    attention = layer.self_attn
    with torch.no_grad():
        native = {
            "attn_norm": layer.self_attn_layer_norm(previous),
            "q": attention.q_proj(t("attn_norm")),
            "k": attention.k_proj(t("attn_norm")),
            "v": attention.v_proj(t("attn_norm")),
            "context": window_attention(attention, t("q"), t("k"), t("v"), cu_seqlens),
            "attn": attention.out_proj(t("context")),
            "attn_residual": previous + t("attn"),
            "mlp_norm": layer.final_layer_norm(t("attn_residual")),
            "fc1": layer.fc1(t("mlp_norm")),
            "act": layer.activation_fn(t("fc1")),
            "fc2": layer.fc2(t("act")),
            "pre_clamp": t("attn_residual") + t("fc2"),
            "out": torch.clamp(t("pre_clamp"), min=-CLAMP, max=CLAMP),
        }
    return {key: metric(value.numpy(), onnx_out[f"{index}.{key}"]) for key, value in native.items()}


def fp32_control_clamp(tower, features, mask, cu_seqlens) -> dict:
    captured, handles = {}, []
    for index, layer in enumerate(tower.layers):
        handles.append(layer.register_forward_pre_hook(
            lambda m, args, i=index: captured.__setitem__((i, "input"), args[0].detach().clone())))
        handles.append(layer.self_attn.register_forward_hook(
            lambda m, args, out, i=index: captured.__setitem__((i, "attn"), out.detach().clone())))
        handles.append(layer.fc2.register_forward_hook(
            lambda m, args, out, i=index: captured.__setitem__((i, "fc2"), out.detach().clone())))
    try:
        with torch.no_grad():
            tower(features, mask, cu_seqlens=cu_seqlens)
    finally:
        for handle in handles:
            handle.remove()
    rows = {}
    for index in range(len(tower.layers)):
        pre = captured[(index, "input")] + captured[(index, "attn")] + captured[(index, "fc2")]
        rows[index] = {"over_clamp": int((pre.abs() > CLAMP).sum()), "max_abs": float(pre.abs().max())}
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[100, 500, 1100])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    model, _ = load(args.model_dir)
    tower, projector = model.model.audio_tower, model.model.multi_modal_projector
    source = args.artifact_dir / "audio_front_model.onnx"
    graph = onnx.load(str(source), load_external_data=False).graph
    taps = tap_names(graph, len(tower.layers))
    keys = list(taps)
    inputs = {}
    for frames in args.frames:
        names = (f"features-{frames}.npy", f"mask-{frames}.npy", f"schedule-{frames}.bin")
        inputs[frames] = (
            np.load(args.artifact_dir / names[0]), np.load(args.artifact_dir / names[1]),
            np.fromfile(args.artifact_dir / names[2], dtype=np.int32),
            {name: _digest(args.artifact_dir / name) for name in names},
        )

    report = {
        "status": "LAYER_PROBE_MEASURED_NOT_NUMERICAL_ACCEPTANCE",
        "comparator": f"tools/tensor_parity.py::compare_tensors(reference=native_fp16, candidate=onnx, atol={ATOL}, rtol={RTOL})",
        "clamp_threshold": CLAMP,
        "artifact": {"path": str(source), "sha256": _digest(source)},
        "source_sha256": _digest(Path(__file__)),
        "versions": {"onnxruntime": ort.__version__, "torch": torch.__version__, "onnx": onnx.__version__},
        "providers": ["CPUExecutionProvider"],
        "lengths": {},
    }
    with tempfile.TemporaryDirectory(dir=args.artifact_dir.parent) as directory:
        tapped = tapped_graph(source, Path(directory), taps)
        plain = {level: session(source, level) for level in LEVELS}
        probes = {level: session(tapped, level) for level in LEVELS}
        for frames, (features, mask, schedule, digests) in inputs.items():
            feed = {"input_features": features, "input_features_mask": mask, "cu_seqlens": schedule}
            cu_seqlens = torch.from_numpy(schedule.copy())
            half = torch.from_numpy(features).to(torch.float16)
            native, clamp = native_taps(tower, half, torch.from_numpy(mask), cu_seqlens)
            with torch.no_grad():
                native_embeds = projector(native["ln_post"]).float().numpy()
            row = {"inputs": digests, "native_fp16_clamp": clamp, "levels": {}}
            for level in LEVELS:
                baseline = plain[level].run(None, feed)[0]
                results = probes[level].run(["audio_embeds"] + [taps[k] for k in keys], feed)
                onnx_out = dict(zip(keys, results[1:]))
                layers = {}
                for index in range(len(tower.layers)):
                    layers[index] = {
                        "cumulative": {
                            key: metric(native[f"{index}.{key}"].numpy(), onnx_out[f"{index}.{key}"])
                            for key in list(NODE_TAPS) + ["context", "act"]
                        },
                        "isolated": isolated(tower.layers[index], onnx_out, index, cu_seqlens),
                    }
                first_cumulative = next(
                    (i for i, r in layers.items() if not r["cumulative"]["out"]["allclose"]), None)
                first_isolated = next(
                    ((i, k) for i, r in layers.items() for k, m in r["isolated"].items() if not m["allclose"]), None)
                row["levels"][level] = {
                    "tap_validity_bitwise_equal_untapped": bool(np.array_equal(baseline, results[0])),
                    "audio_embeds_vs_native_fp16": compare_tensors(native_embeds, results[0], atol=ATOL, rtol=RTOL),
                    "stem": metric(native["stem"].numpy(), onnx_out["stem"]),
                    "ln_post": metric(native["ln_post"].numpy(), onnx_out["ln_post"]),
                    "first_cumulative_out_fail_layer": first_cumulative,
                    "first_isolated_fail": list(first_isolated) if first_isolated else None,
                    "layers": layers,
                }
            report["lengths"][str(frames)] = row
            print(f"frames={frames} done", file=sys.stderr, flush=True)
        del plain, probes

    model.float()
    for frames, (features, mask, schedule, _) in inputs.items():
        control = torch.from_numpy(features).half().float()
        report["lengths"][str(frames)]["fp32_control_clamp"] = fp32_control_clamp(
            tower, control, torch.from_numpy(mask), torch.from_numpy(schedule.copy()))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
