"""Score native fp16 and ONNX fp16 decoder steps against the fp32 original-weight oracle.

All three paths receive identical fp32 inputs: the support prompt embeddings
around fp32-oracle audio embeddings for the JFK fixture, then the fp32 oracle's
greedy tokens (teacher forcing). Each path carries its own KV cache, so cache
drift accumulates as it does in a real decode. The fp16 relative gate is: for
logits and final present key/value, the ONNX max and mean absolute error to the
oracle must not exceed native fp16's, and every step's argmax must equal the
oracle token. Matching argmax under teacher forcing implies identical free
greedy decoding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from export_qwen3_asr_fp16 import Float32DecoderStep, _digest  # noqa: E402
from tensor_parity import compare_tensors  # noqa: E402

ATOL, RTOL = 1e-4, 1e-3
MAX_STEPS = 40


def load(model_dir: Path, dtype):
    return transformers.AutoModelForMultimodalLM.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager", local_files_only=True).eval()


def relative(onnx_row: dict, native_row: dict) -> bool:
    return (onnx_row["max_abs_error"] <= native_row["max_abs_error"]
            and onnx_row["mean_abs_error"] <= native_row["mean_abs_error"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = args.artifact_dir
    eos = set(json.loads((args.model_dir / "generation_config.json").read_text())["eos_token_id"])

    oracle_model = load(args.model_dir, torch.float32)
    features = torch.from_numpy(np.load(artifact / f"features-{args.frames}.npy"))
    mask = torch.from_numpy(np.load(artifact / f"mask-{args.frames}.npy"))
    schedule = torch.from_numpy(np.fromfile(artifact / f"schedule-{args.frames}.bin", dtype=np.int32))
    with torch.no_grad():
        hidden = oracle_model.model.audio_tower(features, mask, cu_seqlens=schedule).last_hidden_state
        audio = oracle_model.model.multi_modal_projector(hidden).float().numpy()[None]
    prefix = np.load(artifact / "prompt_prefix_embeds.npy").astype(np.float32)
    suffix = np.load(artifact / "prompt_suffix_embeds.npy").astype(np.float32)
    prompt = np.concatenate([prefix, audio, suffix], axis=1)
    embed = oracle_model.model.language_model.embed_tokens
    config = oracle_model.config.text_config
    empty = np.zeros((config.num_hidden_layers, 1, config.num_key_value_heads, 0, config.head_dim), np.float32)

    native_model = load(args.model_dir, torch.float16)
    steps = {"oracle": Float32DecoderStep(oracle_model).eval(), "native": Float32DecoderStep(native_model).eval()}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    decoder_path = artifact / "decoder_step_model.onnx"
    session = ort.InferenceSession(str(decoder_path), options, providers=["CPUExecutionProvider"])

    def run(path, embeds, cache):
        if path == "onnx":
            return session.run(None, {"inputs_embeds": embeds, "past_key": cache[0], "past_value": cache[1]})
        with torch.no_grad():
            out = steps[path](torch.from_numpy(embeds), torch.from_numpy(cache[0]), torch.from_numpy(cache[1]))
        return [value.numpy() for value in out]

    caches = {path: (empty, empty) for path in ("oracle", "native", "onnx")}
    embeds, tokens, rows = prompt, [], []
    for step in range(MAX_STEPS + 1):
        outputs = {}
        for path in caches:
            logits, key, value = run(path, embeds, caches[path])
            caches[path] = (key, value)
            outputs[path] = logits[:, -1]
        token = int(outputs["oracle"].argmax())
        rows.append({
            "step": step, "oracle_token": token,
            "native_argmax": int(outputs["native"].argmax()), "onnx_argmax": int(outputs["onnx"].argmax()),
            "native": compare_tensors(outputs["oracle"], outputs["native"], atol=ATOL, rtol=RTOL),
            "onnx": compare_tensors(outputs["oracle"], outputs["onnx"], atol=ATOL, rtol=RTOL),
        })
        tokens.append(token)
        if token in eos:
            break
        with torch.no_grad():
            embeds = embed(torch.tensor([[token]])).float().numpy()

    def aggregate(path):
        return {
            "max_abs_error": max(r[path]["max_abs_error"] for r in rows),
            "mean_abs_error": float(np.mean([r[path]["mean_abs_error"] for r in rows])),
            "min_cosine": min(r[path]["cosine_similarity"] for r in rows),
            "allclose_steps": sum(r[path]["allclose"] for r in rows),
            "argmax_equal_steps": sum(r[f"{path}_argmax"] == r["oracle_token"] for r in rows),
        }

    logits = {path: aggregate(path) for path in ("native", "onnx")}
    cache = {}
    for index, name in enumerate(("present_key", "present_value")):
        cache[name] = {
            path: compare_tensors(caches["oracle"][index], caches[path][index], atol=ATOL, rtol=RTOL)
            for path in ("native", "onnx")
        }
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    gate = {
        "logits_relative": relative(logits["onnx"], logits["native"]),
        "present_key_relative": relative(cache["present_key"]["onnx"], cache["present_key"]["native"]),
        "present_value_relative": relative(cache["present_value"]["onnx"], cache["present_value"]["native"]),
        "onnx_argmax_equals_oracle_every_step": logits["onnx"]["argmax_equal_steps"] == len(rows),
    }
    report = {
        "status": "DECODER_RELATIVE_GATE_" + ("PASS" if all(gate.values()) else "FAIL"),
        "gate": gate,
        "gate_definition": __doc__.split("The fp16 relative gate is: ")[1].split(" Matching")[0],
        "comparator": f"tools/tensor_parity.py::compare_tensors(reference=fp32_oracle, candidate, atol={ATOL}, rtol={RTOL})",
        "oracle": "original bf16 checkpoint weights loaded as fp32, fp32 compute, eager attention",
        "artifact": {"path": str(decoder_path), "sha256": _digest(decoder_path)},
        "source_sha256": _digest(Path(__file__)),
        "versions": {"onnxruntime": ort.__version__, "torch": torch.__version__, "transformers": transformers.__version__},
        "prompt_tokens": int(prompt.shape[1]), "decode_steps": len(rows),
        "oracle_text": tokenizer.decode([t for t in tokens if t not in eos], skip_special_tokens=False),
        "oracle_tokens": tokens,
        "logits": logits, "cache": cache, "steps": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "gate", "logits", "oracle_text")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
