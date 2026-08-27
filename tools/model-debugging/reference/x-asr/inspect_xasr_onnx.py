"""Inspect official X-ASR sherpa-onnx graphs and emit a library graph contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as rt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect X-ASR encoder/decoder/joiner IO.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"N:\models\x-asr\zh-en\chunk-160ms-model"),
    )
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def meta(session: rt.InferenceSession, kind: str) -> list[dict[str, object]]:
    nodes = session.get_inputs() if kind == "input" else session.get_outputs()
    return [{"name": item.name, "type": item.type, "shape": [str(dim) for dim in item.shape]} for item in nodes]


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    graphs = {}
    for name in ("encoder", "decoder", "joiner"):
        path = model_dir / f"{name}-{args.chunk_ms}ms.onnx"
        session = rt.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        graphs[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "inputs": meta(session, "input"),
            "outputs": meta(session, "output"),
        }

    encoder_inputs = graphs["encoder"]["inputs"]
    feature_names = {item["name"] for item in encoder_inputs}
    feature_input = next((name for name in ("x", "features", "audio_signal") if name in feature_names), encoder_inputs[0]["name"])
    length_input = next((name for name in ("x_lens", "features_lens", "length") if name in feature_names), None)
    state_inputs = [item for item in encoder_inputs if item["name"] not in {feature_input, length_input}]
    encoder_outputs = graphs["encoder"]["outputs"]
    encoder_out = next((item["name"] for item in encoder_outputs if "encoder_out" in item["name"] or item["name"] in {"encoder_out", "out", "y"}), encoder_outputs[0]["name"])
    state_outputs = [item["name"] for item in encoder_outputs if item["name"] != encoder_out]

    contract = {
        "featureInputName": feature_input,
        "featureLengthInputName": length_input,
        "encoderOutputName": encoder_out,
        "encoderStateInputs": [
            {
                "name": item["name"],
                "type": "float32" if "float" in str(item["type"]) else "int64" if "int64" in str(item["type"]) else "int32",
                "shape": item["shape"],
            }
            for item in state_inputs
        ],
        "encoderStateOutputs": state_outputs,
        "decoderInputs": [item["name"] for item in graphs["decoder"]["inputs"]],
        "decoderOutputs": [item["name"] for item in graphs["decoder"]["outputs"]],
        "joinerInputs": [item["name"] for item in graphs["joiner"]["inputs"]],
        "joinerOutputs": [item["name"] for item in graphs["joiner"]["outputs"]],
        "note": "encoderFrameSize/Shift must be confirmed from sherpa-onnx zipformer2 chunk config, not guessed from ONNX dynamic axes alone.",
    }
    payload = {"schema_version": 1, "chunk_ms": args.chunk_ms, "graphs": graphs, "library_contract": contract}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contract, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
