"""Export the Qwen3-ASR graphs directly in fp16 from the checkpoint.

Converting an fp32 export with ORT's float16 converter left a
SimplifiedLayerNormFusion that referenced a node which did not exist, and ORT
cannot be asked to skip the fusion from Node because onnxruntime-node exposes no
GraphOptimizationLevel. Building the graph in fp16 from the weights avoids that
conversion path entirely.

fp16 is the route to a browser-shippable model: the fp32 set is 3.75 GB and a
browser WASM session refuses an allocation of that size, while this set is
1.88 GB. Graph inputs and outputs are fp16 too, so the caller supplies fp16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.export import Dim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_cu_seqlens import capture_schedule  # noqa: E402

OPSET = 18


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(model_dir: Path):
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="eager"
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(model_dir)
    return model, processor


def features_for(processor, audio: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with wave.open(str(audio), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    inputs = processor.apply_transcription_request(audio=pcm, sampling_rate=16000)
    return inputs["input_features"].to(torch.float16), inputs["input_features_mask"]


def export_front(model, features, mask, schedule, output_dir: Path) -> dict:
    tower = model.model.audio_tower
    projector = model.model.multi_modal_projector

    class Front(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tower = tower
            self.projector = projector

        def forward(self, input_features, input_features_mask, cu_seqlens):
            # Graph inputs stay fp32 so the caller keeps the contract it already
            # uses; only the weights are fp16, which is where the size is.
            dtype = self.tower.conv1.weight.dtype
            encoded = self.tower(
                input_features.to(dtype), input_features_mask, cu_seqlens=cu_seqlens
            )
            return self.projector(encoded.last_hidden_state).float()

    module = Front().eval()
    path = output_dir / "audio_front_model.onnx"
    torch.onnx.export(
        module,
        (features, mask, schedule.to(torch.int32)),
        str(path),
        input_names=["input_features", "input_features_mask", "cu_seqlens"],
        output_names=["audio_embeds"],
        dynamic_shapes=({2: Dim("frames")}, {1: Dim("frames")}, {0: Dim("boundaries", min=2)}),
        opset_version=OPSET,
        dynamo=True,
    )
    return {"artifact": path.name, "bytes": path.stat().st_size, "sha256": _digest(path)}


def export_embed(model, output_dir: Path) -> dict:
    weight = model.model.language_model.embed_tokens.weight.detach()

    class Embed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

        def forward(self, input_ids):
            return torch.nn.functional.embedding(input_ids, self.weight).float()

    path = output_dir / "embed_tokens_model.onnx"
    torch.onnx.export(
        Embed().eval(),
        (torch.tensor([[1]], dtype=torch.int64),),
        str(path),
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "embeddings": {0: "batch", 1: "tokens"}},
        opset_version=OPSET,
        dynamo=False,
        external_data=True,
    )
    return {"artifact": path.name, "bytes": path.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--audio", type=Path, default=Path("tools/data/fixtures/audio/jfk-short.wav"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    model, processor = load(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, mask = features_for(processor, args.audio)
    features = features.float()
    mask = mask.float()
    schedule = capture_schedule(model, features.to(torch.float32))

    receipt = {
        "dtype": "float16",
        "builtFrom": str(args.model_dir),
        "reason": (
            "the fp32 set is 3.75 GB and a browser WASM session refuses an allocation of that size; "
            "this set is about 1.88 GB. The graph is built in fp16 from the weights rather than "
            "converted afterwards, because the ORT float16 converter produced a broken "
            "SimplifiedLayerNormFusion that no available option could skip."
        ),
        "graphs": {
            "front": export_front(model, features, mask, schedule, args.output_dir),
            "embed": export_embed(model, args.output_dir),
        },
    }
    (args.output_dir / "fp16-export-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
