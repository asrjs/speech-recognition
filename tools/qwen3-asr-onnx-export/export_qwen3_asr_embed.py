"""Export the Qwen3-ASR token embedding table.

The decode loop needs an embedding for each token it emits. Baking the whole
vocabulary is one graph and no per-step lookup, at the cost of a large sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx/qwen3-asr-0.6b"))
    args = parser.parse_args(argv)

    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    embed = model.model.language_model.embed_tokens
    weight = embed.weight.detach()

    class Embed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.embedding(input_ids, self.weight)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "embed_tokens_model.onnx"
    torch.onnx.export(
        Embed().eval(),
        (torch.tensor([[1]], dtype=torch.int64),),
        str(output),
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={"input_ids": {0: "batch", 1: "tokens"}, "embeddings": {0: "batch", 1: "tokens"}},
        opset_version=18,
        dynamo=False,
        external_data=True,
    )
    import onnx

    reloaded = onnx.load(str(output), load_external_data=True)
    onnx.save_model(
        reloaded,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.with_suffix(".onnx.data").name,
        size_threshold=1024,
        convert_attribute=False,
    )

    sidecar = output.with_suffix(".onnx.data")
    print(
        json.dumps(
            {
                "artifact": output.name,
                "graphBytes": output.stat().st_size,
                "sidecarBytes": sidecar.stat().st_size if sidecar.exists() else 0,
                "sidecarSha256": hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.exists() else None,
                "vocab": int(weight.shape[0]),
                "hidden": int(weight.shape[1]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
