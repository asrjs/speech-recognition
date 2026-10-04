"""Export the Moonshine decoder cross-attention key/value projections as a graph.

The published `decoder_with_past_model.onnx` returns self-attention history only,
so cross-attention key and value must come from the checkpoint. That makes the
decoder unusable for any utterance longer than the exported seed, and it forces a
32 MB seed cache into the artifact set.

This module exports the same projections as a standalone graph that runs once per
utterance, so the decoder can be seeded for any length and the seed cache is no
longer a runtime dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def collect_cross_attention_projections(model):
    """Return the linear layers that project encoder states into cross-attn K/V."""
    decoder = getattr(model, "decoder", None) or model.model.decoder

    return [
        {"k": layer.encoder_attn.k_proj, "v": layer.encoder_attn.v_proj}
        for layer in decoder.layers
    ]


@torch.no_grad()
def build_trace(model, frames: int):
    """Record one forward pass and return the cross-attention K/V it produced."""
    captured: dict[str, torch.Tensor] = {}
    layers = collect_cross_attention_projections(model)
    hidden_size = model.config.hidden_size

    handles = []
    for index, entry in enumerate(layers):
        for role in ("k", "v"):
            module = entry[role]

            def hook(_module, _inputs, output, index=index, role=role):
                captured[f"layer{index}.{role}"] = output.detach()

            handles.append(module.register_forward_hook(hook))

    hidden = torch.randn(1, frames, hidden_size, dtype=torch.float32)
    try:
        layers_owner = getattr(model, "decoder", None) or model.model.decoder
        layers_owner(
            inputs_embeds=torch.zeros(1, 1, hidden_size, dtype=torch.float32),
            encoder_hidden_states=hidden,
            encoder_attention_mask=None,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    return captured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=457)
    args = parser.parse_args(argv)

    from transformers import MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    captured = build_trace(model, args.frames)
    expected = len(collect_cross_attention_projections(model)) * 2
    if len(captured) != expected:
        raise SystemExit(f"captured {len(captured)} tensors, expected {expected}")

    class CrossKv(torch.nn.Module):
        """Projects encoder states into per-layer cross-attention K/V."""

        def __init__(self, projections):
            super().__init__()
            self.keys = torch.nn.ModuleList([entry["k"] for entry in projections])
            self.values = torch.nn.ModuleList([entry["v"] for entry in projections])

        def forward(self, encoder_states):
            outputs = []
            for key, value in zip(self.keys, self.values):
                outputs.append(key(encoder_states))
                outputs.append(value(encoder_states))

            return tuple(outputs)

    wrapper = CrossKv(collect_cross_attention_projections(model)).eval()
    output = args.output_dir / "decoder_cross_kv_model.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (torch.randn(1, args.frames, model.config.hidden_size, dtype=torch.float32),),
        output,
        input_names=["encoder_states"],
        output_names=[f"layer{i}.{role}" for i in range(len(wrapper.keys)) for role in ("k", "v")],
        dynamic_axes={
            "encoder_states": {1: "frames"},
            **{f"layer{i}.{role}": {1: "frames"} for i in range(len(wrapper.keys)) for role in ("k", "v")},
        },
        opset_version=17,
        dynamo=False,
    )

    receipt = {
        "artifact": str(output),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "dModel": model.config.hidden_size,
        "decoderLayers": len(wrapper.keys),
        "framesAtExport": args.frames,
        "outputsPerLayer": ["k", "v"],
        "purpose": "Seed decoder cross-attention history for any utterance length.",
        "removesSeedCacheDependency": True,
    }
    (args.output_dir / "cross-kv-export-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
