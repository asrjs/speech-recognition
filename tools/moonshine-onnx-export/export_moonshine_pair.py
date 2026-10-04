"""Export the Moonshine encoder, decoder and cross-attention graphs from the checkpoint.

The browser graphs shipped with the model carry weights from a different
checkpoint revision of the same architecture: every comparable encoder tensor
differs, which puts the encoder output far off PyTorch. Exporting our own pair
removes that dependency.

The encoder only traces with the dynamo exporter, which needs ``onnxscript``.
The legacy tracer fails with ``All tensors must have the same rank`` because
Moonshine's third conv yields a rank-4 tensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

OPSET = 18


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_encoder(model, output_dir: Path, frames: int) -> dict:
    """Conv subsampler plus Conformer encoder, dynamic batch and time axes."""

    output = output_dir / "encoder_model.onnx"
    example = torch.randn(1, frames * 320, dtype=torch.float32)
    torch.onnx.export(
        model.model.encoder,
        (example,),
        output,
        input_names=["input_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "samples"},
            "last_hidden_state": {0: "batch", 1: "frames"},
        },
        opset_version=OPSET,
        dynamo=True,
    )

    return {"artifact": output.name, "bytes": output.stat().st_size, "sha256": _digest(output)}


def export_cross_kv(model, output_dir: Path, frames: int) -> dict:
    """Project encoder states into per-layer cross-attention key and value.

    Cross-attention memory is a fixed projection of the encoder states, so this
    graph replaces a per-utterance seed cache and works at any frame count.
    """

    projections = [
        (layer.encoder_attn.k_proj, layer.encoder_attn.v_proj)
        for layer in model.model.decoder.layers
    ]

    class CrossKv(torch.nn.Module):
        def __init__(self, pairs):
            super().__init__()
            self.keys = torch.nn.ModuleList([pair[0] for pair in pairs])
            self.values = torch.nn.ModuleList([pair[1] for pair in pairs])

        def forward(self, encoder_states):
            outputs = []
            for key, value in zip(self.keys, self.values):
                outputs.append(key(encoder_states))
                outputs.append(value(encoder_states))

            return tuple(outputs)

    wrapper = CrossKv(projections).eval()
    output = output_dir / "decoder_cross_kv_model.onnx"
    example = torch.randn(1, frames, model.config.hidden_size, dtype=torch.float32)
    names = [f"layer{i}.{role}" for i in range(len(wrapper.keys)) for role in ("k", "v")]
    torch.onnx.export(
        wrapper,
        (example,),
        output,
        input_names=["encoder_states"],
        output_names=names,
        dynamic_axes={"encoder_states": {1: "frames"}, **{name: {1: "frames"} for name in names}},
        opset_version=OPSET,
        dynamo=True,
    )

    return {"artifact": output.name, "bytes": output.stat().st_size, "sha256": _digest(output)}


def export_decoder(model, output_dir: Path, frames: int) -> dict:
    """One autoregressive step with an explicit past-state interface.

    Past state is passed as two stacked tensors rather than 24 named ones: the
    dynamo exporter traces a *args wrapper as a single argument, and per-tensor
    names contain dots that are not valid Python identifiers.
    """

    from transformers.cache_utils import EncoderDecoderCache

    layers = len(model.model.decoder.layers)
    heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // heads
    slots = layers * 2

    class Step(torch.nn.Module):
        def __init__(self, decoder, projection):
            super().__init__()
            self.decoder = decoder
            # Logits live at model level: the decoder returns hidden states and
            # the vocabulary projection is a separate module.
            self.projection = projection

        def forward(self, input_ids, self_past, cross_past, encoder_states):
            # encoder_hidden_states must be real: MoonshineAttention gates the
            # whole cross-attention block on it, and passing None traces every
            # layer down the self-attention branch, which yields a decoder that
            # silently ignores its cross-attention memory.
            outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=None,
                use_cache=True,
                past_key_values=EncoderDecoderCache.from_legacy_cache(
                    [
                        (
                            self_past[layer * 2],
                            self_past[layer * 2 + 1],
                            cross_past[layer * 2],
                            cross_past[layer * 2 + 1],
                        )
                        for layer in range(layers)
                    ]
                ),
            )
            legacy = outputs.past_key_values.to_legacy_cache()
            present_self = []
            present_cross = []
            # to_legacy_cache returns one (self_key, self_value, cross_key,
            # cross_value) tuple per layer.
            for layer in range(layers):
                self_key, self_value, cross_key, cross_value = legacy[layer]
                present_self.append(self_key)
                present_self.append(self_value)
                present_cross.append(cross_key)
                present_cross.append(cross_value)

            return (
                self.projection(outputs.last_hidden_state),
                torch.stack(present_self),
                torch.stack(present_cross),
            )

    wrapper = Step(model.model.decoder, model.proj_out).eval()
    output = output_dir / "decoder_with_past_model.onnx"
    from torch.export import Dim

    tokens = Dim("tokens")
    past_length = Dim("past_decoder_sequence_length")
    encoder_length = Dim("encoder_sequence_length_out")
    example = (
        torch.tensor([[1]], dtype=torch.int64),
        torch.zeros(slots, 1, heads, 1, head_dim),
        torch.zeros(slots, 1, heads, frames, head_dim),
        torch.zeros(1, frames, model.config.hidden_size),
    )
    dynamic_shapes = (
        {1: tokens},
        {0: Dim("slots"), 3: past_length},
        {0: Dim("slots"), 3: encoder_length},
        {1: encoder_length},
    )
    torch.onnx.export(
        wrapper,
        example,
        output,
        input_names=["input_ids", "self_past", "cross_past", "encoder_states"],
        output_names=["logits", "self_present", "cross_present"],
        dynamic_shapes=dynamic_shapes,
        opset_version=OPSET,
        dynamo=True,
    )

    return {
        "artifact": output.name,
        "bytes": output.stat().st_size,
        "sha256": _digest(output),
        "interface": {
            "slots": slots,
            "heads": heads,
            "headDim": head_dim,
            "selfPast": "stacked [slots, batch, heads, past_length, headDim]",
            "crossPast": "stacked [slots, batch, heads, encoder_length, headDim]",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/moonshine-tiny"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=457)
    args = parser.parse_args(argv)

    from transformers import MoonshineForConditionalGeneration

    model = MoonshineForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    receipt = {
        "schemaVersion": 1,
        "source": str(args.model_dir),
        "opset": OPSET,
        "exporter": "torch.onnx.export(dynamo=True)",
        "hiddenSize": model.config.hidden_size,
        "decoderLayers": len(model.model.decoder.layers),
        "framesAtExport": args.frames,
        "graphs": {
            "encoder": export_encoder(model, args.output_dir, args.frames),
            "crossKv": export_cross_kv(model, args.output_dir, args.frames),
            "decoder": export_decoder(model, args.output_dir, args.frames),
        },
        "why": (
            "The browser graphs shipped with the model carry weights from a different "
            "checkpoint revision, so the encoder output drifts from PyTorch. These graphs "
            "are exported from the validated checkpoint instead."
        ),
    }
    (args.output_dir / "export-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())