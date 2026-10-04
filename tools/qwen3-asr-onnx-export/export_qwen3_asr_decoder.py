"""Export the Qwen3-ASR language model as a single autoregressive step.

The decoder is a standard 28-layer causal transformer with a KV cache, so it is
the part that exports cleanly. Past state is passed as one stacked tensor per
attention kind rather than one tensor per layer, which keeps the interface small
and stops the graph from carrying 28 anonymous state inputs.

Run with the transformers 5.x environment:
    tools/model-debugging/reference/transformers5/Scripts/python.exe \\
        tools/qwen3-asr-onnx-export/export_qwen3_asr_decoder.py --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers

# Opset 18 is the floor for the scaled_dot_product_attention symbolic with
# grouped-query attention, which this model uses at 16 heads over 8 KV heads.
OPSET = 20


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_cache(config, past_key, past_value):
    """Rebuild a real DynamicCache from the stacked past tensors."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=config)
    for index in range(config.num_hidden_layers):
        cache.update(past_key[index], past_value[index], index)

    return cache


def _positions(past_key, inputs_embeds):
    """Cache position of the first new token, which is the past length."""
    if past_key is None:
        return torch.arange(inputs_embeds.shape[1]).unsqueeze(0)
    return torch.arange(
        past_key.shape[-2], past_key.shape[-2] + inputs_embeds.shape[1]
    ).unsqueeze(0)


class DecoderStep(torch.nn.Module):
    """One decode step over inputs_embeds with a KV cache."""

    def __init__(self, language_model: torch.nn.Module, projection: torch.nn.Module, config) -> None:
        super().__init__()
        self.language_model = language_model
        self.projection = projection
        self.config = config

    def forward(self, inputs_embeds, past_key, past_value):
        # The cache is rebuilt from the stacked past tensors so the step can
        # consume its own present outputs. A DynamicCache object cannot cross the
        # tracer boundary, so layers are addressed by index.
        cache = None
        if past_key is not None:
            cache = _build_cache(self.config, past_key, past_value)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            past_key_values=cache if past_key is not None else None,
            use_cache=True,
            cache_position=_positions(past_key, inputs_embeds),
        )
        hidden = outputs.last_hidden_state
        cache = outputs.past_key_values
        # A DynamicCache object cannot cross the tracer boundary, so key and value
        # are stacked per layer into single tensors.
        keys = torch.stack([layer.keys for layer in cache.layers])
        values = torch.stack([layer.values for layer in cache.layers])
        return self.projection(hidden), keys, values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-trace", action="store_true")
    args = parser.parse_args(argv)

    # Eager attention avoids fused scaled_dot_product_attention, whose symbolic
    # conversion does not support grouped-query attention at any opset. The maths is
    # unchanged; only the kernel differs.
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads

    step = DecoderStep(model.model.language_model, model.lm_head, config).eval()

    embeds = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)
    # The past inputs are exported as real zero-length tensors rather than None,
    # so the tracer records those axes instead of dropping them.
    empty_past = torch.zeros(layers, 1, config.num_key_value_heads, 0, head_dim, dtype=torch.float32)
    with torch.no_grad():
        logits, present_key, present_value = step(embeds, empty_past, empty_past)
        # Second step consumes the first step's present state, so the cache path
        # is exercised rather than assumed.
        second = torch.randn(1, 1, config.hidden_size, dtype=torch.float32)
        step_logits2, present_key2, present_value2 = step(second, present_key, present_value)
        reference_cache = _build_cache(config, present_key, present_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "decoder_step_model.onnx"
    torch.onnx.export(
        step,
        (embeds, empty_past, empty_past),
        str(output),
        input_names=["inputs_embeds", "past_key", "past_value"],
        output_names=["logits", "present_key", "present_value"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "tokens"},
            "past_key": {1: "batch", 3: "past_length"},
            "past_value": {1: "batch", 3: "past_length"},
            "logits": {0: "batch", 1: "tokens"},
            "present_key": {1: "batch", 3: "past_length"},
            "present_value": {1: "batch", 3: "past_length"},
        },
        opset_version=OPSET,
        dynamo=not args.legacy_trace,
        # The 28-layer model exceeds the 2 GiB protobuf limit, so weights go to a
        # sidecar next to the graph, which is what the rest of this repository does.
        external_data=True,
    )
    # The torch exporter writes one file per large tensor; consolidate into a
    # single sidecar the way the rest of this repository stores its graphs.
    import onnx as _onnx

    # Load WITH the external data, otherwise the tensors are empty and the
    # consolidated sidecar comes out with no weights in it.
    reloaded = _onnx.load(str(output), load_external_data=True)
    _onnx.save_model(
        reloaded,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.with_suffix(".onnx.data").name,
        size_threshold=1024,
        convert_attribute=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    produced = session.run(
        None,
        {
            "inputs_embeds": embeds.numpy(),
            "past_key": empty_past.numpy(),
            "past_value": empty_past.numpy(),
        },
    )[0]
    difference = np.abs(logits.numpy() - produced)
    # Cache round-trip: feeding the present state back must reproduce PyTorch step 2.
    cached = session.run(
        None,
        {
            "inputs_embeds": second.numpy(),
            "past_key": present_key.numpy(),
            "past_value": present_value.numpy(),
        },
    )[0]
    cache_difference = float(np.abs(step_logits2.numpy() - cached).max())

    receipt = {
        "artifact": output.name,
        "bytes": output.stat().st_size,
        "sha256": _digest(output),
        "opset": OPSET,
        "exporter": "torch.onnx.export(dynamo=False)" if args.legacy_trace else "torch.onnx.export(dynamo=True)",
        "geometry": {"layers": layers, "heads": heads, "head_dim": head_dim, "hidden": config.hidden_size},
        "outputs": {
            "logits": list(produced.shape),
            "present_key": list(present_key.shape),
            "present_value": list(present_value.shape),
            "cacheLayout": "[layers, batch, heads, past_length, head_dim], layer-major",
        },
        "parity": {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "allclose_atol_1e-4_rtol_1e-3": bool(np.allclose(logits.numpy(), produced, atol=1e-4, rtol=1e-3)),
        },
        "cache_parity": {
            "max_abs_error_second_step": cache_difference,
            "allclose_atol_1e-4_rtol_1e-3": bool(
                np.allclose(step_logits2.numpy(), cached, atol=1e-4, rtol=1e-3)
            ),
            "note": "the present state is fed back through the graph and compared with the PyTorch second step",
        },
        "attention": "eager, so fused SDPA with grouped-query attention is avoided; the exported maths is unchanged",
        "open_problem": "the exported step takes no past state: it is a first step only, so autoregressive decoding over more than one token still needs a cache-carrying interface",
    }
    (args.output_dir / "decoder-step-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
