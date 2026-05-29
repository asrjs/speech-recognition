#!/usr/bin/env python3
"""Tests for the 4-graph KV-cache decoder export.

Validates that export_all() produces:
  1. encoder_model.onnx
  2. decoder_init.onnx
  3. decoder_step.onnx
  4. decoder_align.onnx
  5. manifest.json with correct structure

Usage:
  python test_kv_export.py [--model openai/whisper-tiny] [--output-dir /tmp/test-export]
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import onnx
import onnxruntime as ort

# Add the export tool dir to path so we can import export_whisper
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Will import after sys.path setup
# from export_whisper import export_all  # current version (RED)
# from export_whisper import export_all  # new version (GREEN)


def load_onnx_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path))


def get_io_names(model: onnx.ModelProto) -> Dict[str, List[str]]:
    return {
        "inputs": [n.name for n in model.graph.input],
        "outputs": [n.name for n in model.graph.output],
    }


def get_io_shapes(model: onnx.ModelProto) -> Dict[str, List[str]]:
    def shape_str(n):
        return [str(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
                for d in n.type.tensor_type.shape.dim]
    return {
        "inputs": shape_str(model.graph.input[0]) if model.graph.input else [],
        "outputs": shape_str(model.graph.output[0]) if model.graph.output else [],
    }


def validate_encoder(export_dir: Path):
    """Validates encoder_model.onnx."""
    path = export_dir / "encoder_model.onnx"
    assert path.exists(), f"Missing: {path}"

    model = load_onnx_model(path)
    io = get_io_names(model)

    assert "input_features" in io["inputs"], (
        f"encoder_model.onnx missing 'input_features' input. Got: {io['inputs']}"
    )
    assert "last_hidden_state" in io["outputs"], (
        f"encoder_model.onnx missing 'last_hidden_state' output. Got: {io['outputs']}"
    )

    print(f"  ✓ encoder_model.onnx — inputs={io['inputs']}, outputs={io['outputs'][:1]}...")


def validate_decoder_init(export_dir: Path, expected_layers: int):
    """Validates decoder_init.onnx."""
    path = export_dir / "decoder_init.onnx"
    assert path.exists(), f"Missing: {path}"

    model = load_onnx_model(path)
    io = get_io_names(model)

    assert "input_ids" in io["inputs"], (
        f"decoder_init.onnx missing 'input_ids'. Got: {io['inputs']}"
    )
    assert "encoder_hidden_states" in io["inputs"], (
        f"decoder_init.onnx missing 'encoder_hidden_states'. Got: {io['inputs']}"
    )
    assert "logits" in io["outputs"], (
        f"decoder_init.onnx missing 'logits'. Got: {io['outputs']}"
    )

    # Verify KV cache outputs exist for all layers
    for i in range(expected_layers):
        for kv_type in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
            name = f"present.{i}.{kv_type}"
            assert name in io["outputs"], (
                f"decoder_init.onnx missing '{name}'. Got: {io['outputs']}"
            )

    print(f"  ✓ decoder_init.onnx — outputs: logits + {expected_layers}×4 KV tensors")


def validate_decoder_step(export_dir: Path, expected_layers: int):
    """Validates decoder_step.onnx.

    The step model uses KV cache: encoder_hidden_states is NOT needed because
    cross-attention K/V are provided via past_key_values.{i}.encoder.{key,value}.
    cache_position is derived internally from cache sequence length.
    """
    path = export_dir / "decoder_step.onnx"
    assert path.exists(), f"Missing: {path}"

    model = load_onnx_model(path)
    io = get_io_names(model)

    assert "input_ids" in io["inputs"], (
        f"decoder_step.onnx missing 'input_ids'. Got: {io['inputs'][:3]}..."
    )
    # encoder_hidden_states and cache_position are NOT needed because KV cache
    # provides cross-attention K/V and position is derived from cache length.
    assert "logits" in io["outputs"], (
        f"decoder_step.onnx missing 'logits'. Got: {io['outputs'][:3]}..."
    )

    # Verify past KV inputs (both decoder and encoder) and present KV outputs
    for i in range(expected_layers):
        for kv_type in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"]:
            name = f"past_key_values.{i}.{kv_type}"
            assert name in io["inputs"], (
                f"decoder_step.onnx missing past input '{name}'."
            )
    for i in range(expected_layers):
        for kv_type in ["decoder.key", "decoder.value"]:
            name = f"present.{i}.{kv_type}"
            assert name in io["outputs"], (
                f"decoder_step.onnx missing present output '{name}'."
            )

    # Cross-attention KV should NOT be in step outputs (kept from init)
    for i in range(expected_layers):
        assert f"present.{i}.encoder.key" not in io["outputs"], (
            f"decoder_step.onnx should NOT output encoder KV."
        )

    print(f"  ✓ decoder_step.onnx — input_ids + {expected_layers}×4 past KV → logits + {expected_layers}×2 present KV")


def validate_decoder_align(export_dir: Path):
    """Validates decoder_align.onnx — optional, may not exist."""
    path = export_dir / "decoder_align.onnx"
    if not path.exists():
        print(f"  - decoder_align.onnx — SKIPPED (not exported, word timestamps use fallback)")
        return

    model = load_onnx_model(path)
    io = get_io_names(model)

    assert "input_ids" in io["inputs"], (
        f"decoder_align.onnx missing 'input_ids'. Got: {io['inputs']}"
    )
    assert "encoder_hidden_states" in io["inputs"], (
        f"decoder_align.onnx missing 'encoder_hidden_states'. Got: {io['inputs']}"
    )
    assert "alignment" in io["outputs"], (
        f"decoder_align.onnx missing 'alignment'. Got: {io['outputs']}"
    )

    print(f"  ✓ decoder_align.onnx — inputs={io['inputs']}, outputs={io['outputs']}")


def validate_manifest(export_dir: Path, model_id: str, expected_layers: int, expected_heads: int):
    """Validates manifest.json."""
    path = export_dir / "manifest.json"
    assert path.exists(), f"Missing: {path}"

    with open(path) as f:
        manifest = json.load(f)

    assert manifest["model_id"] == model_id, f"model_id mismatch: {manifest['model_id']}"
    assert manifest["format"] == "whisper-browser-self-export-v1", (
        f"format mismatch: {manifest.get('format')}"
    )
    assert manifest["decoder_layers"] == expected_layers, (
        f"decoder_layers: {manifest.get('decoder_layers')} != {expected_layers}"
    )
    assert manifest["decoder_attention_heads"] == expected_heads, (
        f"decoder_attention_heads: {manifest.get('decoder_attention_heads')} != {expected_heads}"
    )
    assert "alignment_heads" in manifest, "Missing alignment_heads"
    assert "special_tokens" in manifest, "Missing special_tokens"
    assert "artifacts" in manifest, "Missing artifacts"

    artifacts = manifest["artifacts"]
    def artifact_file(key: str) -> str:
        """Handle both old (string) and new ({file, externalData?}) artifact formats."""
        val = artifacts.get(key)
        if isinstance(val, dict):
            return val.get("file", "")
        return val or ""
    assert artifact_file("encoder") == "encoder_model.onnx", f"encoder artifact: {artifacts.get('encoder')}"
    assert artifact_file("decoder_init") == "decoder_init.onnx", f"decoder_init artifact: {artifacts.get('decoder_init')}"
    assert artifact_file("decoder_step") == "decoder_step.onnx", f"decoder_step artifact: {artifacts.get('decoder_step')}"
    assert artifact_file("decoder_align") == "decoder_align.onnx", f"decoder_align artifact: {artifacts.get('decoder_align')}"

    # Special tokens sanity
    st = manifest["special_tokens"]
    assert st.get("eos_token_id") == 50257, f"eos_token_id: {st.get('eos_token_id')}"
    assert "timestamp_begin" in st, f"timestamp_begin missing from special_tokens"

    has_align = "decoder_align" in artifacts
    align_note = " (without decoder_align)" if not has_align else ""
    print(f"  ✓ manifest.json — layers={expected_layers}, heads={expected_heads}, {len(artifacts)} artifacts{align_note}")


def test_export_all_4graph():
    """Main test: run export_all and validate all 4 graphs + manifest."""
    from export_whisper import export_all

    model_id = "openai/whisper-tiny"
    print(f"\n--- Exporting {model_id} to 4-graph format ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "whisper-tiny-4graph"
        output_dir.mkdir()

        export_all(
            model_id=model_id,
            output_dir=output_dir,
            opset=17,
            prompt_len=4,
            past_len=4,
        )

        print(f"\n--- Validating output in {output_dir} ---")

        expected_layers = 4   # whisper-tiny
        expected_heads = 6

        validate_encoder(output_dir)
        validate_decoder_init(output_dir, expected_layers)
        validate_decoder_step(output_dir, expected_layers)
        validate_decoder_align(output_dir)
        validate_manifest(output_dir, model_id, expected_layers, expected_heads)

        # Also verify ONNX Runtime can load the models
        print(f"\n--- ONNX Runtime load check ---")
        for fname in ["encoder_model.onnx", "decoder_init.onnx", "decoder_step.onnx"]:
            path = output_dir / fname
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inputs = [i.name for i in sess.get_inputs()]
            outputs = [o.name for o in sess.get_outputs()]
            print(f"  ✓ ORT loaded {fname}: {len(inputs)} inputs, {len(outputs)} outputs")
        align_path = output_dir / "decoder_align.onnx"
        if align_path.exists():
            sess = ort.InferenceSession(str(align_path), providers=["CPUExecutionProvider"])
            inputs = [i.name for i in sess.get_inputs()]
            outputs = [o.name for o in sess.get_outputs()]
            print(f"  ✓ ORT loaded decoder_align.onnx: {len(inputs)} inputs, {len(outputs)} outputs")

    print(f"\n✓ All 4-graph export tests passed\n")


if __name__ == "__main__":
    test_export_all_4graph()
