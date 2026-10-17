"""Export the Qwen3-ASR graphs directly in fp16 from the checkpoint.

The converted candidate loads with Node graphOptimizationLevel basic/extended,
but its /Concat_2 fails during execution on past lengths zero and one in both
Node ORT 1.25-dev and Python ORT 1.30. This builder uses original weights rather
than that converter. The old candidate remains a negative control.

Floating graph inputs and outputs stay fp32 for the existing runner. Internal
weights and compute use fp16. Disk size does not establish browser feasibility;
native numerical gates and measured browser execution are separate requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_cu_seqlens import capture_schedule  # noqa: E402
from export_qwen3_asr_decoder import DecoderStep, OPSET as DECODER_OPSET  # noqa: E402

OPSET = 18


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def consolidate(path: Path) -> dict:
    """Write one fresh sidecar without appending to an existing export."""
    import onnx

    graph = onnx.load(str(path), load_external_data=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        destination = Path(directory) / path.name
        onnx.save_model(
            graph, str(destination), save_as_external_data=True,
            all_tensors_to_one_file=True, location=path.name + ".data",
            size_threshold=1024, convert_attribute=False,
        )
        sidecar = destination.with_name(path.name + ".data")
        sidecar.replace(path.with_name(sidecar.name))
        destination.replace(path)
    onnx.checker.check_model(str(path))
    data = path.with_name(path.name + ".data")
    return {
        "artifact": path.name, "bytes": path.stat().st_size, "sha256": _digest(path),
        "external_data": {"artifact": data.name, "bytes": data.stat().st_size, "sha256": _digest(data)},
    }


def load(model_dir: Path):
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="eager", local_files_only=True
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    return model, processor


def request_for(processor, audio: Path):
    with wave.open(str(audio), "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return processor.apply_transcription_request(audio=pcm, sampling_rate=16000)


def features_for(processor, audio: Path) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = request_for(processor, audio)
    return inputs["input_features"].float(), inputs["input_features_mask"]


def save_support(model, inputs, schedule, model_dir: Path, output_dir: Path) -> dict:
    """Save the processor inputs and same-builder fixed text prompt for the runner."""
    ids = inputs["input_ids"][0]
    positions = torch.nonzero(ids == model.config.audio_token_id).flatten()
    start, end = int(positions[0]), int(positions[-1]) + 1
    if positions.numel() != end - start:
        raise ValueError("The runner requires one contiguous audio placeholder span")
    embedding = model.model.language_model.embed_tokens
    with torch.no_grad():
        prefix = embedding(ids[:start].unsqueeze(0)).float().numpy()
        suffix = embedding(ids[end:].unsqueeze(0)).float().numpy()
    np.save(output_dir / "prompt_prefix_embeds.npy", prefix)
    np.save(output_dir / "prompt_suffix_embeds.npy", suffix)
    np.save(output_dir / "jfk_features.npy", inputs["input_features"].float().numpy())
    np.save(output_dir / "jfk_features_mask.npy", inputs["input_features_mask"].float().numpy())
    schedule.cpu().numpy().astype(np.int32).tofile(output_dir / "jfk_cu_seqlens.bin")
    shutil.copyfile(model_dir / "tokenizer.json", output_dir / "tokenizer.json")
    names = [
        "prompt_prefix_embeds.npy", "prompt_suffix_embeds.npy", "jfk_features.npy",
        "jfk_features_mask.npy", "jfk_cu_seqlens.bin", "tokenizer.json",
    ]
    return {
        "prefix_token_ids": ids[:start].tolist(), "suffix_token_ids": ids[end:].tolist(),
        "audio_placeholder_id": model.config.audio_token_id,
        "note": "jfk_* names preserve the existing runner convention for the --audio fixture.",
        "files": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": _digest(output_dir / name)}
            for name in names
        },
    }


@torch.jit.script
def _window_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    cu_seqlens: torch.Tensor, scaling: float,
) -> torch.Tensor:
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for index in range(cu_seqlens.shape[0] - 1):
        start = int(cu_seqlens[index])
        end = int(cu_seqlens[index + 1])
        q = query[:, :, start:end]
        k = key[:, :, start:end]
        v = value[:, :, start:end]
        weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        outputs.append(torch.matmul(weights, v).transpose(1, 2).contiguous())
    return torch.cat(outputs, dim=1)


class DynamicWindowAttention(torch.nn.Module):
    """Trace projections normally; script only the data-dependent window loop."""

    def __init__(self, attention) -> None:
        super().__init__()
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.out_proj = attention.out_proj
        self.num_heads = attention.num_heads
        self.scaling = attention.scaling

    def forward(self, hidden_states, cu_seqlens, max_seqlen=None):
        length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).reshape(length, self.num_heads, -1)
        key = self.k_proj(hidden_states).reshape(length, self.num_heads, -1)
        value = self.v_proj(hidden_states).reshape(length, self.num_heads, -1)
        # ONNX Slice starts/ends/axes/steps share Tind. Legacy-exported axes
        # are int64; normalize internal boundaries without changing int32 I/O.
        boundaries = cu_seqlens.to(torch.int64)
        output = _window_attention(
            query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0), boundaries, self.scaling,
        )
        return self.out_proj(output.reshape(length, -1).contiguous())


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
            dtype = self.tower.get_input_embeddings().weight.dtype
            encoded = self.tower(
                input_features.to(dtype), input_features_mask, cu_seqlens=cu_seqlens
            )
            return self.projector(encoded.last_hidden_state).float()

    module = Front().eval()
    path = output_dir / "audio_front_model.onnx"
    original_attention = [layer.self_attn for layer in tower.layers]
    try:
        for layer, attention in zip(tower.layers, original_attention):
            layer.self_attn = DynamicWindowAttention(attention).eval()
        torch.onnx.export(
            module, (features, mask, schedule.to(torch.int32)), str(path),
            input_names=["input_features", "input_features_mask", "cu_seqlens"],
            output_names=["audio_embeds"],
            dynamic_axes={
                "input_features": {2: "frames"},
                "input_features_mask": {1: "frames"},
                "cu_seqlens": {0: "boundaries"},
                "audio_embeds": {0: "audio_frames"},
            },
            opset_version=OPSET, dynamo=False, external_data=True,
        )
    finally:
        for layer, attention in zip(tower.layers, original_attention):
            layer.self_attn = attention
    return consolidate(path)


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
    return consolidate(path)


class Float32DecoderStep(torch.nn.Module):
    """Preserve floating runner I/O while executing the original fp16 decoder."""

    def __init__(self, model) -> None:
        super().__init__()
        self.dtype = model.model.language_model.embed_tokens.weight.dtype
        self.step = DecoderStep(model.model.language_model, model.lm_head, model.config.text_config)

    def forward(self, inputs_embeds, past_key, past_value):
        outputs = self.step(
            inputs_embeds.to(self.dtype), past_key.to(self.dtype), past_value.to(self.dtype)
        )
        return tuple(value.float() for value in outputs)


def export_decoder(model, output_dir: Path) -> dict:
    config = model.config.text_config
    module = Float32DecoderStep(model).eval()
    embeds = torch.zeros(1, 1, config.hidden_size, dtype=torch.float32)
    past = torch.zeros(
        config.num_hidden_layers, 1, config.num_key_value_heads, 4, config.head_dim,
        dtype=torch.float32,
    )
    path = output_dir / "decoder_step_model.onnx"
    torch.onnx.export(
        module, (embeds, past, past), str(path),
        input_names=["inputs_embeds", "past_key", "past_value"],
        output_names=["logits", "present_key", "present_value"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "tokens"},
            "past_key": {1: "batch", 3: "past_length"},
            "past_value": {1: "batch", 3: "past_length"},
            "logits": {0: "batch", 1: "tokens"},
            "present_key": {1: "batch", 3: "present_length"},
            "present_value": {1: "batch", 3: "present_length"},
        },
        opset_version=DECODER_OPSET, dynamo=False, external_data=True,
    )
    return consolidate(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--audio", type=Path, default=Path("tools/data/fixtures/audio/jfk-short.wav"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    model, processor = load(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = request_for(processor, args.audio)
    features = inputs["input_features"].float()
    mask = inputs["input_features_mask"]
    mask = mask.float()
    tower_dtype = model.model.audio_tower.get_input_embeddings().weight.dtype
    schedule = capture_schedule(model, features.to(tower_dtype), mask)

    receipt = {
        "dtype": "float16",
        "builtFrom": str(args.model_dir),
        "reason": (
            "Original-weight fp16 builder; fp32 floating I/O. Converted candidate /Concat_2 "
            "failed in Node ORT 1.25-dev and Python ORT 1.30 at past lengths 0 and 1. "
            "Node basic/extended loaded it; default all failed during fusion. "
            "Browser memory and provider acceptance remain unmeasured."
        ),
        "graphs": {
            "front": export_front(model, features, mask, schedule, args.output_dir),
            "embed": export_embed(model, args.output_dir),
            "decoder": export_decoder(model, args.output_dir),
        },
        "support": save_support(model, inputs, schedule, args.model_dir, args.output_dir),
        "source_sha256": _digest(Path(__file__)),
        "floating_io": "float32",
        "audio_sha256": _digest(args.audio),
    }
    (args.output_dir / "fp16-export-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
