"""Export official Qwen3-ASR-0.6B graphs from the qwen-asr Transformers stack.

Does not use third-party ONNX. Records the exact export blocker when the
official forward cannot be serialized (dynamic chunking, DynamicCache, etc.).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(r"N:\models\Qwen3-ASR-0.6B"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"N:\models\onnx\qwen3-asr-0.6b-official"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mel-frames", type=int, default=800)
    parser.add_argument("--remainder-frames", type=int, default=1050, help="Non-multiple-of-100 T for pad/crop vs official.")
    return parser.parse_args()


def _disable_tls_verify() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_token_count(frames: int) -> int:
    leave = frames % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (frames // 100) * 13


class AudioEncoderWrapper(nn.Module):
    def __init__(self, tower: nn.Module) -> None:
        super().__init__()
        self.tower = tower

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        output = self.tower(input_features, feature_lens=feature_lens)
        return output.last_hidden_state


class StaticWindowAudioEncoder(nn.Module):
    """Official encoder weights, T % 100 == 0 via reshape(-1, 100, 128).

    Dynamic axes can vary T as long as the caller pads to a 100-frame multiple.
    """

    def __init__(self, tower: nn.Module) -> None:
        super().__init__()
        self.tower = tower
        self.chunk_wave = int(tower.n_window * 2)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        tower = self.tower
        n_mels = input_features.shape[0]
        x = input_features.transpose(0, 1).reshape(-1, self.chunk_wave, n_mels)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = torch.nn.functional.gelu(tower.conv2d1(x))
        x = torch.nn.functional.gelu(tower.conv2d2(x))
        x = torch.nn.functional.gelu(tower.conv2d3(x))
        batch, channels, freq, time = x.size()
        x = tower.conv_out(x.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * freq))
        pos = tower.positional_embedding.positional_embedding[: x.shape[1], :].unsqueeze(0).to(dtype=x.dtype)
        x = x + pos
        hidden = x.reshape(-1, x.shape[-1])
        seq = hidden.shape[0]
        attention_mask = torch.zeros(1, 1, seq, seq, dtype=hidden.dtype, device=hidden.device)
        cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device=hidden.device)
        for layer in tower.layers:
            hidden = layer(hidden, cu_seqlens=cu_seqlens, attention_mask=attention_mask)[0]
        hidden = tower.ln_post(hidden)
        hidden = tower.proj1(hidden)
        hidden = tower.act(hidden)
        return tower.proj2(hidden)


class PadToChunkAudioEncoder(nn.Module):
    """Zero-pad T to the next 100-frame multiple, then run the static window path."""

    def __init__(self, tower: nn.Module) -> None:
        super().__init__()
        self.inner = StaticWindowAudioEncoder(tower)
        self.chunk_wave = int(tower.n_window * 2)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        frames = input_features.shape[-1]
        pad = (self.chunk_wave - (frames % self.chunk_wave)) % self.chunk_wave
        padded = F.pad(input_features, (0, pad)) if pad else input_features
        return self.inner(padded)


def try_export(name: str, fn) -> dict:
    try:
        details = fn()
        return {"name": name, "ok": True, **(details or {})}
    except Exception as error:  # noqa: BLE001
        return {
            "name": name,
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "traceback": traceback.format_exc()[-6000:],
        }


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from qwen_asr import Qwen3ASRModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3ASRModel.from_pretrained(
        str(args.model_dir.resolve()),
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=8,
    )
    inner = model.model
    thinker = inner.thinker
    tower = thinker.audio_tower
    tower.eval()
    thinker.eval()

    report: dict = {
        "schema_version": 1,
        "family": "qwen3-asr-0.6b",
        "source": "official qwen-asr Transformers Qwen3ASRForConditionalGeneration",
        "mel_frames": args.mel_frames,
        "attempts": [],
    }

    wrapper = AudioEncoderWrapper(tower).eval()
    dummy_features = torch.randn(128, args.mel_frames, dtype=torch.float32)
    dummy_lens = torch.tensor([args.mel_frames], dtype=torch.long)
    encoder_path = args.output_dir / "audio-encoder.onnx"

    def export_encoder_onnx() -> dict:
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (dummy_features, dummy_lens),
                strict=False,
                check_trace=False,
            )
        torch.onnx.export(
            traced,
            (dummy_features, dummy_lens),
            str(encoder_path),
            input_names=["input_features", "feature_lens"],
            output_names=["audio_embeddings"],
            opset_version=17,
            dynamo=False,
        )
        return {"onnx_path": str(encoder_path), "size_bytes": encoder_path.stat().st_size}

    def export_encoder_dynamo() -> dict:
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_lens),
            str(args.output_dir / "audio-encoder-dynamo.onnx"),
            input_names=["input_features", "feature_lens"],
            output_names=["audio_embeddings"],
            opset_version=18,
            dynamo=True,
        )
        path = args.output_dir / "audio-encoder-dynamo.onnx"
        return {"onnx_path": str(path), "size_bytes": path.stat().st_size if path.is_file() else None}

    def export_encoder_script() -> dict:
        scripted = torch.jit.script(wrapper)
        torch.onnx.export(
            scripted,
            (dummy_features, dummy_lens),
            str(args.output_dir / "audio-encoder-script.onnx"),
            input_names=["input_features", "feature_lens"],
            output_names=["audio_embeddings"],
            opset_version=17,
            dynamo=False,
        )
        path = args.output_dir / "audio-encoder-script.onnx"
        return {"onnx_path": str(path), "size_bytes": path.stat().st_size if path.is_file() else None}

    report["attempts"].append(try_export("encoder_jit_trace_onnx", export_encoder_onnx))
    report["attempts"].append(try_export("encoder_dynamo_onnx", export_encoder_dynamo))
    report["attempts"].append(try_export("encoder_jit_script_onnx", export_encoder_script))

    def probe_forward() -> dict:
        with torch.no_grad():
            out = wrapper(dummy_features, dummy_lens)
        return {"output_shape": list(out.shape), "output_dtype": str(out.dtype)}

    report["attempts"].append(try_export("encoder_eager_forward", probe_forward))

    static = StaticWindowAudioEncoder(tower).eval()
    padded = PadToChunkAudioEncoder(tower).eval()
    static_path = args.output_dir / f"audio-encoder-static-t{args.mel_frames}.onnx"
    dynamic_path = args.output_dir / "audio-encoder-dynamic.onnx"

    def compare_static() -> dict:
        with torch.no_grad():
            official = wrapper(dummy_features, dummy_lens)
            rewritten = static(dummy_features)
        delta = (official - rewritten).abs()
        return {
            "official_shape": list(official.shape),
            "static_shape": list(rewritten.shape),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
        }

    def compare_remainder() -> dict:
        remainder_t = int(args.remainder_frames)
        remainder = torch.randn(128, remainder_t, dtype=torch.float32)
        remainder_lens = torch.tensor([remainder_t], dtype=torch.long)
        pad_t = ((remainder_t + 99) // 100) * 100
        tokens = official_token_count(remainder_t)
        with torch.no_grad():
            official = wrapper(remainder, remainder_lens)
            padded_out = padded(remainder)
            cropped = padded_out[:tokens]
            aligned = static(F.pad(remainder, (0, pad_t - remainder_t)))
        delta_pad = (official - cropped).abs()
        delta_static = (padded_out - aligned).abs()
        return {
            "remainder_frames": remainder_t,
            "padded_frames": pad_t,
            "official_tokens": list(official.shape),
            "padded_tokens": list(padded_out.shape),
            "cropped_tokens": list(cropped.shape),
            "formula_tokens": tokens,
            "pad_crop_vs_official_max_abs": float(delta_pad.max()) if official.shape == cropped.shape else None,
            "pad_crop_vs_official_mean_abs": float(delta_pad.mean()) if official.shape == cropped.shape else None,
            "pad_module_vs_static_max_abs": float(delta_static.max()),
            "shape_match": list(official.shape) == list(cropped.shape),
        }

    def export_static() -> dict:
        with torch.no_grad():
            torch.onnx.export(
                static,
                (dummy_features,),
                str(static_path),
                input_names=["input_features"],
                output_names=["audio_embeddings"],
                opset_version=17,
                dynamo=False,
            )
        return {
            "onnx_path": str(static_path),
            "size_bytes": static_path.stat().st_size,
            "sha256": sha256_file(static_path),
        }

    def export_dynamic() -> dict:
        dummy_dynamic = torch.randn(128, 800, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                static,
                (dummy_dynamic,),
                str(dynamic_path),
                input_names=["input_features"],
                output_names=["audio_embeddings"],
                opset_version=17,
                dynamo=False,
                dynamic_axes={
                    "input_features": {1: "mel_frames"},
                    "audio_embeddings": {0: "audio_tokens"},
                },
            )
        return {
            "onnx_path": str(dynamic_path),
            "size_bytes": dynamic_path.stat().st_size,
            "sha256": sha256_file(dynamic_path),
        }

    def ort_static() -> dict:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(static_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pytorch = static(dummy_features).numpy()
        onnx_out = session.run(None, {"input_features": dummy_features.numpy()})[0]
        delta = np.abs(pytorch - onnx_out)
        return {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "shape": list(onnx_out.shape),
        }

    def ort_dynamic_lengths() -> dict:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(dynamic_path), providers=["CPUExecutionProvider"])
        results = {}
        for frames in (800, 1100, ((int(args.remainder_frames) + 99) // 100) * 100):
            feats = torch.randn(128, frames, dtype=torch.float32)
            with torch.no_grad():
                pytorch = static(feats).numpy()
            onnx_out = session.run(None, {"input_features": feats.numpy()})[0]
            delta = np.abs(pytorch - onnx_out)
            results[str(frames)] = {
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "shape": list(onnx_out.shape),
            }
        return results

    report["attempts"].append(try_export("encoder_static_vs_official", compare_static))
    report["attempts"].append(try_export("encoder_pad_crop_vs_official_remainder", compare_remainder))
    report["attempts"].append(try_export("encoder_static_onnx", export_static))
    report["attempts"].append(try_export("encoder_dynamic_onnx", export_dynamic))
    if static_path.is_file():
        report["attempts"].append(try_export("encoder_static_onnxruntime_cpu", ort_static))
    if dynamic_path.is_file():
        report["attempts"].append(try_export("encoder_dynamic_onnxruntime_cpu", ort_dynamic_lengths))

    encoder_ok = any(
        item.get("ok") and item.get("name") in {"encoder_static_onnx", "encoder_dynamic_onnx", "encoder_jit_trace_onnx"}
        for item in report["attempts"]
    )
    if encoder_ok:
        report["failureClass"] = None
        report["status"] = "exported-encoder"
    else:
        report["failureClass"] = "EXPORT_BLOCKED"
        report["missing"] = (
            "Official Qwen3ASRAudioEncoder.forward is not ONNX-serializable: "
            "data-dependent chunk_lengths.tolist() split, pad_sequence of ragged CNN chunks, "
            "boolean-mask ragged gather, and Python-built cu_seqlens for encoder attention. "
            "Decoder uses transformers DynamicCache + create_causal_mask. "
            "These are in the official qwen-asr 0.0.6 forward, not missing weights."
        )
        report["status"] = "experimental-blocked"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failureClass": report.get("failureClass"), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    _disable_tls_verify()
    main()
