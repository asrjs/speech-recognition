"""Export independently testable FireRedASR2-AED ONNX graph boundaries.

The exporter consumes an existing upstream checkpoint and emits:

* encoder.onnx: fbank/CMVN features plus lengths to encoder states/mask;
* decoder.onnx: full teacher-forced AED logits for a token prefix;
* ctc.onnx: CTC log probabilities used by timestamp refinement.

The decoder graph is intentionally a full-prefix graph. It is a correctness
boundary for parity work, not a claim that the eventual browser decoder should
recompute every prefix instead of using a step/cache graph.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REQUIRED_MODEL_FILES = (
    "cmvn.ark",
    "model.pth.tar",
    "dict.txt",
    "train_bpe1000.model",
)


class FullDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_mask = self.decoder.ignored_target_position_is_0(
            input_ids,
            self.decoder.pad_id,
        )
        decoder_output = self.decoder.dropout(
            self.decoder.tgt_word_emb(input_ids) * self.decoder.scale
            + self.decoder.positional_encoding(input_ids)
        )
        for layer in self.decoder.layer_stack:
            decoder_output = layer(
                decoder_output,
                encoder_outputs,
                target_mask,
                src_mask,
                cache=None,
            )
        decoder_output = self.decoder.layer_norm_out(decoder_output)
        return self.decoder.tgt_word_prj(decoder_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FireRedASR2-AED encoder, decoder, and CTC ONNX graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fire-red-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("encoder", "decoder", "ctc"),
        default=("encoder", "decoder", "ctc"),
    )
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--dummy-frames", type=int, default=400)
    parser.add_argument("--dummy-target-length", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def export_graph(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset_version: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        inputs,
        str(path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def main() -> None:
    args = parse_args()
    fire_red_root = args.fire_red_root.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not (fire_red_root / "fireredasr2s").is_dir():
        raise FileNotFoundError(f"FireRed source package not found: {fire_red_root}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FireRed model directory not found: {model_dir}")
    if args.dummy_frames < 16 or args.dummy_target_length < 1:
        raise ValueError("dummy frames must be at least 16 and target length positive")
    for name in REQUIRED_MODEL_FILES:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Required FireRed checkpoint file is missing: {model_dir / name}")

    sys.path.insert(0, str(fire_red_root))
    from fireredasr2s.fireredasr2.asr import load_fireredasr_aed_model

    model = load_fireredasr_aed_model(str(model_dir / "model.pth.tar"))
    model.eval()
    if args.dtype == "float16":
        model.half()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    dummy_features = torch.zeros((1, args.dummy_frames, 80), dtype=dtype)
    dummy_lengths = torch.tensor([args.dummy_frames], dtype=torch.long)
    with torch.inference_mode():
        dummy_encoder_outputs, _, dummy_src_mask = model.encoder(
            dummy_features,
            dummy_lengths,
        )
    dummy_input_ids = torch.full(
        (1, args.dummy_target_length),
        model.decoder.pad_id,
        dtype=torch.long,
    )
    dummy_input_ids[0, 0] = model.decoder.sos_id
    decoder_wrapper = FullDecoderWrapper(model.decoder).eval()

    components: dict[str, dict[str, Any]] = {}
    selected = set(args.components)
    if "encoder" in selected:
        encoder_path = output_dir / "encoder.onnx"
        export_graph(
            model.encoder,
            (dummy_features, dummy_lengths),
            encoder_path,
            ["padded_input", "input_lengths"],
            ["enc_output", "output_lengths", "src_mask"],
            {
                "padded_input": {0: "batch_size", 1: "feature_frames"},
                "input_lengths": {0: "batch_size"},
                "enc_output": {0: "batch_size", 1: "encoder_frames"},
                "output_lengths": {0: "batch_size"},
                "src_mask": {0: "batch_size", 2: "encoder_frames"},
            },
            args.opset_version,
        )
        components["encoder"] = {
            "path": encoder_path.name,
            "inputs": ["padded_input", "input_lengths"],
            "outputs": ["enc_output", "output_lengths", "src_mask"],
        }

    if "decoder" in selected:
        decoder_path = output_dir / "decoder.onnx"
        export_graph(
            decoder_wrapper,
            (dummy_input_ids, dummy_encoder_outputs, dummy_src_mask),
            decoder_path,
            ["input_ids", "encoder_outputs", "src_mask"],
            ["logits"],
            {
                "input_ids": {0: "batch_size", 1: "target_frames"},
                "encoder_outputs": {0: "batch_size", 1: "encoder_frames"},
                "src_mask": {0: "batch_size", 2: "encoder_frames"},
                "logits": {0: "batch_size", 1: "target_frames"},
            },
            args.opset_version,
        )
        components["decoder"] = {
            "path": decoder_path.name,
            "inputs": ["input_ids", "encoder_outputs", "src_mask"],
            "outputs": ["logits"],
        }

    if "ctc" in selected:
        ctc_path = output_dir / "ctc.onnx"
        export_graph(
            model.ctc,
            (dummy_encoder_outputs,),
            ctc_path,
            ["encoder_outputs"],
            ["ctc_log_probs"],
            {
                "encoder_outputs": {0: "batch_size", 1: "encoder_frames"},
                "ctc_log_probs": {0: "batch_size", 1: "encoder_frames"},
            },
            args.opset_version,
        )
        components["ctc"] = {
            "path": ctc_path.name,
            "inputs": ["encoder_outputs"],
            "outputs": ["ctc_log_probs"],
        }

    manifest = {
        "schema_version": 1,
        "artifact_kind": "fireredasr2-aed-onnx",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_only": True,
        "source": {
            "fire_red_root": str(fire_red_root),
            "fire_red_git_revision": git_revision(fire_red_root),
            "model_dir": str(model_dir),
        },
        "export": {
            "dtype": args.dtype,
            "opset_version": args.opset_version,
            "dummy_frames": args.dummy_frames,
            "dummy_target_length": args.dummy_target_length,
            "components": components,
            "boundary_note": (
                "decoder.onnx is a full teacher-forced prefix graph; it is a "
                "parity boundary, not a cached decoder-step runtime contract"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for component in components.values():
        path = output_dir / component["path"]
        component["size_bytes"] = path.stat().st_size
        component["sha256"] = sha256_file(path)
    (output_dir / "fireredasr2-aed-onnx-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote FireRed ONNX graphs and manifest to {output_dir}")


if __name__ == "__main__":
    main()
