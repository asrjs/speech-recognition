from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from parakeet_realtime_common import (
    DEFAULT_MODEL_ID,
    build_runtime_config,
    export_vocab_lines,
    load_model,
)


class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.preprocessor = model.preprocessor

    def forward(self, waveforms: torch.Tensor, waveforms_lens: torch.Tensor):
        return self.preprocessor(input_signal=waveforms, length=waveforms_lens)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        encoded, _ = self.encoder(audio_signal=audio_signal, length=length)
        return encoded


class DecoderJointWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.joint = model.joint
        self.joint.set_fuse_loss_wer(False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ):
        decoder_outputs, _, next_state = self.decoder(
            targets=targets,
            target_length=target_length,
            states=(input_states_1, input_states_2),
        )
        logits = self.joint(
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
        )
        return logits, next_state[0], next_state[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Parakeet Realtime EOU 120M v1 to encoder/decoder_joint ONNX artifacts.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-preprocessor", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    return parser.parse_args()


def export_preprocessor(
    wrapper: torch.nn.Module,
    output_path: Path,
    opset: int,
    sample_rate: int,
) -> None:
    dummy_waveforms = torch.randn(1, sample_rate, dtype=torch.float32)
    dummy_lengths = torch.tensor([sample_rate], dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_waveforms, dummy_lengths),
        str(output_path),
        input_names=["waveforms", "waveforms_lens"],
        output_names=["features", "features_lens"],
        dynamic_axes={
            "waveforms": {0: "batch", 1: "samples"},
            "waveforms_lens": {0: "batch"},
            "features": {0: "batch", 2: "frames"},
            "features_lens": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def export_encoder(wrapper: torch.nn.Module, output_path: Path, opset: int, mel_bins: int) -> None:
    dummy_signal = torch.randn(1, mel_bins, 320, dtype=torch.float32)
    dummy_lengths = torch.tensor([320], dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy_signal, dummy_lengths),
        str(output_path),
        input_names=["audio_signal", "length"],
        output_names=["outputs"],
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "frames"},
            "length": {0: "batch"},
            "outputs": {0: "batch", 2: "encoded_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def export_decoder_joint(
    wrapper: torch.nn.Module,
    output_path: Path,
    opset: int,
    encoder_hidden: int,
    pred_hidden: int,
    pred_layers: int,
    blank_id: int,
) -> None:
    dummy_encoder = torch.randn(1, encoder_hidden, 1, dtype=torch.float32)
    dummy_targets = torch.tensor([[blank_id]], dtype=torch.int32)
    dummy_target_length = torch.tensor([1], dtype=torch.int32)
    dummy_state_1 = torch.zeros(pred_layers, 1, pred_hidden, dtype=torch.float32)
    dummy_state_2 = torch.zeros(pred_layers, 1, pred_hidden, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (
            dummy_encoder,
            dummy_targets,
            dummy_target_length,
            dummy_state_1,
            dummy_state_2,
        ),
        str(output_path),
        input_names=[
            "encoder_outputs",
            "targets",
            "target_length",
            "input_states_1",
            "input_states_2",
        ],
        output_names=["outputs", "output_states_1", "output_states_2"],
        dynamic_axes={
            "encoder_outputs": {0: "batch", 2: "encoded_frames"},
            "targets": {0: "batch", 1: "target_frames"},
            "target_length": {0: "batch"},
            "input_states_1": {1: "batch"},
            "input_states_2": {1: "batch"},
            "outputs": {0: "batch", 2: "target_plus_sos"},
            "output_states_1": {1: "batch"},
            "output_states_2": {1: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_id, args.device, args.model_path)
    runtime_config = build_runtime_config(args.model_id, model)

    preprocessor_path = args.output_dir / "nemo128.onnx"
    encoder_path = args.output_dir / "encoder-model.onnx"
    decoder_joint_path = args.output_dir / "decoder_joint-model.onnx"
    vocab_path = args.output_dir / "vocab.txt"
    config_path = args.output_dir / "config.json"

    if not args.skip_preprocessor:
        try:
            export_preprocessor(
                PreprocessorWrapper(model).eval(),
                preprocessor_path,
                args.opset,
                int(runtime_config["sample_rate"]),
            )
            print(f"Exported preprocessor to {preprocessor_path}")
        except Exception as error:  # noqa: BLE001
            print(
                "WARNING: Preprocessor export failed. "
                "The current PyTorch ONNX exporter still rejects the NeMo STFT path in this env. "
                "Use the shared JS frontend with centered/raw-log NeMo settings for runtime parity."
            )
            print(f"Preprocessor export error: {error}")

    export_encoder(
        EncoderWrapper(model).eval(),
        encoder_path,
        args.opset,
        int(runtime_config["mel_bins"]),
    )
    print(f"Exported encoder to {encoder_path}")

    export_decoder_joint(
        DecoderJointWrapper(model).eval(),
        decoder_joint_path,
        args.opset,
        encoder_hidden=int(model.cfg.encoder.d_model),
        pred_hidden=int(runtime_config["pred_hidden"]),
        pred_layers=int(runtime_config["pred_layers"]),
        blank_id=int(runtime_config["blank_id"]),
    )
    print(f"Exported decoder_joint to {decoder_joint_path}")

    if not args.skip_tokenizer:
        vocab_path.write_text("\n".join(export_vocab_lines(model.tokenizer)) + "\n", encoding="utf-8")
        print(f"Wrote tokenizer vocabulary to {vocab_path}")

    if not args.skip_config:
        config_path.write_text(
            json.dumps(runtime_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote runtime config to {config_path}")


if __name__ == "__main__":
    main()
