from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from canary_common import DEFAULT_MODEL_ID, exported_tokenizer_payload, load_model, runtime_config_payload


class CanaryPreprocessorWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.preprocessor = model.preprocessor

    def forward(self, waveforms: torch.Tensor, waveforms_lens: torch.Tensor):
        return self.preprocessor(input_signal=waveforms, length=waveforms_lens)


class CanaryEncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.encoder_decoder_proj = model.encoder_decoder_proj
        self.use_transf_encoder = bool(getattr(model, "use_transf_encoder", False))
        self.transf_encoder = getattr(model, "transf_encoder", None)

    def forward(self, processed_signal: torch.Tensor, processed_signal_length: torch.Tensor):
        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoder_states = encoded.permute(0, 2, 1)
        encoder_states = self.encoder_decoder_proj(encoder_states)
        encoder_mask = (
            torch.arange(encoder_states.shape[1], device=encoder_states.device).unsqueeze(0)
            < encoded_len.unsqueeze(1)
        ).to(encoder_states.dtype)
        if self.use_transf_encoder and self.transf_encoder is not None:
            encoder_states = self.transf_encoder(
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
            )
        return encoder_states, encoded_len, encoder_mask


class CanaryDecoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.transf_decoder = model.transf_decoder
        self.log_softmax = model.log_softmax

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor,
        encoder_mask: torch.Tensor,
    ):
        decoder_mask = torch.ones_like(input_ids)
        decoder_states = self.transf_decoder(
            input_ids=input_ids,
            decoder_mask=decoder_mask,
            encoder_embeddings=encoder_states,
            encoder_mask=encoder_mask,
        )
        logits = self.log_softmax(hidden_states=decoder_states)
        return logits[:, -1, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Canary 180M Flash into ONNX wrappers usable by @asrjs/speech-recognition.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
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
        input_names=["processed_signal", "processed_signal_length"],
        output_names=["encoder_states", "encoded_length", "encoder_mask"],
        dynamic_axes={
            "processed_signal": {0: "batch", 2: "frames"},
            "processed_signal_length": {0: "batch"},
            "encoder_states": {0: "batch", 1: "encoded_frames"},
            "encoded_length": {0: "batch"},
            "encoder_mask": {0: "batch", 1: "encoded_frames"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def export_decoder(
    wrapper: torch.nn.Module,
    output_path: Path,
    opset: int,
    decoder_hidden_size: int,
) -> None:
    dummy_input_ids = torch.tensor([[7, 4, 16, 62, 62, 5, 9, 11, 13]], dtype=torch.long)
    dummy_encoder_states = torch.randn(1, 40, decoder_hidden_size, dtype=torch.float32)
    dummy_encoder_mask = torch.ones(1, 40, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_encoder_states, dummy_encoder_mask),
        str(output_path),
        input_names=["input_ids", "encoder_states", "encoder_mask"],
        output_names=["next_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "target_length"},
            "encoder_states": {0: "batch", 1: "encoded_frames"},
            "encoder_mask": {0: "batch", 1: "encoded_frames"},
            "next_logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_id, args.device)
    model.eval()

    preprocessor_path = args.output_dir / "nemo128.onnx"
    encoder_path = args.output_dir / "encoder-model.onnx"
    decoder_path = args.output_dir / "decoder-model.onnx"
    tokenizer_path = args.output_dir / "tokenizer.json"
    config_path = args.output_dir / "config.json"

    if not args.skip_preprocessor:
        try:
            export_preprocessor(
                CanaryPreprocessorWrapper(model).eval(),
                preprocessor_path,
                args.opset,
                int(model.cfg.preprocessor.sample_rate),
            )
            print(f"Exported preprocessor to {preprocessor_path}")
        except Exception as error:  # noqa: BLE001
            print(
                "WARNING: Preprocessor export failed. "
                "Current PyTorch ONNX export does not reliably support the NeMo STFT path in this env. "
                "Reuse a shared 128-bin NeMo preprocessor artifact such as nemo128.onnx if needed."
            )
            print(f"Preprocessor export error: {error}")

    export_encoder(
        CanaryEncoderWrapper(model).eval(),
        encoder_path,
        args.opset,
        int(model.cfg.preprocessor.features),
    )
    print(f"Exported encoder to {encoder_path}")

    export_decoder(
        CanaryDecoderWrapper(model).eval(),
        decoder_path,
        args.opset,
        int(model.cfg.model_defaults.lm_dec_hidden),
    )
    print(f"Exported decoder to {decoder_path}")

    if not args.skip_tokenizer:
        tokenizer_path.write_text(
            json.dumps(exported_tokenizer_payload(model), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote tokenizer metadata to {tokenizer_path}")

    if not args.skip_config:
        config_path.write_text(
            json.dumps(runtime_config_payload(args.model_id, model), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote runtime config to {config_path}")


if __name__ == "__main__":
    main()
