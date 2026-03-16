from __future__ import annotations

import argparse
from pathlib import Path

import torch

from parakeet_realtime_common import (
    DEFAULT_AUDIO,
    DEFAULT_MODEL_ID,
    build_runtime_config,
    dump_json,
    load_audio,
    load_model,
    manual_greedy_decode,
    numpy_payload,
    run_model_transcribe,
    tensor_payload,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference tensors and greedy decode outputs for Parakeet Realtime EOU 120M v1.",
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = resolve_device(args.device)
    model = load_model(args.model_id, device, args.model_path)
    runtime_config = build_runtime_config(args.model_id, model)

    waveform, sample_rate = load_audio(audio_path, int(runtime_config["sample_rate"]))
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
    waveform_length = torch.tensor([waveform.shape[0]], dtype=torch.long, device=device)

    with torch.inference_mode():
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=waveform_tensor,
            length=waveform_length,
        )
        encoder_outputs, encoded_length = model.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
        )

    manual = manual_greedy_decode(model, encoder_outputs, runtime_config)
    nemo_transcribe = run_model_transcribe(model, audio_path)

    payload = {
        "meta": {
            "audio_path": str(audio_path),
            "model_id": args.model_id,
            "device": device,
            "input_sample_rate": sample_rate,
            "input_samples": int(waveform.shape[0]),
            "duration_seconds": round(float(waveform.shape[0]) / float(sample_rate), 6),
        },
        "runtime_config": runtime_config,
        "audio": {
            "waveform": numpy_payload(waveform, [1, int(waveform.shape[0])]),
        },
        "preprocessor": {
            "features": tensor_payload(processed_signal),
            "feature_lengths": processed_signal_length.detach().cpu().tolist(),
        },
        "encoder": {
            "outputs": tensor_payload(encoder_outputs),
            "lengths": encoded_length.detach().cpu().tolist(),
        },
        "decode": {
            "manual_greedy": manual,
            "nemo_transcribe": nemo_transcribe,
            "visible_text_matches_nemo_transcribe": manual["text"] == nemo_transcribe["text"].replace("<EOU>", "").replace("<EOB>", ""),
            "raw_text_matches_nemo_transcribe": manual["raw_text"] == nemo_transcribe["text"],
        },
    }

    dump_json(args.output.resolve(), payload)
    print(f"Wrote Parakeet realtime reference JSON to {args.output}")


if __name__ == "__main__":
    main()
