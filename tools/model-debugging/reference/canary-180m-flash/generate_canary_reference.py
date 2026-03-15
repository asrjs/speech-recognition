from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecodingConfig

from canary_common import (
    DEFAULT_AUDIO,
    DEFAULT_MODEL_ID,
    PromptSettings,
    build_prompt_slots,
    decode_text,
    encode_prompt,
    ids_to_pieces,
    load_audio,
    load_model,
    manual_decode,
    numpy_payload,
    resolve_device,
    runtime_config_payload,
    tensor_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PyTorch reference tensors/tokens/text for Canary 180M Flash parity work.",
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--decoder-context", default="")
    parser.add_argument("--emotion", default="<|emo:undefined|>")
    parser.add_argument("--disable-pnc", action="store_true")
    parser.add_argument("--enable-itn", action="store_true")
    parser.add_argument("--enable-timestamp", action="store_true")
    parser.add_argument("--enable-diarize", action="store_true")
    parser.add_argument("--skip-nemo-greedy-check", action="store_true")
    return parser.parse_args()


def build_prompt_settings(args: argparse.Namespace) -> PromptSettings:
    return PromptSettings(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        decoder_context=args.decoder_context,
        emotion=args.emotion,
        pnc=not args.disable_pnc,
        itn=args.enable_itn,
        timestamp=args.enable_timestamp,
        diarize=args.enable_diarize,
    )


def run_nemo_greedy_decode(
    model: Any,
    encoder_states: torch.Tensor,
    encoder_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
) -> dict[str, Any]:
    original_cfg = model.cfg.decoding
    greedy_cfg = MultiTaskDecodingConfig(strategy="greedy")
    model.change_decoding_strategy(greedy_cfg)
    try:
        hypotheses = model.decoding.decode_predictions_tensor(
            encoder_hidden_states=encoder_states,
            encoder_input_mask=encoder_mask,
            decoder_input_ids=prompt_ids.unsqueeze(0),
            return_hypotheses=True,
        )
        hypothesis = hypotheses[0]
        token_ids = [int(item) for item in hypothesis.y_sequence.tolist()]
        return {
            "text": str(hypothesis.text),
            "token_ids": token_ids,
            "token_pieces": ids_to_pieces(model.tokenizer, token_ids),
        }
    finally:
        model.change_decoding_strategy(original_cfg)


def main() -> None:
    args = parse_args()
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = load_model(args.model_id, device)
    prompt_settings = build_prompt_settings(args)

    waveform, sample_rate = load_audio(audio_path, int(model.cfg.preprocessor.sample_rate))
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
    waveform_length = torch.tensor([waveform.shape[0]], dtype=torch.long, device=device)

    with torch.inference_mode():
        processed_signal, processed_signal_length = model.preprocessor(
            input_signal=waveform_tensor,
            length=waveform_length,
        )
        _, encoded_length, encoder_states, encoder_mask = model(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
        )

    prompt_ids = encode_prompt(model, prompt_settings).to(device)
    manual = manual_decode(model, encoder_states, encoder_mask, prompt_ids)

    nemo_greedy = (
        None
        if args.skip_nemo_greedy_check
        else run_nemo_greedy_decode(model, encoder_states, encoder_mask, prompt_ids)
    )

    payload = {
        "meta": {
            "audio_path": str(audio_path),
            "model_id": args.model_id,
            "device": device,
            "input_sample_rate": sample_rate,
            "input_samples": int(waveform.shape[0]),
            "duration_seconds": round(float(waveform.shape[0]) / float(sample_rate), 6),
        },
        "runtime_config": runtime_config_payload(args.model_id, model),
        "prompt": {
            "settings": {
                "source_lang": prompt_settings.source_lang,
                "target_lang": prompt_settings.target_lang,
                "decoder_context": prompt_settings.decoder_context,
                "emotion": prompt_settings.emotion,
                "pnc": prompt_settings.pnc,
                "itn": prompt_settings.itn,
                "timestamp": prompt_settings.timestamp,
                "diarize": prompt_settings.diarize,
            },
            "slots": build_prompt_slots(model, prompt_settings),
            "ids": prompt_ids.detach().cpu().tolist(),
            "pieces": ids_to_pieces(model.tokenizer, prompt_ids.detach().cpu().tolist()),
        },
        "audio": {
            "waveform": numpy_payload(waveform, [1, int(waveform.shape[0])]),
        },
        "preprocessor": {
            "features": tensor_payload(processed_signal),
            "feature_lengths": processed_signal_length.detach().cpu().tolist(),
        },
        "encoder": {
            "states": tensor_payload(encoder_states),
            "mask": tensor_payload(encoder_mask),
            "lengths": encoded_length.detach().cpu().tolist(),
        },
        "decode": {
            "manual_greedy": manual,
            "nemo_greedy": nemo_greedy,
            "texts_match": None if nemo_greedy is None else manual["text"] == nemo_greedy["text"],
            "token_ids_match": None
            if nemo_greedy is None
            else manual["token_ids"] == nemo_greedy["token_ids"],
        },
        "transcript": {
            "text": manual["text"],
            "normalized_text": decode_text(model.tokenizer, manual["token_ids"]),
        },
    }

    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote Canary reference JSON to {args.output}")


if __name__ == "__main__":
    main()
