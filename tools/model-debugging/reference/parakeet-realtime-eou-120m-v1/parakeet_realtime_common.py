from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

PROJECT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_AUDIO = PROJECT_DIR / "tools" / "data" / "fixtures" / "audio" / "jfk-short.wav"
DEFAULT_MODEL_ID = "nvidia/parakeet_realtime_eou_120m-v1"
DEFAULT_MODEL_FILENAME = "parakeet_realtime_eou_120m-v1.nemo"


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available in this environment.")
    return name


def resolve_model_path(model_id: str, model_path: Path | None = None) -> Path:
    if model_path is not None:
        resolved = model_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Model package not found: {resolved}")
        return resolved

    if model_id != DEFAULT_MODEL_ID:
        raise ValueError(
            "Automatic model download is only configured for the default Parakeet realtime model. "
            "Pass --model-path for custom model ids."
        )

    downloaded = hf_hub_download(model_id, DEFAULT_MODEL_FILENAME)
    return Path(downloaded).resolve()


def load_model(model_id: str, device: str, model_path: Path | None = None) -> Any:
    resolved_model_path = resolve_model_path(model_id, model_path)
    model = ASRModel.restore_from(str(resolved_model_path), map_location=device)
    model.eval()
    model.joint.set_fuse_loss_wer(False)
    return model


def load_audio(audio_path: Path, expected_sample_rate: int) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(str(audio_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if int(sample_rate) != int(expected_sample_rate):
        raise ValueError(
            f"Expected {expected_sample_rate} Hz audio but found {sample_rate} Hz in {audio_path}"
        )
    return waveform.astype(np.float32, copy=False), int(sample_rate)


def ids_to_pieces(tokenizer: Any, token_ids: list[int]) -> list[str]:
    if not token_ids:
        return []
    if hasattr(tokenizer, "ids_to_tokens"):
        return [str(piece) for piece in tokenizer.ids_to_tokens(token_ids)]
    return [str(tokenizer.ids_to_text([token_id])) for token_id in token_ids]


def sanitize_pieces(pieces: list[str], skip_control_tokens: bool) -> str:
    text_parts: list[str] = []
    for piece in pieces:
        if skip_control_tokens and piece.startswith("<") and piece.endswith(">"):
            continue
        text_parts.append(piece)

    return (
        "".join(text_parts)
        .replace("\u2581", " ")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .strip()
    )


def decode_visible_text(tokenizer: Any, token_ids: list[int]) -> str:
    return sanitize_pieces(ids_to_pieces(tokenizer, token_ids), skip_control_tokens=True)


def decode_raw_text(tokenizer: Any, token_ids: list[int]) -> str:
    return sanitize_pieces(ids_to_pieces(tokenizer, token_ids), skip_control_tokens=False)


def tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    return {
        "dims": [int(dim) for dim in detached.shape],
        "data": detached.reshape(-1).tolist(),
    }


def numpy_payload(array: np.ndarray, dims: list[int] | tuple[int, ...]) -> dict[str, Any]:
    return {
        "dims": [int(dim) for dim in dims],
        "data": np.asarray(array).reshape(-1).tolist(),
    }


def export_vocab_lines(tokenizer: Any) -> list[str]:
    return ids_to_pieces(tokenizer, list(range(int(tokenizer.vocab_size))))


def build_runtime_config(model_id: str, model: Any) -> dict[str, Any]:
    eou_id = None
    eob_id = None
    for index, piece in enumerate(export_vocab_lines(model.tokenizer)):
        if piece == "<EOU>":
            eou_id = index
        if piece == "<EOB>":
            eob_id = index

    return {
        "model_id": model_id,
        "sample_rate": int(model.cfg.preprocessor.sample_rate),
        "mel_bins": int(model.cfg.preprocessor.features),
        "frame_shift_seconds": float(model.cfg.preprocessor.window_stride),
        "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
        "vocab_size": int(model.cfg.decoder.vocab_size),
        "blank_id": int(model.decoder.blank_idx),
        "eou_id": eou_id,
        "eob_id": eob_id,
        "pred_hidden": int(model.cfg.decoder.prednet.pred_hidden),
        "pred_layers": int(model.cfg.decoder.prednet.pred_rnn_layers),
        "max_symbols_per_step": int(model.cfg.decoding.greedy.max_symbols),
    }


def run_model_transcribe(model: Any, audio_path: Path) -> dict[str, Any]:
    hypotheses = model.transcribe([str(audio_path)], batch_size=1, return_hypotheses=True)
    hypothesis = hypotheses[0]
    text = str(hypothesis.text)
    return {
        "text": text,
        "contains_eou": "<EOU>" in text,
        "contains_eob": "<EOB>" in text,
    }


def manual_greedy_decode(
    model: Any,
    encoder_outputs: torch.Tensor,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    blank_id = int(runtime_config["blank_id"])
    max_symbols_per_step = int(runtime_config["max_symbols_per_step"])
    pred_hidden = int(runtime_config["pred_hidden"])
    pred_layers = int(runtime_config["pred_layers"])
    device = encoder_outputs.device

    state = [
        torch.zeros(pred_layers, 1, pred_hidden, dtype=torch.float32, device=device),
        torch.zeros(pred_layers, 1, pred_hidden, dtype=torch.float32, device=device),
    ]
    token_ids: list[int] = []
    frame_indices: list[int] = []
    frame_count = int(encoder_outputs.shape[-1])

    with torch.inference_mode():
        for frame_index in range(frame_count):
            encoder_frame = encoder_outputs[:, :, frame_index : frame_index + 1]
            emitted_on_frame = 0
            while emitted_on_frame < max_symbols_per_step:
                current_token = token_ids[-1] if token_ids else blank_id
                targets = torch.tensor([[current_token]], dtype=torch.long, device=device)
                target_length = torch.tensor([1], dtype=torch.long, device=device)
                decoder_outputs, _, next_state = model.decoder(
                    targets=targets,
                    target_length=target_length,
                    states=state,
                )
                logits = model.joint(
                    encoder_outputs=encoder_frame,
                    decoder_outputs=decoder_outputs,
                )
                step_logits = logits[:, -1, -1, :]
                next_id = int(torch.argmax(step_logits[0]).item())
                if next_id == blank_id:
                    break

                token_ids.append(next_id)
                frame_indices.append(frame_index)
                state = [next_state[0], next_state[1]]
                emitted_on_frame += 1

    pieces = ids_to_pieces(model.tokenizer, token_ids)
    raw_text = decode_raw_text(model.tokenizer, token_ids)
    visible_text = decode_visible_text(model.tokenizer, token_ids)
    return {
        "token_ids": token_ids,
        "token_pieces": pieces,
        "frame_indices": frame_indices,
        "raw_text": raw_text,
        "text": visible_text,
        "contains_eou": int(runtime_config["eou_id"]) in token_ids
        if runtime_config["eou_id"] is not None
        else False,
        "contains_eob": int(runtime_config["eob_id"]) in token_ids
        if runtime_config["eob_id"] is not None
        else False,
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
