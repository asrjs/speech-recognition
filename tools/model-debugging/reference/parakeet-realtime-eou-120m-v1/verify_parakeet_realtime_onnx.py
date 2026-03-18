from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify exported Parakeet realtime ONNX artifacts against a reference JSON.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder-name", default="encoder-model.onnx")
    parser.add_argument("--decoder-name", default="decoder_joint-model.onnx")
    parser.add_argument("--vocab-name", default="vocab.txt")
    parser.add_argument("--preprocessor-name", default="nemo128.onnx")
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_flat_tensor(payload: dict[str, Any], dtype: np.dtype) -> np.ndarray:
    dims = payload["dims"]
    data = np.asarray(payload["data"], dtype=dtype)
    return data.reshape(dims)


def load_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def decode_text(vocab: list[str], token_ids: list[int], skip_control_tokens: bool) -> str:
    pieces: list[str] = []
    for token_id in token_ids:
        if token_id < 0 or token_id >= len(vocab):
            continue
        piece = vocab[token_id]
        if skip_control_tokens and piece.startswith("<") and piece.endswith(">"):
            continue
        pieces.append(piece)

    return (
        "".join(pieces)
        .replace("\u2581", " ")
        .replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .strip()
    )


def summarize_numeric_diff(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    diff = np.abs(lhs - rhs)
    return {
        "shape_match": list(lhs.shape) == list(rhs.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }


def decode_greedy_onnx(
    decoder_session: ort.InferenceSession,
    encoder_outputs: np.ndarray,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    blank_id = int(runtime_config["blank_id"])
    max_symbols_per_step = int(runtime_config["max_symbols_per_step"])
    pred_layers = int(runtime_config["pred_layers"])
    pred_hidden = int(runtime_config["pred_hidden"])
    state_1 = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
    state_2 = np.zeros((pred_layers, 1, pred_hidden), dtype=np.float32)
    token_ids: list[int] = []
    frame_indices: list[int] = []

    frame_count = int(encoder_outputs.shape[-1])
    for frame_index in range(frame_count):
        encoder_frame = encoder_outputs[:, :, frame_index : frame_index + 1].astype(np.float32, copy=False)
        emitted_on_frame = 0
        while emitted_on_frame < max_symbols_per_step:
            current_token = token_ids[-1] if token_ids else blank_id
            feeds = {
                "encoder_outputs": encoder_frame,
                "targets": np.asarray([[current_token]], dtype=np.int32),
                "input_states_1": state_1,
                "input_states_2": state_2,
            }
            required_inputs = {item.name for item in decoder_session.get_inputs()}
            if "target_length" in required_inputs:
                feeds["target_length"] = np.asarray([1], dtype=np.int32)
            outputs = decoder_session.run(None, feeds)
            logits = np.asarray(outputs[0], dtype=np.float32)
            step_logits = logits.reshape(-1, logits.shape[-1])[-1]
            next_id = int(np.argmax(step_logits))
            if next_id == blank_id:
                break

            token_ids.append(next_id)
            frame_indices.append(frame_index)
            state_1 = np.asarray(outputs[1], dtype=np.float32)
            state_2 = np.asarray(outputs[2], dtype=np.float32)
            emitted_on_frame += 1

    return {
        "token_ids": token_ids,
        "frame_indices": frame_indices,
    }


def main() -> None:
    args = parse_args()
    reference = load_reference(args.reference)
    vocab = load_vocab(args.model_dir / args.vocab_name)

    preprocessor_session = None
    preprocessor_path = args.model_dir / args.preprocessor_name
    if preprocessor_path.exists():
        preprocessor_session = ort.InferenceSession(str(preprocessor_path), providers=["CPUExecutionProvider"])
    encoder_session = ort.InferenceSession(
        str(args.model_dir / args.encoder_name),
        providers=["CPUExecutionProvider"],
    )
    decoder_session = ort.InferenceSession(
        str(args.model_dir / args.decoder_name),
        providers=["CPUExecutionProvider"],
    )

    waveform = load_flat_tensor(reference["audio"]["waveform"], np.float32)
    reference_features = load_flat_tensor(reference["preprocessor"]["features"], np.float32)
    reference_encoder_outputs = load_flat_tensor(reference["encoder"]["outputs"], np.float32)
    feature_lengths = np.asarray(reference["preprocessor"]["feature_lengths"], dtype=np.int64)
    runtime_config = reference["runtime_config"]

    verification: dict[str, Any] = {
        "reference": str(args.reference),
        "model_dir": str(args.model_dir),
        "encoder_name": args.encoder_name,
        "decoder_name": args.decoder_name,
        "preprocessor_name": args.preprocessor_name,
        "vocab_name": args.vocab_name,
    }

    if preprocessor_session is not None:
        onnx_features, onnx_feature_lengths = preprocessor_session.run(
            None,
            {
                "waveforms": waveform.astype(np.float32, copy=False),
                "waveforms_lens": np.asarray([waveform.shape[-1]], dtype=np.int64),
            },
        )
        verification["preprocessor"] = {
            **summarize_numeric_diff(reference_features, np.asarray(onnx_features, dtype=np.float32)),
            "lengths_match": feature_lengths.tolist() == np.asarray(onnx_feature_lengths, dtype=np.int64).tolist(),
        }
        encoder_input = np.asarray(onnx_features, dtype=np.float32)
        encoder_lengths = np.asarray(onnx_feature_lengths, dtype=np.int64)
    else:
        verification["preprocessor"] = {"skipped": True}
        encoder_input = reference_features.astype(np.float32, copy=False)
        encoder_lengths = feature_lengths

    onnx_encoder_outputs = encoder_session.run(
        None,
        {
            "audio_signal": encoder_input,
            "length": encoder_lengths,
        },
    )[0]
    verification["encoder"] = summarize_numeric_diff(
        reference_encoder_outputs,
        np.asarray(onnx_encoder_outputs, dtype=np.float32),
    )

    decoded = decode_greedy_onnx(
        decoder_session,
        np.asarray(onnx_encoder_outputs, dtype=np.float32),
        runtime_config,
    )
    decoded["raw_text"] = decode_text(vocab, decoded["token_ids"], skip_control_tokens=False)
    decoded["text"] = decode_text(vocab, decoded["token_ids"], skip_control_tokens=True)
    eou_id = runtime_config.get("eou_id")
    eob_id = runtime_config.get("eob_id")
    decoded["contains_eou"] = eou_id in decoded["token_ids"] if eou_id is not None else False
    decoded["contains_eob"] = eob_id in decoded["token_ids"] if eob_id is not None else False

    reference_manual = reference["decode"]["manual_greedy"]
    verification["decode"] = {
        "token_ids_match": decoded["token_ids"] == reference_manual["token_ids"],
        "frame_indices_match": decoded["frame_indices"] == reference_manual["frame_indices"],
        "text_match": decoded["text"] == reference_manual["text"],
        "raw_text_match": decoded["raw_text"] == reference_manual["raw_text"],
        "contains_eou_match": decoded["contains_eou"] == reference_manual["contains_eou"],
        "contains_eob_match": decoded["contains_eob"] == reference_manual["contains_eob"],
        "reference_text": reference_manual["text"],
        "onnx_text": decoded["text"],
        "reference_raw_text": reference_manual["raw_text"],
        "onnx_raw_text": decoded["raw_text"],
        "reference_token_ids": reference_manual["token_ids"],
        "onnx_token_ids": decoded["token_ids"],
    }

    encoded = json.dumps(verification, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
