from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify exported Canary ONNX artifacts against a reference JSON.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder-name", default="encoder-model.onnx")
    parser.add_argument("--decoder-name", default="decoder-model.onnx")
    parser.add_argument("--preprocessor-name", default="nemo128.onnx")
    parser.add_argument("--tokenizer-name", default="tokenizer.json")
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_flat_tensor(payload: dict[str, Any], dtype: np.dtype) -> np.ndarray:
    dims = payload["dims"]
    data = np.asarray(payload["data"], dtype=dtype)
    return data.reshape(dims)


def load_tokenizer(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_ids(tokenizer: dict[str, Any], token_ids: list[int]) -> str:
    subtokenizers = tokenizer["subtokenizers"]
    special_ids = {int(value) for value in tokenizer.get("special_tokens", {}).values()}

    ranges: list[tuple[str, int, int]] = []
    for lang, spec in subtokenizers.items():
        offset = int(spec["offset"])
        size = int(spec["size"])
        ranges.append((lang, offset, offset + size))
    ranges.sort(key=lambda item: item[1])

    pieces: list[str] = []
    for token_id in token_ids:
        if token_id in special_ids:
            continue
        for lang, start, end in ranges:
            if start <= token_id < end:
                spec = subtokenizers[lang]
                pieces.append(spec["pieces"][token_id - start])
                break

    return "".join(pieces).replace("\u2581", " ").strip()


def greedy_decode_onnx(
    decoder_session: ort.InferenceSession,
    tokenizer: dict[str, Any],
    encoder_states: np.ndarray,
    encoder_mask: np.ndarray,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    eos_id = int(tokenizer["eos_id"])
    token_ids: list[int] = []
    input_ids = np.asarray([prompt_ids], dtype=np.int64)

    for _ in range(max_new_tokens):
        logits = decoder_session.run(
            None,
            {
                "input_ids": input_ids,
                "encoder_states": encoder_states.astype(np.float32, copy=False),
                "encoder_mask": encoder_mask.astype(np.float32, copy=False),
            },
        )[0]
        next_id = int(np.argmax(logits[0], axis=-1))
        token_ids.append(next_id)
        input_ids = np.concatenate([input_ids, np.asarray([[next_id]], dtype=np.int64)], axis=1)
        if next_id == eos_id:
            break

    return {
        "token_ids": token_ids,
        "text": decode_ids(tokenizer, token_ids),
    }


def summarize_numeric_diff(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    diff = np.abs(lhs - rhs)
    return {
        "shape_match": list(lhs.shape) == list(rhs.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }


def evaluate_verification(verification: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    preprocessor = verification.get("preprocessor", {})
    if preprocessor.get("skipped"):
        preprocessor_scope = "SKIPPED_REFERENCE_FEATURES"
    else:
        preprocessor_scope = "CHECKED_ONNX"
        if not preprocessor.get("shape_match"):
            failures.append("preprocessor_shape_mismatch")
        if not preprocessor.get("lengths_match"):
            failures.append("preprocessor_length_mismatch")

    encoder = verification.get("encoder", {})
    if not encoder.get("shape_match"):
        failures.append("encoder_shape_mismatch")
    if not encoder.get("mask", {}).get("shape_match"):
        failures.append("encoder_mask_shape_mismatch")
    mask_values_match = (
        encoder.get("mask", {}).get("max_abs_diff") == 0.0
        and encoder.get("mask", {}).get("mean_abs_diff") == 0.0
    )
    if not mask_values_match:
        failures.append("encoder_mask_value_mismatch")
    if not encoder.get("lengths_match"):
        failures.append("encoder_length_mismatch")

    decoded = verification.get("decode", {})
    if not decoded.get("token_ids_match"):
        failures.append("decoder_token_ids_mismatch")
    if not decoded.get("text_match"):
        failures.append("decoder_text_mismatch")

    verification["status"] = "FAIL_COMPONENT_GATES" if failures else "PASS_GATED_COMPONENTS"
    verification["scope"] = {
        "kind": "nemo-aed-onnx-component-parity",
        "preprocessor": preprocessor_scope,
        "encoder": "CHECKED",
        "decoder": "CHECKED",
        "browser_runtime": "NOT_RUN",
    }
    verification["gates"] = {
        "preprocessor_shape_and_lengths": (
            "SKIPPED" if preprocessor.get("skipped") else
            bool(preprocessor.get("shape_match") and preprocessor.get("lengths_match"))
        ),
        "encoder_shapes_lengths_and_mask": bool(
            encoder.get("shape_match")
            and encoder.get("mask", {}).get("shape_match")
            and mask_values_match
            and encoder.get("lengths_match")
        ),
        "decoder_token_ids_and_text": bool(decoded.get("token_ids_match") and decoded.get("text_match")),
        "numeric_differences": "MEASURED_NOT_GATED",
    }
    verification["failures"] = failures
    return verification


def run_verification(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_reference(args.reference)
    tokenizer = load_tokenizer(args.model_dir / args.tokenizer_name)

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
    reference_encoder_states = load_flat_tensor(reference["encoder"]["states"], np.float32)
    reference_encoder_mask = load_flat_tensor(reference["encoder"]["mask"], np.float32)
    feature_lengths = np.asarray(reference["preprocessor"]["feature_lengths"], dtype=np.int64)
    prompt_ids = list(reference["prompt"]["ids"])

    verification: dict[str, Any] = {
        "reference": str(args.reference),
        "model_dir": str(args.model_dir),
        "encoder_name": args.encoder_name,
        "decoder_name": args.decoder_name,
        "preprocessor_name": args.preprocessor_name,
        "tokenizer_name": args.tokenizer_name,
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
        verification["preprocessor"] = {
            "skipped": True,
            "scope": "reference frontend features supplied directly to the encoder",
        }
        encoder_input = reference_features.astype(np.float32, copy=False)
        encoder_lengths = feature_lengths

    onnx_encoder_states, onnx_encoded_length, onnx_encoder_mask = encoder_session.run(
        None,
        {
            "processed_signal": encoder_input,
            "processed_signal_length": encoder_lengths,
        },
    )
    verification["encoder"] = {
        **summarize_numeric_diff(reference_encoder_states, np.asarray(onnx_encoder_states, dtype=np.float32)),
        "mask": summarize_numeric_diff(reference_encoder_mask, np.asarray(onnx_encoder_mask, dtype=np.float32)),
        "lengths_match": list(reference["encoder"]["lengths"]) == np.asarray(onnx_encoded_length).tolist(),
    }

    decoded = greedy_decode_onnx(
        decoder_session,
        tokenizer,
        np.asarray(onnx_encoder_states, dtype=np.float32),
        np.asarray(onnx_encoder_mask, dtype=np.float32),
        prompt_ids,
        max_new_tokens=int(reference["runtime_config"]["max_target_positions"]) - len(prompt_ids),
    )
    reference_manual = reference["decode"]["manual_greedy"]
    verification["decode"] = {
        "token_ids_match": decoded["token_ids"] == reference_manual["token_ids"],
        "text_match": decoded["text"] == reference_manual["text"],
        "reference_text": reference_manual["text"],
        "onnx_text": decoded["text"],
        "reference_token_ids": reference_manual["token_ids"],
        "onnx_token_ids": decoded["token_ids"],
    }

    return verification


def main() -> int:
    args = parse_args()
    try:
        verification = evaluate_verification(run_verification(args))
        exit_code = 0 if verification["status"] == "PASS_GATED_COMPONENTS" else 1
    except Exception as error:  # noqa: BLE001
        verification = {
            "status": "ERROR",
            "scope": {
                "kind": "nemo-aed-onnx-component-parity",
                "preprocessor": "UNKNOWN",
                "encoder": "UNKNOWN",
                "decoder": "UNKNOWN",
                "browser_runtime": "NOT_RUN",
            },
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 2

    encoded = json.dumps(verification, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
