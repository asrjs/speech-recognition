"""Verify FireRedASR2-AED ONNX graphs against a captured native reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FireRed ONNX encoder/decoder/CTC graphs with a reference JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def tensor_from_payload(payload: dict[str, Any]) -> np.ndarray:
    dtype_name = payload.get("dtype", "float32")
    dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "uint8": np.uint8,
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported reference tensor dtype: {dtype_name}")
    return np.asarray(payload["data"], dtype=dtype).reshape(payload["dims"])


def numeric_diff(
    reference: np.ndarray,
    actual: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    shape_match = list(reference.shape) == list(actual.shape)
    if not shape_match:
        return {
            "shape_match": False,
            "allclose": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    difference = np.abs(reference.astype(np.float32) - actual.astype(np.float32))
    return {
        "shape_match": True,
        "allclose": bool(np.allclose(reference, actual, atol=atol, rtol=rtol)),
        "max_abs_diff": float(difference.max()) if difference.size else 0.0,
        "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_session(path: Path) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"ONNX graph not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def session_dtype(session: ort.InferenceSession, name: str) -> np.dtype:
    for input_info in session.get_inputs():
        if input_info.name != name:
            continue
        type_name = input_info.type
        if type_name == "tensor(float16)":
            return np.dtype(np.float16)
        if type_name == "tensor(float)":
            return np.dtype(np.float32)
        if type_name == "tensor(double)":
            return np.dtype(np.float64)
        if type_name == "tensor(int64)":
            return np.dtype(np.int64)
        if type_name == "tensor(int32)":
            return np.dtype(np.int32)
        if type_name == "tensor(uint8)":
            return np.dtype(np.uint8)
        if type_name == "tensor(bool)":
            return np.dtype(np.bool_)
        raise ValueError(f"Unsupported ONNX input type for {name}: {type_name}")
    raise KeyError(f"ONNX input not found: {name}")


def cast_for_session(
    session: ort.InferenceSession,
    name: str,
    value: np.ndarray,
) -> np.ndarray:
    return value.astype(session_dtype(session, name), copy=False)


def run_greedy(
    decoder_session: ort.InferenceSession,
    encoder_outputs: np.ndarray,
    src_mask: np.ndarray,
    sos_id: int,
    eos_id: int,
    max_new_tokens: int,
) -> list[int]:
    token_ids: list[int] = []
    input_dtype = session_dtype(decoder_session, "input_ids")
    input_ids = np.asarray([[sos_id]], dtype=input_dtype)
    encoder_outputs = cast_for_session(
        decoder_session,
        "encoder_outputs",
        encoder_outputs,
    )
    src_mask = cast_for_session(decoder_session, "src_mask", src_mask)
    for _ in range(max_new_tokens):
        logits = decoder_session.run(
            None,
            {
                "input_ids": input_ids,
                "encoder_outputs": encoder_outputs,
                "src_mask": src_mask,
            },
        )[0]
        next_id = int(np.argmax(logits[0, -1]))
        if next_id == eos_id:
            break
        token_ids.append(next_id)
        input_ids = np.concatenate(
            [input_ids, np.asarray([[next_id]], dtype=input_dtype)],
            axis=1,
        )
    return token_ids


def main() -> None:
    args = parse_args()
    reference = load_json(args.reference)
    stages = reference.get("stages", [])
    if not stages:
        raise ValueError("Reference JSON has no stage captures; regenerate it with the current capture tool")
    if args.batch_index < 0 or args.batch_index >= len(stages):
        raise IndexError(f"batch index {args.batch_index} is outside the captured stage range")
    stage = stages[args.batch_index]
    reference_model_dir = args.model_dir.resolve()
    encoder_session = load_session(reference_model_dir / "encoder.onnx")
    decoder_session = load_session(reference_model_dir / "decoder.onnx")
    ctc_path = reference_model_dir / "ctc.onnx"
    ctc_session = load_session(ctc_path) if ctc_path.is_file() else None

    features = tensor_from_payload(stage["features"])
    feature_lengths = np.asarray(stage["feature_lengths"], dtype=np.int64)
    encoder_reference = tensor_from_payload(stage["encoder_output"])
    encoder_lengths_reference = np.asarray(stage["encoder_lengths"], dtype=np.int64)
    mask_reference = tensor_from_payload(stage["src_mask"])
    encoder_outputs, encoder_lengths, src_mask = encoder_session.run(
        None,
        {
            "padded_input": cast_for_session(encoder_session, "padded_input", features),
            "input_lengths": cast_for_session(
                encoder_session,
                "input_lengths",
                feature_lengths,
            ),
        },
    )
    encoder_outputs = np.asarray(encoder_outputs)
    encoder_lengths = np.asarray(encoder_lengths)
    src_mask = np.asarray(src_mask)

    verification: dict[str, Any] = {
        "schema_version": 1,
        "reference": str(args.reference.resolve()),
        "model_dir": str(reference_model_dir),
        "batch_index": args.batch_index,
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "encoder": {
            "states": numeric_diff(
                encoder_reference,
                encoder_outputs,
                args.atol,
                args.rtol,
            ),
            "lengths_match": encoder_lengths_reference.tolist() == encoder_lengths.tolist(),
            "mask": numeric_diff(mask_reference, src_mask, args.atol, args.rtol),
        },
    }

    decoder_input_ids = np.asarray(stage["decoder_input_ids"], dtype=np.int64)
    decoder_reference = tensor_from_payload(stage["decoder_teacher_forced_logits"])
    decoder_logits = np.asarray(
        decoder_session.run(
            None,
            {
                "input_ids": cast_for_session(
                    decoder_session,
                    "input_ids",
                    decoder_input_ids,
                ),
                "encoder_outputs": cast_for_session(
                    decoder_session,
                    "encoder_outputs",
                    encoder_outputs,
                ),
                "src_mask": cast_for_session(decoder_session, "src_mask", src_mask),
            },
        )[0]
    )
    verification["decoder"] = {
        "teacher_forced_logits": numeric_diff(
            decoder_reference,
            decoder_logits,
            args.atol,
            args.rtol,
        ),
    }

    if ctc_session is not None and "ctc_logits" in stage:
        ctc_reference = tensor_from_payload(stage["ctc_logits"])
        ctc_logits = np.asarray(
            ctc_session.run(
                None,
                {
                    "encoder_outputs": cast_for_session(
                        ctc_session,
                        "encoder_outputs",
                        encoder_outputs,
                    )
                },
            )[0]
        )
        verification["ctc"] = numeric_diff(
            ctc_reference,
            ctc_logits,
            args.atol,
            args.rtol,
        )
    else:
        verification["ctc"] = {"skipped": True}

    decoder_ids = reference.get("decoder_ids")
    if decoder_ids:
        max_tokens = max(
            len(sample.get("token_ids", []))
            for sample in reference.get("samples", [])
        )
        if max_tokens > 0:
            greedy_tokens = run_greedy(
                decoder_session,
                encoder_outputs[:1],
                src_mask[:1],
                int(decoder_ids["sos_id"]),
                int(decoder_ids["eos_id"]),
                max_tokens,
            )
            verification["greedy_batch0"] = {
                "token_ids": greedy_tokens,
                "reference_token_ids": reference["samples"][0]["token_ids"],
                "matches_reference_first_sample": (
                    greedy_tokens == reference["samples"][0]["token_ids"]
                ),
                "note": "A beam reference is not expected to equal greedy output unless beam size is one.",
            }

    numeric_checks = [
        item
        for section in ("encoder", "decoder", "ctc")
        for item in (
            [verification[section]["states"]]
            if section == "encoder"
            else [verification[section]]
            if section == "ctc" and "skipped" not in verification[section]
            else [verification[section]["teacher_forced_logits"]]
            if section == "decoder"
            else []
        )
    ]
    verification["ok"] = bool(
        all(item.get("shape_match", False) and item.get("allclose", False) for item in numeric_checks)
        and verification["encoder"]["lengths_match"]
        and verification["encoder"]["mask"].get("shape_match", False)
        and verification["encoder"]["mask"].get("allclose", False)
    )

    encoded = json.dumps(verification, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    if not verification["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
