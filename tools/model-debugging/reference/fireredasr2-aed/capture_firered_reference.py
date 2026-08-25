"""Capture a deterministic FireRedASR2-AED native reference.

This helper is deliberately local-only. The checkpoint, source checkout, and
audio fixtures must already exist; it never resolves or downloads a model ID.
The output is the reference contract for later ONNX/WASM/WebGPU parity work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REQUIRED_MODEL_FILES = (
    "cmvn.ark",
    "model.pth.tar",
    "dict.txt",
    "train_bpe1000.model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture FireRedASR2-AED native token/text/timestamp references.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fire-red-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--decode-max-len", type=int, default=0)
    parser.add_argument("--softmax-smoothing", type=float, default=1.25)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--eos-penalty", type=float, default=1.0)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--use-half", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(model_dir: Path) -> list[dict[str, Any]]:
    manifest = []
    for name in REQUIRED_MODEL_FILES:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required FireRed checkpoint file is missing: {path}")
        manifest.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


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


def jsonable(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def tensor_payload(tensor: Any) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    return {
        "dims": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype).replace("torch.", ""),
        "data": detached.reshape(-1).tolist(),
    }


def decoder_teacher_forced_logits(
    decoder: Any,
    input_ids: Any,
    encoder_outputs: Any,
    src_mask: Any,
) -> Any:
    target_mask = decoder.ignored_target_position_is_0(input_ids, decoder.pad_id)
    decoder_output = decoder.dropout(
        decoder.tgt_word_emb(input_ids) * decoder.scale
        + decoder.positional_encoding(input_ids)
    )
    for layer in decoder.layer_stack:
        decoder_output = layer(
            decoder_output,
            encoder_outputs,
            target_mask,
            src_mask,
            cache=None,
        )
    decoder_output = decoder.layer_norm_out(decoder_output)
    return decoder.tgt_word_prj(decoder_output)


def capture_batch(
    asr: Any,
    audio_paths: list[Path],
    sample_ids: list[str],
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path_strings = [str(path) for path in audio_paths]
    features, lengths, durations, kept_paths, kept_ids = asr.feat_extractor(
        path_strings,
        sample_ids,
    )
    if features is None or lengths is None:
        raise RuntimeError("FireRed feature extraction returned no frames")
    if kept_ids != sample_ids or len(kept_paths) != len(audio_paths):
        raise RuntimeError(
            "FireRed feature extraction dropped or reordered an input; "
            "the reference capture requires stable batch order"
        )

    if asr.config.use_gpu:
        features = features.cuda()
        lengths = lengths.cuda()
        if asr.config.use_half:
            features = features.half()

    started = time.perf_counter()
    with torch.inference_mode():
        encoder_outputs, encoder_lengths, src_mask = asr.model.encoder(features, lengths)
        hypotheses = asr.model.decoder.batch_beam_search(
            encoder_outputs,
            src_mask,
            asr.config.beam_size,
            asr.config.nbest,
            asr.config.decode_max_len,
            asr.config.softmax_smoothing,
            asr.config.aed_length_penalty,
            asr.config.eos_penalty,
            asr.elm,
            asr.config.elm_weight,
        )
        if asr.config.return_timestamp:
            hypotheses = asr.model.get_token_timestamp_torchaudio(
                encoder_outputs,
                encoder_lengths,
                hypotheses,
            )
    elapsed_seconds = time.perf_counter() - started

    if len(hypotheses) != len(sample_ids):
        raise RuntimeError(
            f"FireRed returned {len(hypotheses)} hypotheses for {len(sample_ids)} inputs"
        )

    token_sequences = []
    for nbest_hypotheses in hypotheses:
        if not nbest_hypotheses:
            raise RuntimeError("FireRed returned an empty hypothesis list")
        token_sequences.append(
            [int(token_id) for token_id in nbest_hypotheses[0]["yseq"].detach().cpu().tolist()]
        )

    decoder_input_ids = torch.full(
        (len(token_sequences), max(len(tokens) for tokens in token_sequences) + 1),
        asr.model.decoder.pad_id,
        dtype=torch.long,
        device=encoder_outputs.device,
    )
    for row_index, token_ids in enumerate(token_sequences):
        sequence = [asr.model.decoder.sos_id, *token_ids]
        decoder_input_ids[row_index, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=encoder_outputs.device,
        )
    with torch.inference_mode():
        decoder_teacher_forced = decoder_teacher_forced_logits(
            asr.model.decoder,
            decoder_input_ids,
            encoder_outputs,
            src_mask,
        )

    ctc_logits = None
    if asr.config.return_timestamp:
        with torch.inference_mode():
            ctc_logits = asr.model.ctc(encoder_outputs)

    rows = []
    for sample_id, path, duration, nbest_hypotheses in zip(
        sample_ids,
        audio_paths,
        durations,
        hypotheses,
        strict=True,
    ):
        if not nbest_hypotheses:
            raise RuntimeError(f"FireRed returned no hypothesis for {sample_id}")
        hypothesis = nbest_hypotheses[0]
        token_ids = [int(token_id) for token_id in hypothesis["yseq"].detach().cpu().tolist()]
        token_pieces = [
            asr.tokenizer.detokenize([token_id], "", False) for token_id in token_ids
        ]
        native_text = re.sub(
            r"(<blank>)|(<sil>)",
            "",
            asr.tokenizer.detokenize(token_ids),
        )
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "audio_path": str(path),
            "audio_sha256": sha256_file(path),
            "duration_seconds": round(float(duration), 6),
            "text": native_text.lower(),
            "text_native": native_text,
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "confidence": float(hypothesis["confidence"].detach().cpu().item()),
            "batch_size": len(sample_ids),
            "batch_inference_seconds": round(elapsed_seconds, 6),
        }
        if asr.config.return_timestamp:
            row["timestamps"] = jsonable(
                asr._get_and_fix_timestamp(hypothesis, token_ids, float(duration))
            )
        rows.append(row)

    total_duration = sum(float(duration) for duration in durations)
    batch_info = {
        "sample_ids": sample_ids,
        "batch_size": len(sample_ids),
        "audio_duration_seconds": round(total_duration, 6),
        "inference_seconds": round(elapsed_seconds, 6),
        "rtf": round(elapsed_seconds / total_duration, 6) if total_duration > 0 else None,
    }
    stages = {
        "features": tensor_payload(features),
        "feature_lengths": [int(length) for length in lengths.detach().cpu().tolist()],
        "encoder_output": tensor_payload(encoder_outputs),
        "encoder_lengths": [int(length) for length in encoder_lengths.detach().cpu().tolist()],
        "src_mask": tensor_payload(src_mask),
        "decoder_input_ids": decoder_input_ids.detach().cpu().tolist(),
        "decoder_teacher_forced_logits": tensor_payload(decoder_teacher_forced),
    }
    if ctc_logits is not None:
        stages["ctc_logits"] = tensor_payload(ctc_logits)
    return rows, batch_info, stages


def main() -> None:
    args = parse_args()
    fire_red_root = args.fire_red_root.resolve()
    model_dir = args.model_dir.resolve()
    audio_paths = [path.resolve() for path in args.audio]
    if not (fire_red_root / "fireredasr2s").is_dir():
        raise FileNotFoundError(
            f"FireRed source package not found below {fire_red_root / 'fireredasr2s'}"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"FireRed model directory not found: {model_dir}")
    if args.batch_size < 1 or args.beam_size < 1 or args.nbest < 1:
        raise ValueError("batch size, beam size, and nbest must be positive")
    if args.nbest > args.beam_size:
        raise ValueError("nbest cannot exceed beam size")
    for audio_path in audio_paths:
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio fixture not found: {audio_path}")
    if args.device == "cpu" and args.use_half:
        raise ValueError("--use-half requires --device cuda")

    sys.path.insert(0, str(fire_red_root))
    import torch

    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model_files = file_manifest(model_dir)
    config = FireRedAsr2Config(
        use_gpu=args.device == "cuda",
        use_half=args.use_half,
        beam_size=args.beam_size,
        nbest=args.nbest,
        decode_max_len=args.decode_max_len,
        softmax_smoothing=args.softmax_smoothing,
        aed_length_penalty=args.length_penalty,
        eos_penalty=args.eos_penalty,
        return_timestamp=args.timestamps,
    )
    asr = FireRedAsr2.from_pretrained("aed", str(model_dir), config)

    rows_by_id: dict[str, dict[str, Any]] = {}
    batches = []
    stages = []
    for batch_index, start in enumerate(range(0, len(audio_paths), args.batch_size)):
        batch_paths = audio_paths[start : start + args.batch_size]
        batch_ids = [
            f"{index:04d}-{path.stem}"
            for index, path in enumerate(batch_paths, start=start)
        ]
        rows, batch_info, batch_stages = capture_batch(
            asr,
            batch_paths,
            batch_ids,
            torch,
        )
        for row in rows:
            row["batch_index"] = batch_index
            rows_by_id[row["sample_id"]] = row
        batch_info["batch_index"] = batch_index
        batches.append(batch_info)
        stages.append({"batch_index": batch_index, **batch_stages})

    rows = [
        rows_by_id[f"{index:04d}-{path.stem}"]
        for index, path in enumerate(audio_paths)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "reference_kind": "fireredasr2-aed-native-inference",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_only": True,
        "source": {
            "fire_red_root": str(fire_red_root),
            "fire_red_git_revision": git_revision(fire_red_root),
            "model_dir": str(model_dir),
            "model_files": model_files,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "dtype": "float16" if args.use_half else "float32",
        },
        "feature_contract": {
            "sample_rate_hz": 16000,
            "num_mel_bins": 80,
            "frame_length_ms": 25,
            "frame_shift_ms": 10,
            "snip_edges": True,
            "dither": 0.0,
            "cmvn": "cmvn.ark",
            "encoder_inputs": {
                "padded_input": "[batch, frames, 80]",
                "input_lengths": "[batch]",
            },
        },
        "decode": {
            "beam_size": args.beam_size,
            "nbest": args.nbest,
            "decode_max_len": args.decode_max_len,
            "softmax_smoothing": args.softmax_smoothing,
            "length_penalty": args.length_penalty,
            "eos_penalty": args.eos_penalty,
            "return_timestamp": args.timestamps,
        },
        "decoder_ids": {
            "sos_id": int(asr.model.decoder.sos_id),
            "eos_id": int(asr.model.decoder.eos_id),
            "pad_id": int(asr.model.decoder.pad_id),
        },
        "batching": {
            "requested_batch_size": args.batch_size,
            "batch_count": len(batches),
            "batches": batches,
        },
        "stages": stages,
        "samples": rows,
    }
    args.output.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote FireRed reference JSON to {args.output}")


if __name__ == "__main__":
    main()
