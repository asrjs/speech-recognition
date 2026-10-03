from __future__ import annotations

import argparse
import faulthandler
import gzip
import hashlib
import inspect
import json
import os
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import torch
from nemo.collections.asr.models import ASRModel

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from artifact_outputs import require_new_outputs, write_new_text


def default_audio_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "fixtures" / "audio" / "librivox.org.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeMo ASR inference and dump hypothesis/token details for debugging decode parity.",
    )
    parser.add_argument("--audio", type=Path, default=default_audio_path())
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--restore-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/nemo-reference"))
    parser.add_argument("--stack-timeout", type=int, default=60)
    parser.add_argument("--stage-output", type=Path, help="Optional CPU TDT capture JSON or JSON.gz")
    parser.add_argument("--capture-feature-stft", action="store_true",
                        help="With --stage-output, capture actual input/window/filterbank/complex components and layouts")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--capture-max-calls", type=int, default=4096)
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    return value


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def find_context(text: str, needle: str, radius: int = 64) -> str | None:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return None
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    return text[start:end]


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def hypothesis_token_ids(hypothesis: Any) -> list[int]:
    seq = getattr(hypothesis, "y_sequence", None)
    if seq is None:
        return []
    if hasattr(seq, "tolist"):
        return [int(x) for x in seq.tolist()]
    return [int(x) for x in seq]


def decode_pieces(model: Any, token_ids: Iterable[int]) -> list[str] | None:
    tokenizer = getattr(model, "tokenizer", None)
    ids = list(token_ids)
    if tokenizer is None or not ids:
        return None

    if hasattr(tokenizer, "ids_to_tokens"):
        try:
            return list(tokenizer.ids_to_tokens(ids))
        except Exception:  # noqa: BLE001
            pass

    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "id_to_piece"):
        try:
            return [inner.id_to_piece(int(i)) for i in ids]
        except Exception:  # noqa: BLE001
            pass

    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "id_to_piece"):
        try:
            return [inner.id_to_piece(int(i)) for i in ids]
        except Exception:  # noqa: BLE001
            pass

    return None


def load_model(args: argparse.Namespace, device: str) -> Any:
    if args.restore_path:
        model = ASRModel.restore_from(str(args.restore_path), map_location=device)
    else:
        model = ASRModel.from_pretrained(model_name=args.model, map_location=device)
    model = model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def call_transcribe(model: Any, audio_path: Path, args: argparse.Namespace) -> Any:
    kwargs = {
        "batch_size": args.batch_size,
        "return_hypotheses": True,
        "verbose": False,
        "timestamps": args.timestamps,
    }
    try:
        accepted = set(inspect.signature(model.transcribe).parameters)
    except Exception:  # noqa: BLE001
        accepted = set(kwargs)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return model.transcribe([str(audio_path)], **filtered_kwargs)


def main() -> None:
    args = parse_args()
    require_new_outputs([args.output, args.stage_output], protected_inputs=[args.audio, args.restore_path])
    if args.capture_feature_stft and not args.stage_output:
        raise ValueError("--capture-feature-stft requires --stage-output")
    if args.cpu_threads < 1 or args.stack_timeout < 0:
        raise ValueError("--cpu-threads must be positive; --stack-timeout must be non-negative")
    numba_cache = args.cache_dir.resolve() / "numba"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    # NeMo has already imported Numba; changing the environment alone does not
    # update its cached config. Reload before Librosa's lazy JIT initialization.
    import numba

    numba.config.reload_config()
    torch.set_num_threads(args.cpu_threads)
    faulthandler.enable()
    if args.stack_timeout:
        faulthandler.dump_traceback_later(args.stack_timeout, repeat=True)
    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = resolve_device(args.device)
    print(f"Restoring {args.restore_path or args.model} on {device} ...", flush=True)
    model = load_model(args, device)
    print(f"Transcribing {audio_path.name} ...", flush=True)
    capture = None
    capture_control = None
    if args.stage_output:
        if device != "cpu" or args.batch_size != 1:
            raise ValueError("--stage-output requires --device cpu and --batch-size 1")
        from nemo_tdt_stage_capture import NemoTdtStageCapture
        from omegaconf import OmegaConf

        baseline = call_transcribe(model, audio_path, args)[0]
        decoding_cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
        decoding_cfg.tdt_include_token_duration = True
        model.change_decoding_strategy(decoding_cfg, verbose=False)
        with NemoTdtStageCapture(model, max_calls=args.capture_max_calls,
                                 capture_feature_stft=args.capture_feature_stft) as capture:
            results = call_transcribe(model, audio_path, args)
        capture.validate_complete()
        model_input = capture.model_input_contract(int(model.preprocessor._sample_rate))
        after = call_transcribe(model, audio_path, args)[0]
        captured = results[0]
        capture_control = {
            "baseline_tokens_match": hypothesis_token_ids(baseline) == hypothesis_token_ids(captured),
            "baseline_text_match": baseline.text == captured.text,
            "baseline_frames_match": normalize_for_json(baseline.timestamp) == normalize_for_json(captured.timestamp),
            "after_tokens_match": hypothesis_token_ids(after) == hypothesis_token_ids(captured),
            "after_text_match": after.text == captured.text,
            "after_durations_match": normalize_for_json(after.token_duration) == normalize_for_json(captured.token_duration),
        }
        if not all(capture_control.values()):
            raise RuntimeError(f"Instrumentation/duration metadata changed official output: {capture_control}")
    else:
        results = call_transcribe(model, audio_path, args)
    if not results:
        raise RuntimeError("NeMo returned no transcription results.")

    hypothesis = results[0]
    token_ids = hypothesis_token_ids(hypothesis)
    token_pieces = decode_pieces(model, token_ids)

    payload = {
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "nemo_toolkit": version("nemo_toolkit"),
            "audio": str(audio_path),
            "model": args.model,
            "model_revision": args.model_revision,
            "restore_path": str(args.restore_path.resolve()) if args.restore_path else None,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "batch_size": args.batch_size,
            "cpu_threads": args.cpu_threads,
            "numba_cache": str(numba_cache),
            "timestamps": bool(args.timestamps),
            "capture_feature_stft": args.capture_feature_stft,
        },
        "result": {
            "text": getattr(hypothesis, "text", None),
            "normalized_text": normalize_text(getattr(hypothesis, "text", "")),
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "score": normalize_for_json(getattr(hypothesis, "score", None)),
            "timestamp": normalize_for_json(getattr(hypothesis, "timestamp", None)),
            "timestep": normalize_for_json(getattr(hypothesis, "timestep", None)),
            "token_duration": normalize_for_json(getattr(hypothesis, "token_duration", None)),
            "context_librivox": find_context(getattr(hypothesis, "text", ""), "librivox"),
            "context_domain": find_context(getattr(hypothesis, "text", ""), "librivox.org"),
        },
    }

    if capture is not None:
        def sha256_file(file_path: Path) -> str:
            digest = hashlib.sha256()
            with file_path.open("rb") as file:
                for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        payload["meta"]["audio_sha256"] = sha256_file(audio_path)
        payload["meta"]["weight_sha256"] = sha256_file(args.restore_path) if args.restore_path else None
        if args.capture_feature_stft:
            import nemo_tdt_stage_capture
            payload["meta"]["preprocessor_config"] = OmegaConf.to_container(model.cfg.preprocessor, resolve=True)
            payload["meta"]["torch_build_config"] = torch.__config__.show()
            payload["meta"]["capture_tool_sha256"] = sha256_file(Path(nemo_tdt_stage_capture.__file__))
            payload["meta"]["inspect_tool_sha256"] = sha256_file(Path(__file__))
        payload["stage_capture"] = {
            "path": str(args.stage_output.resolve()), "stage_count": len(capture.stages),
            "call_counts": capture.counts, "instrumentation_control": capture_control,
            "model_input": model_input,
            "feature_stft_contract": capture.feature_stft_contract,
            "numerical_export_parity": "UNTESTED", "browser_parity": "UNTESTED",
        }
        stage_payload = {
            "schema_version": 1, "meta": payload["meta"],
            "stage_order": list(capture.stages), "instrumentation_control": capture_control,
            "samples": [{
                "sample_id": args.sample_id or audio_path.stem,
                "audio": {"sha256": payload["meta"]["audio_sha256"]},
                "model_input": model_input,
                "feature_stft_contract": capture.feature_stft_contract,
                "tokens": token_ids, "transcript": hypothesis.text,
                "token_duration": payload["result"]["token_duration"],
                "token_frames": payload["result"]["timestamp"],
                "stages": capture.stages, "calls": capture.calls,
            }],
        }
        args.stage_output.parent.mkdir(parents=True, exist_ok=True)
        stage_json = json.dumps(stage_payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if args.stage_output.suffix == ".gz":
            with gzip.open(args.stage_output, "xt", encoding="utf-8") as file:
                file.write(stage_json)
        else:
            write_new_text(args.stage_output, stage_json)
        print(f"Captured {len(capture.stages)} stages: {args.stage_output}", flush=True)

    encoded = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        write_new_text(args.output, encoded)
    else:
        print(encoded)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
