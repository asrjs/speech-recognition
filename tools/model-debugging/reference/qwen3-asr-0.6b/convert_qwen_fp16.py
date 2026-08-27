"""Convert official Qwen3-ASR ONNX graphs to fp16 and share decoder weights.

Does not change the explicit-KV contract. Int64 IDs stay int64. Float I/O
become float16 (`keep_io_types=False`) so the graph has fewer mixed Casts.
Large initializers are forced to float16. Native ORT greedy still matches
the JFK oracle. onnxruntime-web currently rejects these converted graphs
(SimplifiedLayerNormFusion + inserted Casts); WASM e2e uses sequential fp32.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import ssl
import traceback
from pathlib import Path

import numpy as np


AUDIO_TOKEN_ID = 151676
EOS_TOKEN_IDS = (151645, 151643)
ORACLE_JFK = (
    "And so, my fellow Americans, ask not what your country can do for you; "
    "ask what you can do for your country."
)
SHARED_DECODER_DATA = "decoder-fp16.onnx.data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(r"N:\models\Qwen3-ASR-0.6B"))
    parser.add_argument("--onnx-dir", type=Path, default=Path(r"N:\models\onnx\qwen3-asr-0.6b-official"))
    parser.add_argument("--audio", type=Path, default=Path(r"N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\jfk-short.wav"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument("--skip-greedy", action="store_true")
    return parser.parse_args()


def _disable_tls_verify() -> None:
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def try_step(name: str, fn) -> dict:
    try:
        details = fn()
        return {"name": name, "ok": True, **(details or {})}
    except Exception as error:  # noqa: BLE001
        return {
            "name": name,
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "traceback": traceback.format_exc()[-4000:],
        }


def pack_onnx(model, dest: Path, data_name: str) -> dict:
    import onnx

    dest.parent.mkdir(parents=True, exist_ok=True)
    data_path = dest.parent / data_name
    if dest.exists():
        dest.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(dest),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"Expected external data file {data_path}")
    return {
        "onnx_path": str(dest),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "external_data": str(data_path),
        "external_size_bytes": data_path.stat().st_size,
        "external_sha256": sha256_file(data_path),
    }


def retarget_external_location(src: Path, dest: Path, data_name: str) -> dict:
    import onnx
    from onnx.external_data_helper import uses_external_data

    model = onnx.load(str(src), load_external_data=False)
    for tensor in list(model.graph.initializer) + list(model.graph.sparse_initializer):
        if not uses_external_data(tensor):
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = data_name
    if dest.exists() and dest.resolve() != src.resolve():
        dest.unlink()
    onnx.save_model(model, str(dest), save_as_external_data=False)
    if src.resolve() != dest.resolve() and src.exists():
        src.unlink()
    data_path = dest.parent / data_name
    return {
        "onnx_path": str(dest),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "external_data": str(data_path),
        "external_size_bytes": data_path.stat().st_size if data_path.is_file() else 0,
        "external_sha256": sha256_file(data_path) if data_path.is_file() else None,
        "shared_data_name": data_name,
    }


def convert_graph(src: Path, dest: Path, data_name: str) -> dict:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(src), load_external_data=True)
    converted = convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
        force_fp16_initializers=True,
    )
    del model
    gc.collect()
    packed = pack_onnx(converted, dest, data_name)
    del converted
    gc.collect()
    packed["source"] = str(src)
    packed["source_size_bytes"] = src.stat().st_size
    return packed


def numpy_feed(session, name: str, array: np.ndarray) -> np.ndarray:
    info = next(item for item in session.get_inputs() if item.name == name)
    if "float16" in info.type:
        return array.astype(np.float16)
    if "int64" in info.type:
        return array.astype(np.int64)
    return array


def greedy_native(
    encoder_path: Path,
    prefill_path: Path,
    step_path: Path,
    model_dir: Path,
    audio: Path,
    max_new_tokens: int,
) -> dict:
    import onnxruntime as ort
    import torch
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import normalize_audios, parse_asr_output as official_parse

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    wrapper = Qwen3ASRModel.from_pretrained(
        str(model_dir.resolve()),
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
    )
    processor = wrapper.processor
    prompt = wrapper._build_text_prompt(context="", force_language=None)
    waveform = normalize_audios(str(audio.resolve()))[0]
    inputs = processor(text=[prompt], audio=[waveform], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    seq_len = int(input_ids.shape[1])
    audio_index = (input_ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=False).squeeze(-1)
    del wrapper
    gc.collect()

    sess_opt = ort.SessionOptions()
    sess_opt.enable_mem_pattern = True
    encoder = ort.InferenceSession(str(encoder_path), sess_opt, providers=["CPUExecutionProvider"])
    features = input_features[0].numpy() if input_features.ndim == 3 else input_features.numpy()
    audio_embeddings = encoder.run(None, {"input_features": features})[0]
    del encoder
    gc.collect()
    aligned = np.zeros((1, seq_len, audio_embeddings.shape[-1]), dtype=np.float32)
    token_count = int(audio_index.numel())
    aligned[0, audio_index.numpy()] = audio_embeddings[:token_count]
    position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

    prefill = ort.InferenceSession(str(prefill_path), sess_opt, providers=["CPUExecutionProvider"])
    logits, keys, values = prefill.run(
        None,
        {
            "input_ids": numpy_feed(prefill, "input_ids", input_ids.numpy()),
            "audio_embeddings": numpy_feed(prefill, "audio_embeddings", aligned),
            "position_ids": numpy_feed(prefill, "position_ids", position_ids),
        },
    )
    del prefill
    gc.collect()
    tokens: list[int] = []
    next_id = int(np.argmax(logits[0, -1]))
    seq = seq_len
    step = ort.InferenceSession(str(step_path), sess_opt, providers=["CPUExecutionProvider"])
    for _ in range(max_new_tokens):
        if next_id in EOS_TOKEN_IDS:
            break
        tokens.append(next_id)
        logits, keys, values = step.run(
            None,
            {
                "input_ids": numpy_feed(step, "input_ids", np.array([[next_id]], dtype=np.int64)),
                "position_ids": numpy_feed(step, "position_ids", np.array([[seq]], dtype=np.int64)),
                "past_keys": keys,
                "past_values": values,
            },
        )
        next_id = int(np.argmax(logits[0, -1]))
        seq += 1
    del step
    gc.collect()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    language, text = official_parse(raw)
    return {
        "token_count": len(tokens),
        "raw": raw,
        "language": language,
        "text": text,
        "matches_oracle": text == ORACLE_JFK,
        "encoder_tokens": int(audio_embeddings.shape[0]),
        "prompt_len": seq_len,
        "first_token": tokens[0] if tokens else None,
        "io_kept_float32": False,
    }


def main() -> None:
    args = parse_args()
    report: dict = {
        "family": "qwen3-asr-0.6b",
        "status": "fp16-pending",
        "onnx_dir": str(args.onnx_dir),
        "attempts": [],
    }

    encoder_src = args.onnx_dir / "audio-encoder-static-t1100.onnx"
    encoder_fp16 = args.onnx_dir / "audio-encoder-static-t1100-fp16.onnx"
    prefill_src = args.onnx_dir / "decoder-prefill.onnx"
    step_src = args.onnx_dir / "decoder-step.onnx"
    prefill_fp16 = args.onnx_dir / "decoder-prefill-fp16.onnx"
    step_fp16 = args.onnx_dir / "decoder-step-fp16.onnx"
    step_tmp = args.onnx_dir / "_decoder-step-fp16.onnx"

    if not args.skip_encoder:
        report["attempts"].append(
            try_step(
                "encoder_fp16",
                lambda: convert_graph(encoder_src, encoder_fp16, encoder_fp16.name + ".data"),
            )
        )

    if not args.skip_decoder:
        report["attempts"].append(
            try_step(
                "decoder_prefill_fp16",
                lambda: convert_graph(prefill_src, prefill_fp16, SHARED_DECODER_DATA),
            )
        )

        def convert_step_shared() -> dict:
            packed = convert_graph(step_src, step_tmp, step_tmp.name + ".data")
            prefill_data = args.onnx_dir / SHARED_DECODER_DATA
            step_data = step_tmp.parent / (step_tmp.name + ".data")
            same = (
                prefill_data.is_file()
                and step_data.is_file()
                and sha256_file(prefill_data) == sha256_file(step_data)
            )
            if not same:
                # Keep a dedicated file; still usable, just not shared on disk.
                final = pack_onnx(
                    __import__("onnx").load(str(step_tmp), load_external_data=True),
                    step_fp16,
                    step_fp16.name + ".data",
                )
                if step_tmp.exists():
                    step_tmp.unlink()
                return {**final, "shared_weights": False, "tmp": packed}
            if step_data.exists():
                step_data.unlink()
            retargeted = retarget_external_location(step_tmp, step_fp16, SHARED_DECODER_DATA)
            return {**retargeted, "shared_weights": True, "tmp_sha256": packed.get("external_sha256")}

        report["attempts"].append(try_step("decoder_step_fp16_shared", convert_step_shared))

    encoder_for_greedy = encoder_fp16 if encoder_fp16.is_file() else encoder_src
    if not args.skip_greedy and prefill_fp16.is_file() and step_fp16.is_file():
        report["attempts"].append(
            try_step(
                "native_ort_fp16_greedy_jfk",
                lambda: greedy_native(
                    encoder_for_greedy,
                    prefill_fp16,
                    step_fp16,
                    args.model_dir,
                    args.audio,
                    args.max_new_tokens,
                ),
            )
        )

    greedy = next((item for item in report["attempts"] if item.get("name") == "native_ort_fp16_greedy_jfk"), None)
    prefill_ok = any(item.get("ok") and item["name"] == "decoder_prefill_fp16" for item in report["attempts"])
    step_ok = any(item.get("ok") and item["name"] == "decoder_step_fp16_shared" for item in report["attempts"])
    if greedy and greedy.get("ok") and greedy.get("matches_oracle"):
        report["status"] = "fp16-oracle-match"
        report["failureClass"] = None
    elif prefill_ok and step_ok:
        report["status"] = "fp16-exported"
        report["failureClass"] = None if not greedy else "PREPROCESSING_MISMATCH"
        if greedy and not greedy.get("ok"):
            report["failureClass"] = "ORT_SESSION_FAILED"
    else:
        report["status"] = "fp16-blocked"
        report["failureClass"] = "EXPORT_BLOCKED"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failureClass": report.get("failureClass"), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    _disable_tls_verify()
    main()
