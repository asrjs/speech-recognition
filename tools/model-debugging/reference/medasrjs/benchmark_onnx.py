"""
Benchmark google/medasr (ONNX) on local PARROT v1.0 WAV+JSON dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from benchmark_common import (
    DEFAULT_DATASET_DIR,
    DEFAULT_ONNX_PATH,
    compute_metrics,
    ctc_collapse,
    load_english_normalizer,
    load_samples,
)


def _resolve_mask_dtype(session: ort.InferenceSession) -> np.dtype:
    input_types = {inp.name: inp.type for inp in session.get_inputs()}
    mask_type = input_types.get("attention_mask", "tensor(int32)")
    return np.int64 if mask_type == "tensor(int64)" else np.int32

def _resolve_feats_dtype(session: ort.InferenceSession) -> np.dtype:
    input_types = {inp.name: inp.type for inp in session.get_inputs()}
    feats_type = input_types.get("input_features", "tensor(float)")
    return np.float16 if feats_type == "tensor(float16)" else np.float32


def benchmark_onnx(
    dataset_dir: Path,
    onnx_path: Path,
    model_id: str,
    num_samples: int,
    out_json: Path,
    summary_json: Path,
) -> dict:
    normalizer, normalizer_source = load_english_normalizer()
    print(f"Text normalizer: {normalizer_source}")

    print(f"Loading {model_id} processor and ONNX model...")
    processor = AutoProcessor.from_pretrained(model_id)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_name = sess.get_outputs()[0].name
    mask_dtype = _resolve_mask_dtype(sess)
    feats_dtype = _resolve_feats_dtype(sess)

    print(f"Loading samples from {dataset_dir}...")
    samples = load_samples(dataset_dir, num_samples)
    if not samples:
        raise RuntimeError(f"No wav/json samples found in {dataset_dir}")
    print(f"Benchmarking {len(samples)} samples...")

    predictions = []
    references = []
    per_sample = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    for s in tqdm(samples):
        speech, sr = sf.read(s["wav_path"])
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if sr != 16000:
            import librosa

            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)

        duration = len(speech) / 16000.0
        total_audio_duration += duration

        inputs = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        feats = inputs.input_features.numpy().astype(feats_dtype, copy=False)
        if "attention_mask" in inputs:
            mask = inputs.attention_mask.numpy().astype(mask_dtype, copy=False)
        else:
            mask = np.ones((1, feats.shape[1]), dtype=mask_dtype)

        t0 = time.perf_counter()
        ort_logits = sess.run(
            [output_name],
            {
                "input_features": feats,
                "attention_mask": mask,
            },
        )[0]
        pred_ids = np.argmax(ort_logits, axis=-1)[0].tolist()
        collapsed_ids = ctc_collapse(pred_ids, blank_id=0)
        text = processor.decode(collapsed_ids, skip_special_tokens=True)
        elapsed = time.perf_counter() - t0
        total_inference_time += elapsed

        predictions.append(text)
        references.append(s["reference"])
        per_sample.append(
            {
                "file": s["file"],
                "prediction": text,
                "reference": s["reference"],
                "audio_duration": duration,
                "inference_time": elapsed,
            }
        )

    metrics = compute_metrics(references, predictions, normalizer)
    wer = metrics["wer"]
    cer = metrics["cer"]
    rtf = total_inference_time / total_audio_duration

    for i, row in enumerate(per_sample):
        row["norm_prediction"] = metrics["norm_predictions"][i]
        row["norm_reference"] = metrics["norm_references"][i]

    summary = {
        "model": model_id,
        "backend": "onnx-cpu",
        "samples": len(per_sample),
        "audio_duration_sec": total_audio_duration,
        "inference_time_sec": total_inference_time,
        "rtf": rtf,
        "wer": wer,
        "cer": cer,
        "normalizer": normalizer_source,
        "dataset_dir": str(dataset_dir),
        "onnx_path": str(onnx_path),
        "attention_mask_dtype": str(mask_dtype),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, indent=2, ensure_ascii=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n--- Benchmark Results ---")
    print(f"Model: {model_id} (ONNX CPU)")
    print(f"Samples: {len(per_sample)}")
    print(f"Audio Duration: {total_audio_duration:.2f} s")
    print(f"Inference Time: {total_inference_time:.2f} s")
    print(f"RTF: {rtf:.4f}x")
    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")
    print(f"Per-sample results saved to {out_json}")
    print(f"Summary saved to {summary_json}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-id", default="google/medasr")
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_onnx_results.json",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_onnx_summary.json",
    )
    args = parser.parse_args()
    benchmark_onnx(
        dataset_dir=args.dataset_dir,
        onnx_path=args.onnx_path,
        model_id=args.model_id,
        num_samples=args.samples,
        out_json=args.out_json,
        summary_json=args.summary_json,
    )
