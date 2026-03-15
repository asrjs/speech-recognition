"""
Export MedASR assets and generate deterministic parity reference files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCTC, AutoProcessor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "models" / "medasr"
DEFAULT_REF_DIR = PROJECT_DIR / "tests" / "reference_medasr"


def ctc_collapse(ids: list[int], blank_id: int = 0) -> list[int]:
    out: list[int] = []
    prev = None
    for tok in ids:
        if tok != prev and tok != blank_id:
            out.append(int(tok))
        prev = tok
    return out


def detect_default_audio() -> Path:
    dataset_dir = Path(
        os.environ.get(
            "PARROT_DATASET_DIR",
            "N:/github/ysdede/scribe-ds/data/PARROT_v1.0/07_audio/labels/_30s",
        )
    )
    if dataset_dir.exists():
        wavs = sorted(dataset_dir.glob("*.wav"))
        if wavs:
            return wavs[0]

    fallback = PROJECT_DIR / "test-assets" / "jfk_short.wav"
    return fallback


def export_medasr(
    model_id: str,
    out_dir: Path,
    force_export: bool = False,
) -> tuple[AutoProcessor, AutoModelForCTC, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / "model.onnx"
    opset = 18

    print(f"Loading processor and model for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).eval()

    if force_export or not out_onnx.exists():
        print(f"Exporting ONNX to {out_onnx}...")

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_features, attention_mask):
                return self.m(input_features=input_features, attention_mask=attention_mask).logits

        wrapped = Wrapper(model)
        dummy_feats = torch.randn(1, 300, 128, dtype=torch.float32)
        dummy_mask = torch.ones(1, 300, dtype=torch.int32)

        torch.onnx.export(
            wrapped,
            (dummy_feats, dummy_mask),
            str(out_onnx),
            input_names=["input_features", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_features": {0: "batch", 1: "frames"},
                "attention_mask": {0: "batch", 1: "frames"},
                "logits": {0: "batch", 1: "frames"},
            },
            opset_version=opset,
            do_constant_folding=True,
        )
        print(f"Exported: {out_onnx}")
    else:
        print(f"ONNX already exists, skipping export: {out_onnx}")

    # Always refresh these metadata artifacts.
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    tokens_path = out_dir / "tokens.txt"
    with open(tokens_path, "w", encoding="utf-8") as f:
        for token, idx in sorted_vocab:
            f.write(f"{token} {idx}\n")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2, ensure_ascii=False)
    with open(out_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(processor.feature_extractor.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Exported tokens/config to {out_dir}")
    return processor, model, out_onnx


def _resolve_blank_id(processor: AutoProcessor) -> int:
    vocab = processor.tokenizer.get_vocab()
    if "<epsilon>" in vocab:
        return int(vocab["<epsilon>"])
    if "<blk>" in vocab:
        return int(vocab["<blk>"])
    if "<pad>" in vocab:
        return int(vocab["<pad>"])
    return 0


def _resolve_mask_dtype(session: ort.InferenceSession):
    input_types = {inp.name: inp.type for inp in session.get_inputs()}
    return np.int64 if input_types.get("attention_mask") == "tensor(int64)" else np.int32


def generate_reference(
    processor: AutoProcessor,
    model: AutoModelForCTC,
    onnx_path: Path,
    audio_path: Path,
    ref_dir: Path,
) -> None:
    print(f"Generating reference output for {audio_path}...")
    speech, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    inputs = processor(
        speech,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    feats = inputs.input_features
    mask = (
        inputs.attention_mask
        if "attention_mask" in inputs
        else torch.ones((1, feats.shape[1]), dtype=torch.int32)
    )

    with torch.inference_mode():
        pt_logits = model(input_features=feats, attention_mask=mask).logits.cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_name = session.get_outputs()[0].name
    mask_dtype = _resolve_mask_dtype(session)
    ort_logits = session.run(
        [output_name],
        {
            "input_features": feats.cpu().numpy().astype(np.float32),
            "attention_mask": mask.cpu().numpy().astype(mask_dtype),
        },
    )[0]

    max_abs = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"Logits max diff (PyTorch vs ONNX): {max_abs}")

    blank_id = _resolve_blank_id(processor)
    pt_pred_ids = np.argmax(pt_logits, axis=-1)[0].tolist()
    onnx_pred_ids = np.argmax(ort_logits, axis=-1)[0].tolist()

    pt_collapsed = ctc_collapse(pt_pred_ids, blank_id=blank_id)
    onnx_collapsed = ctc_collapse(onnx_pred_ids, blank_id=blank_id)

    text_pt = processor.decode(pt_collapsed, skip_special_tokens=True)
    text_onnx = processor.decode(onnx_collapsed, skip_special_tokens=True)
    text_batch_decode_raw = processor.batch_decode(np.array([onnx_pred_ids]))[0]

    output_data = {
        "audio_path": str(audio_path),
        "sample_rate": int(sr),
        "audio_length": int(len(speech)),
        "blank_id": int(blank_id),
        "features_shape": list(feats.shape),
        "attention_mask_shape": list(mask.shape),
        "logits_shape": list(ort_logits.shape),
        "pt_pred_ids": pt_pred_ids,
        "onnx_pred_ids": onnx_pred_ids,
        "pt_collapsed_ids": pt_collapsed,
        "onnx_collapsed_ids": onnx_collapsed,
        "text_pt_collapsed": text_pt,
        "text_onnx_collapsed": text_onnx,
        "text_onnx_batch_decode_raw": text_batch_decode_raw,
        "max_abs_pt_onnx_diff": max_abs,
    }

    ref_dir.mkdir(parents=True, exist_ok=True)
    with open(ref_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    with open(ref_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(feats.cpu().numpy().tolist()[0], f)
    with open(ref_dir / "attention_mask.json", "w", encoding="utf-8") as f:
        json.dump(mask.cpu().numpy().tolist()[0], f)
    with open(ref_dir / "logits.json", "w", encoding="utf-8") as f:
        json.dump(ort_logits.tolist()[0], f)
    print(f"Reference files saved to {ref_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/medasr")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audio", type=Path, default=detect_default_audio())
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    processor, model, onnx_path = export_medasr(
        model_id=args.model_id,
        out_dir=args.out_dir,
        force_export=args.force_export,
    )
    generate_reference(
        processor=processor,
        model=model,
        onnx_path=onnx_path,
        audio_path=args.audio,
        ref_dir=args.ref_dir,
    )
