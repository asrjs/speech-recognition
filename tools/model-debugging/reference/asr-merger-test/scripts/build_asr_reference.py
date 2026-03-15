#!/usr/bin/env python3
"""
Build a ground-truth reference from full-audio ASR (one pass, no chunking).
Outputs JSON with full_text, word-level timestamps, and sentences for use with
--ref-type asr_json in run_all_mergers.py. Sentences are split using an NLP
sentence boundary detection model (pysbd), not heuristic punctuation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Project root = parent of scripts/
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))

from audio import SAMPLE_RATE, load_audio_wav
from stream_sim import _timestamped_to_words


def _split_into_sentences(text: str, language: str = "en") -> list[str]:
    """Split text using NLP sentence boundary detection (pysbd). Returns list of sentence strings."""
    if not text or not text.strip():
        return []
    import pysbd
    segmenter = pysbd.Segmenter(language=language, clean=False)
    segments = segmenter.segment(text.strip())
    return [s.strip() for s in segments if s.strip()]


def run_full_audio_asr(
    audio_path: str | Path,
    model_dir: str | Path,
    model_name: str = "nemo-parakeet-tdt-0.6b-v2",
    sentence_language: str = "en",
) -> dict:
    """
    Run onnx-asr on the entire audio as one chunk. Returns dict with full_text,
    words (list of {text, start_time, end_time, confidence}), and sentences.
    Sentences are split using pysbd (NLP sentence boundary detection).
    """
    try:
        import onnx_asr
    except ImportError:
        raise ImportError("onnx-asr is required; pip install onnx-asr")

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = onnx_asr.load_model(model_name, path=model_dir)
    adapter = model.with_timestamps()

    waveform = load_audio_wav(audio_path)
    # Single chunk = full waveform
    results = adapter.recognize([waveform], sample_rate=SAMPLE_RATE)
    if isinstance(results, list):
        res = results[0] if results else None
    else:
        res = results
    if res is None:
        return {"full_text": "", "words": [], "sentences": []}

    words = _timestamped_to_words(
        res.text or "",
        getattr(res, "tokens", None),
        getattr(res, "timestamps", None),
        time_offset_sec=0.0,
    )
    full_text = (res.text or "").strip()
    sentences = _split_into_sentences(full_text, language=sentence_language)

    return {
        "full_text": full_text,
        "words": words,
        "sentences": sentences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-audio ASR and write reference JSON for run_all_mergers --ref-type asr_json.",
    )
    parser.add_argument("--audio", type=Path, required=True, help="Path to WAV or m4a.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Parakeet ONNX model directory.")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path (default: <audio>.ref.json).")
    parser.add_argument("--model-name", type=str, default="nemo-parakeet-tdt-0.6b-v2", help="Model name for onnx_asr.")
    parser.add_argument("--lang", type=str, default="en", help="Language code for sentence boundary detection (pysbd). Default: en.")
    args = parser.parse_args()

    import os
    _default_model = Path("N:/models/onnx/nemo/parakeet-tdt-0.6b-v2-onnx")
    if args.model_dir is None:
        args.model_dir = Path(os.environ.get("MODEL_PATH", _default_model))

    audio_path = args.audio
    if audio_path.suffix.lower() == ".m4a":
        from audio import m4a_to_wav
        audio_path = m4a_to_wav(audio_path)

    out_path = args.output or audio_path.with_suffix(".ref.json")
    out_path = Path(out_path)

    data = run_full_audio_asr(
        audio_path, args.model_dir,
        model_name=args.model_name,
        sentence_language=args.lang,
    )
    # Write JSON (sentences and full_text; words optional but useful for debugging)
    from json_float import dumps_transcript
    out_path.write_text(dumps_transcript(data), encoding="utf-8")
    print(f"Wrote {out_path} (full_text len={len(data['full_text'])}, sentences={len(data['sentences'])})")


if __name__ == "__main__":
    main()
