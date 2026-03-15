"""
Shared helpers for MedASR benchmark scripts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

import jiwer

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TESTS_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

DEFAULT_DATASET_DIR = Path(
    os.environ.get(
        "PARROT_DATASET_DIR",
        "N:/github/ysdede/scribe-ds/data/PARROT_v1.0/07_audio/labels/_30s",
    )
)
DEFAULT_ONNX_PATH = PROJECT_DIR / "models" / "medasr" / "model.onnx"
DEFAULT_TOKENS_PATH = PROJECT_DIR / "models" / "medasr" / "tokens.txt"


def ctc_collapse(ids: Sequence[int], blank_id: int = 0) -> list[int]:
    """Greedy CTC collapse: remove repeats and blanks."""
    result: list[int] = []
    prev = None
    for tok in ids:
        if tok != prev and tok != blank_id:
            result.append(int(tok))
        prev = tok
    return result


def load_samples(dataset_dir: Path, num_samples: int | None) -> list[dict]:
    """Load PARROT dataset wav/json pairs."""
    json_files = sorted(dataset_dir.glob("*.json"))
    if num_samples and num_samples > 0:
        json_files = json_files[:num_samples]

    samples = []
    for json_path in json_files:
        wav_path = json_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            label = json.load(f)
        samples.append(
            {
                "file": wav_path.name,
                "wav_path": str(wav_path),
                "reference": label.get("transcription", ""),
            }
        )
    return samples


def load_english_normalizer() -> tuple[Callable[[str], str], str]:
    """
    Return (normalizer_callable, source_label).
    Prefers HF open_asr_leaderboard normalizer if available, otherwise jiwer fallback.
    """
    candidates = []
    env_dir = os.environ.get("OPEN_ASR_LEADERBOARD_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            TESTS_DIR / "open_asr_leaderboard",
            PROJECT_DIR / "open_asr_leaderboard",
            WORKSPACE_DIR / "open_asr_leaderboard",
        ]
    )

    for cand in candidates:
        normalizer_py = cand / "normalizer" / "normalizer.py"
        if not normalizer_py.exists():
            continue
        cand_str = str(cand)
        if cand_str not in sys.path:
            sys.path.insert(0, cand_str)
        try:
            from normalizer.normalizer import EnglishTextNormalizer  # type: ignore

            return EnglishTextNormalizer(), f"open_asr_leaderboard ({cand_str})"
        except Exception:
            continue

    fallback = jiwer.Compose(
        [
            jiwer.SubstituteWords({"PARAGRAPH": "", "NEWLINE": ""}),
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
        ]
    )
    return fallback, "jiwer fallback"


def clean_transcription(text: str, normalizer: Callable[[str], str]) -> str:
    text = text.replace("PARAGRAPH", " ").replace("NEWLINE", " ")
    return normalizer(text)


def compute_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    normalizer: Callable[[str], str],
) -> dict:
    norm_refs = [clean_transcription(r, normalizer) for r in references]
    norm_preds = [clean_transcription(p, normalizer) for p in predictions]
    return {
        "norm_references": norm_refs,
        "norm_predictions": norm_preds,
        "wer": float(jiwer.wer(norm_refs, norm_preds)),
        "cer": float(jiwer.cer(norm_refs, norm_preds)),
    }
