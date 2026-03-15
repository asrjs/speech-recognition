from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPTS_DIR.parent
REFERENCE_MEDASR_DIR = TOOLS_DIR / "reference" / "medasrjs"
GITHUB_ROOT = SCRIPTS_DIR.parents[5]


def resolve_open_asr_leaderboard_dir(explicit_dir: str | None) -> str | None:
    if explicit_dir:
        candidate = Path(explicit_dir)
        if (candidate / "normalizer" / "normalizer.py").exists():
            return str(candidate)
        raise FileNotFoundError(
            f'open_asr_leaderboard normalizer was not found under "{candidate}".'
        )

    candidates = [
        Path(os.environ["OPEN_ASR_LEADERBOARD_DIR"])
        if os.environ.get("OPEN_ASR_LEADERBOARD_DIR")
        else None,
        REFERENCE_MEDASR_DIR / "open_asr_leaderboard",
        GITHUB_ROOT / "ysdede" / "medasr.js" / "open_asr_leaderboard",
    ]
    cwd = Path.cwd().resolve()
    candidates.extend(
        root / "ysdede" / "medasr.js" / "open_asr_leaderboard" for root in (cwd, *cwd.parents)
    )
    for candidate in candidates:
        if candidate and (candidate / "normalizer" / "normalizer.py").exists():
            return str(candidate)
    return None


def configure_normalizer_search_path(explicit_dir: str | None) -> None:
    resolved = resolve_open_asr_leaderboard_dir(explicit_dir)
    if resolved:
        os.environ["OPEN_ASR_LEADERBOARD_DIR"] = resolved
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--prediction-path", default="prediction")
    parser.add_argument("--reference-path", default="reference")
    parser.add_argument("--open-asr-leaderboard-dir")
    return parser.parse_args()


ARGS = parse_args()
configure_normalizer_search_path(ARGS.open_asr_leaderboard_dir)

if str(REFERENCE_MEDASR_DIR) not in sys.path:
    sys.path.insert(0, str(REFERENCE_MEDASR_DIR))

from benchmark_common import compute_metrics, load_english_normalizer as load_fallback_normalizer


def load_english_normalizer():
    resolved = os.environ.get("OPEN_ASR_LEADERBOARD_DIR")
    if resolved:
        try:
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            from normalizer.normalizer import EnglishTextNormalizer

            return EnglishTextNormalizer(), f"open_asr_leaderboard ({resolved})"
        except Exception:
            pass

    return load_fallback_normalizer()


def get_nested_value(value, dotted_path: str):
    current = value
    for segment in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def resolve_rows(document):
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and isinstance(document.get("rows"), list):
        return document["rows"]
    raise ValueError('Expected either a JSON array or an object with a top-level "rows" array.')


def normalize_row(row: dict, prediction_path: str, reference_path: str) -> dict | None:
    prediction = get_nested_value(row, prediction_path)
    reference = get_nested_value(row, reference_path)
    if not isinstance(prediction, str) or not isinstance(reference, str):
        return None

    normalized = dict(row)
    normalized["prediction"] = prediction
    normalized["reference"] = reference
    return normalized


def main(results_path: Path, out_summary: Path, prediction_path: str, reference_path: str):
    document = json.loads(results_path.read_text(encoding="utf-8"))
    rows = resolve_rows(document)
    normalized_rows = [
        row
        for row in (
            normalize_row(row, prediction_path=prediction_path, reference_path=reference_path)
            for row in rows
        )
        if row is not None
    ]

    references = [row.get("reference", "") for row in normalized_rows]
    predictions = [row.get("prediction", "") for row in normalized_rows]

    normalizer, source = load_english_normalizer()
    metrics = compute_metrics(references, predictions, normalizer)

    summary = {
        "results": str(results_path),
        "samples": len(normalized_rows),
        "prediction_path": prediction_path,
        "reference_path": reference_path,
        "wer": metrics["wer"],
        "cer": metrics["cer"],
        "normalizer": source,
    }

    for index, row in enumerate(normalized_rows):
        row["norm_prediction"] = metrics["norm_predictions"][index]
        row["norm_reference"] = metrics["norm_references"][index]

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"samples={summary['samples']}")
    print(f"wer={summary['wer'] * 100:.2f}%")
    print(f"cer={summary['cer'] * 100:.2f}%")
    print(f"normalizer={source}")
    print(f"summary={out_summary}")


if __name__ == "__main__":
    main(
        results_path=ARGS.results,
        out_summary=ARGS.out_summary,
        prediction_path=ARGS.prediction_path,
        reference_path=ARGS.reference_path,
    )
