"""
Generate a unified benchmark report from PyTorch, ONNX, and JS result files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from benchmark_common import compute_metrics, load_english_normalizer

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPTS_DIR.parent


def _load_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _index_by_file(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for row in rows:
        key = Path(row["file"]).name
        out[key] = row
    return out


def _sum_float(rows: list[dict], key_a: str, key_b: str | None = None) -> float:
    total = 0.0
    for row in rows:
        if key_a in row:
            total += float(row[key_a] or 0.0)
        elif key_b and key_b in row:
            total += float(row[key_b] or 0.0)
    return total


def _model_metrics(name: str, rows: list[dict], normalizer) -> dict:
    references = [r.get("reference", "") for r in rows]
    predictions = [r.get("prediction", "") for r in rows]
    m = compute_metrics(references, predictions, normalizer)
    total_audio = _sum_float(rows, "audio_duration", "audioDuration")
    total_infer = _sum_float(rows, "inference_time", "inferenceTime")
    rtf = (total_infer / total_audio) if total_audio > 0 else None
    return {
        "name": name,
        "samples": len(rows),
        "wer": m["wer"],
        "cer": m["cer"],
        "rtf": rtf,
        "audio_duration_sec": total_audio,
        "inference_time_sec": total_infer,
    }


def build_report(
    pytorch_path: Path,
    onnx_path: Path,
    js_path: Path,
    out_md: Path,
    out_json: Path,
) -> dict:
    normalizer, normalizer_source = load_english_normalizer()

    pt_rows = _load_rows(pytorch_path)
    onnx_rows = _load_rows(onnx_path)
    js_rows = _load_rows(js_path)

    pt_by_file = _index_by_file(pt_rows)
    onnx_by_file = _index_by_file(onnx_rows)
    js_by_file = _index_by_file(js_rows)

    common_files = sorted(set(pt_by_file) & set(onnx_by_file) & set(js_by_file))
    if not common_files:
        raise RuntimeError("No overlapping files between pytorch/onnx/js result sets.")

    pt_common = [pt_by_file[f] for f in common_files]
    onnx_common = [onnx_by_file[f] for f in common_files]
    js_common = [js_by_file[f] for f in common_files]

    pt_metrics = _model_metrics("PyTorch", pt_common, normalizer)
    onnx_metrics = _model_metrics("ONNX Python", onnx_common, normalizer)
    js_metrics = _model_metrics("ONNX Node.js", js_common, normalizer)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "normalizer": normalizer_source,
        "common_sample_count": len(common_files),
        "files": common_files,
        "models": [pt_metrics, onnx_metrics, js_metrics],
    }

    table_rows = []
    for m in summary["models"]:
        rtf_str = f"{m['rtf']:.4f}x" if m["rtf"] is not None else "n/a"
        table_rows.append(
            f"| {m['name']} | {m['samples']} | {m['wer'] * 100:.2f}% | {m['cer'] * 100:.2f}% | {rtf_str} |"
        )

    markdown = "\n".join(
        [
            "# MedASR Benchmark Report",
            "",
            f"- Generated: {summary['generated_at']}",
            f"- Normalizer: {normalizer_source}",
            f"- Common samples across all three runs: {len(common_files)}",
            "",
            "| Model | Samples | WER | CER | RTF |",
            "|---|---:|---:|---:|---:|",
            *table_rows,
            "",
            "## Notes",
            "- WER/CER are recomputed from raw predictions with one shared normalizer.",
            "- RTF uses summed per-sample inference_time / audio_duration from each result file.",
        ]
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch-results",
        type=Path,
        default=SCRIPTS_DIR / "benchmark_pytorch_results.json",
    )
    parser.add_argument(
        "--onnx-results",
        type=Path,
        default=SCRIPTS_DIR / "benchmark_onnx_results.json",
    )
    parser.add_argument(
        "--js-results",
        type=Path,
        default=PROJECT_DIR / "metrics" / "benchmark_node_results.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_DIR / "benchmark_report.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_DIR / "metrics" / "benchmark_report_summary.json",
    )
    args = parser.parse_args()

    summary = build_report(
        pytorch_path=args.pytorch_results,
        onnx_path=args.onnx_results,
        js_path=args.js_results,
        out_md=args.out_md,
        out_json=args.out_json,
    )
    print(f"Report generated with {summary['common_sample_count']} common samples.")
    print(f"Markdown: {args.out_md}")
    print(f"Summary JSON: {args.out_json}")
