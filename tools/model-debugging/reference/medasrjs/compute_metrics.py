from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from benchmark_common import compute_metrics, load_english_normalizer


def main(results_path: Path, out_summary: Path, backend: str | None = None):
    rows = json.loads(results_path.read_text(encoding='utf-8'))
    references = [r.get('reference', '') for r in rows]
    predictions = [r.get('prediction', '') for r in rows]

    normalizer, source = load_english_normalizer()
    m = compute_metrics(references, predictions, normalizer)

    total_audio = sum(float(r.get('audio_duration', 0.0) or 0.0) for r in rows)
    total_infer = sum(float(r.get('inference_time', 0.0) or 0.0) for r in rows)
    rtf = (total_infer / total_audio) if total_audio > 0 else None

    summary = {
        'backend': backend or 'onnx-node',
        'samples': len(rows),
        'audio_duration_sec': total_audio,
        'inference_time_sec': total_infer,
        'rtf': rtf,
        'wer': m['wer'],
        'cer': m['cer'],
        'normalizer': source,
    }

    for i, row in enumerate(rows):
        row['norm_prediction'] = m['norm_predictions'][i]
        row['norm_reference'] = m['norm_references'][i]

    results_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"samples={summary['samples']}")
    print(f"wer={summary['wer'] * 100:.2f}%")
    print(f"cer={summary['cer'] * 100:.2f}%")
    print(f"rtf={summary['rtf']:.4f}x" if summary['rtf'] is not None else 'rtf=n/a')
    print(f"normalizer={source}")
    print(f"summary={out_summary}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out-summary', type=Path, required=True)
    parser.add_argument('--backend', default='onnx-node')
    args = parser.parse_args()

    main(args.results, args.out_summary, backend=args.backend)
