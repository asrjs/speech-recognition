from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from benchmark_common import compute_metrics, load_english_normalizer  # noqa: E402,F401

print('ok')
