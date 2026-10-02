"""Exact rational operation residuals from a pinned single-frame FFT receipt.

No FFT, weights or production inference. Claims only the recorded arithmetic.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def pair(value):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(item) not in (float, int) or not math.isfinite(item)
                   or f32(item) != item for item in value)):
        raise ValueError('Expected two exact finite f32 components')
    return [Fraction(item) for item in value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__), ROOT / 'tools/artifact_outputs.py']
    require_new_outputs([args.output], protected_inputs=[args.input, *sources],
                        protected_directories=[ROOT / 'src', ROOT / 'models'])
    digest = lambda content: hashlib.sha256(content).hexdigest()
    content = args.input.read_bytes()
    if digest(content) != args.expected_sha256:
        raise ValueError('Input receipt identity differs')
    hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    document = json.loads(content)
    if (document.get('status') != 'DIAGNOSTIC_ONLY'
            or document.get('selected_real512_capture_bits_exact') is not True):
        raise ValueError('Expected selected-frame oracle control receipt')
    results = {}
    for key, row in document['real_half_control'].items():
        index = int(key)
        if not 1 <= index < 256 or str(index) != key:
            raise ValueError('Invalid FFT512 paired bin')
        cosine = Fraction(f32(math.cos(-2 * math.pi * index / 512)))
        sine = Fraction(f32(math.sin(-2 * math.pi * index / 512)))
        target = pair(row['oracle'])
        for name, result_name in [('packed', 'packed_recombined'), ('real_half', 'half_recombined')]:
            even = pair(row[name + '_even'])
            odd = pair(row[name + '_odd'])
            exact_product = [odd[0] * cosine - odd[1] * sine,
                             odd[0] * sine + odd[1] * cosine]
            rotated = [Fraction(f32(float(value))) for value in exact_product]
            exact_sum = [left + right for left, right in zip(even, rotated)]
            rounded_sum = [Fraction(f32(float(value))) for value in exact_sum]
            if rounded_sum != pair(row[result_name]):
                raise ValueError('Recreated operation result differs from receipt')
            required = [value - term for value, term in zip(target, even)]
            results[f'{key}/{name}'] = {
                'product_round_error_exact': [str(a-b) for a,b in zip(rotated, exact_product)],
                'final_add_round_error_exact': [str(a-b) for a,b in zip(rounded_sum, exact_sum)],
                'required_oracle_rotated_exact': [str(value) for value in required],
                'required_oracle_rotated_f32_exact': [Fraction(f32(float(value))) == value
                                                     for value in required],
            }
    if not results:
        raise ValueError('Empty operation control')
    if (args.input.read_bytes() != content
            or hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}):
        raise ValueError('Inputs or sources changed during diagnostic')
    receipt = {'status': 'DIAGNOSTIC_EXACT_ARITHMETIC_ONLY',
               'input_sha256': args.expected_sha256, 'sources_sha256': hashes,
               'input_canonical_json_sha256': digest(json.dumps(document, sort_keys=True,
                   separators=(',', ':'), allow_nan=False).encode('utf-8')),
               'results': results,
               'scope': 'Exact recorded f32 arithmetic, not MKL internals or feature/model parity'}
    write_new_text(args.output, json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
