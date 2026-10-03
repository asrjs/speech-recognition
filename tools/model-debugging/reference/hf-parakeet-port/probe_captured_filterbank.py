"""Actual NeMo buffer versus built JS coefficients; diagnostic, not ASR acceptance."""
import argparse
import base64
import binascii
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from replay_packed_fft import ROOT, read_capture, read_stage
from artifact_outputs import require_new_outputs, write_new_text
from tensor_parity import compare_tensors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def captured_filterbank(sample):
    values = read_stage(sample, 'feature_stft/filterbank', 'float32')
    contract = sample['feature_stft_contract']
    if (values.shape != (1, 128, 257)
            or contract.get('filterbank_boundary') != 'model.preprocessor.featurizer.fb'
            or contract.get('filterbank_f32le_sha256') != hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()):
        raise ValueError('Actual filterbank shape/boundary/hash differs')
    return values


def decode_filterbank_bytes(encoded):
    """Canonical base64 little-endian float32, including signed zero."""
    if not isinstance(encoded, str):
        raise ValueError('Expected base64 filterbank bytes')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('Invalid filterbank encoding') from error
    if len(raw) != 128 * 257 * 4 or base64.b64encode(raw).decode('ascii') != encoded:
        raise ValueError('Filterbank byte count or canonical encoding differs')
    values = np.frombuffer(raw, dtype='<f4').reshape(1, 128, 257)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite filterbank bytes')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-sha256', 'weight-sha256', 'sample-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--triangle-rounding', action='store_true',
                        help='Diagnostic-only float32 triangle store before Slaney scaling')
    args = parser.parse_args()
    sources = [args.capture, Path(__file__), ROOT / 'dist/audio/js-mel.js',
               Path(__file__).with_name('replay_packed_fft.py'), ROOT / 'tools/tensor_parity.py']
    require_new_outputs([args.output], protected_inputs=sources,
                        protected_directories=[ROOT / 'models', ROOT / 'src', ROOT / 'dist'])
    hashes = {str(path.resolve()): digest(path) for path in sources}
    meta, sample, *_ = read_capture(args.capture, args.capture_sha256, args.sample_id)
    if (len(args.weight_sha256) != 64 or meta.get('weight_sha256') != args.weight_sha256
            or not isinstance(meta.get('preprocessor_config'), dict)):
        raise ValueError('Original weight/config identity missing or differs')
    official = captured_filterbank(sample)
    serialize = "const values=createMelFilterbank(128); const bytes=Buffer.alloc(values.length*4); for(let i=0;i<values.length;i++)bytes.writeFloatLE(values[i],i*4); console.log(bytes.toString('base64'));"
    probe = "import {createMelFilterbank} from './dist/audio/js-mel.js'; " + serialize
    if args.triangle_rounding:
        probe = """
import {readFileSync} from 'node:fs';
let source=readFileSync('./dist/audio/js-mel.js','utf8');
const before='Math.max(0, Math.min(downSlope, upSlope)) * enorm';
if(source.split(before).length!==2) throw new Error('Triangle source contract differs');
source=source.replace(before,'Math.fround(Math.max(0, Math.min(downSlope, upSlope))) * enorm');
const {createMelFilterbank}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
""" + serialize
    process = subprocess.run(['node', '--input-type=module', '-e', probe], cwd=ROOT,
                             text=True, capture_output=True, check=True, timeout=30)
    js = decode_filterbank_bytes(process.stdout.strip())
    metrics = compare_tensors(official, js, atol=0, rtol=0)
    unchanged = hashes == {str(path.resolve()): digest(path) for path in sources}
    if not unchanged:
        raise ValueError('Filterbank diagnostic sources changed')
    receipt = {'status': 'FILTERBANK_DIAGNOSTIC_MEASURED', 'sample_id': args.sample_id,
        'scope': 'Actual captured NeMo filterbank versus JS coefficients only; optional in-memory triangle-rounding intervention, no feature or ASR acceptance',
        'triangle_rounding_requested': args.triangle_rounding,
        'node_probe_sha256': hashlib.sha256(probe.encode('utf-8')).hexdigest(),
        'sources_sha256': hashes, 'inputs_unchanged': unchanged, 'identity': meta,
        'filterbank_contract': sample['feature_stft_contract'], 'comparison': metrics,
        'support_mismatch_count': int(np.count_nonzero((official != 0) != (js != 0))),
        'serialized_js_bit_mismatch_count': int(np.count_nonzero(official.view(np.uint32) != js.view(np.uint32))),
        'serialization_note': 'Canonical base64 f32le bytes preserve signed zero; numerical and bit equality are reported separately',
        'js_filterbank_f32le_sha256': hashlib.sha256(js.astype('<f4').tobytes()).hexdigest()}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': receipt['status'], 'comparison': metrics,
                      'serialized_js_bit_mismatch_count': receipt['serialized_js_bit_mismatch_count']}))


if __name__ == '__main__':
    main()
