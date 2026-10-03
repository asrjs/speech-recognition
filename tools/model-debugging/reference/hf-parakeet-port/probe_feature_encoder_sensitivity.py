"""Production JS feature -> native encoder sensitivity; never replaces feature gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text
from tensor_parity import compare_tensors
from repack_onnx_external_data import range_hash
from replay_packed_fft import read_capture, read_stage

NODE = r'''
import { JSMelProcessor } from './dist/audio/js-mel.js';
let text = ''; for await (const chunk of process.stdin) text += chunk;
const input = JSON.parse(text);
const result = new JSMelProcessor({nMels:128, validLengthMode:'onnx',
  normalization:'per_feature'}).process(Float32Array.from(input));
console.log(JSON.stringify({features:Array.from(result.features),
  frames:result.frameCount, length:result.length}));
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'encoder', 'encoder-data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-sha256', 'encoder-sha256', 'encoder-data-sha256', 'sample-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    inputs = [args.capture, args.encoder, args.encoder_data, Path(__file__),
              ROOT / 'dist/audio/js-mel.js', Path(__file__).with_name('replay_packed_fft.py'),
              ROOT / 'tools/tensor_parity.py', ROOT / 'tools/artifact_outputs.py',
              ROOT / 'tools/repack_onnx_external_data.py']
    require_new_outputs([args.output], protected_inputs=inputs,
                        protected_directories=[ROOT / 'models', ROOT / 'src', ROOT / 'dist'])
    hashes = {str(path.resolve()): range_hash(path) for path in inputs}
    for path, expected in ((args.capture, args.capture_sha256),
                           (args.encoder, args.encoder_sha256),
                           (args.encoder_data, args.encoder_data_sha256)):
        if len(expected) != 64 or hashes[str(path.resolve())] != expected:
            raise ValueError('Pinned source hash differs: ' + str(path))
    if args.encoder_data.resolve() != Path(str(args.encoder.resolve()) + '.data'):
        raise ValueError('Expected own packed colocated encoder .data')
    meta, sample, pcm, _, _, features, valid = read_capture(
        args.capture, args.capture_sha256, args.sample_id)
    observed = subprocess.run(['node', '--input-type=module', '-e', NODE], cwd=ROOT,
        input=json.dumps(pcm[0].tolist()), text=True, encoding='utf-8',
        capture_output=True, check=True, timeout=60)
    js = json.loads(observed.stdout)
    if js['frames'] != features.shape[2] or js['length'] != valid:
        raise ValueError('Production feature frame/valid length differs')
    values = np.asarray(js['features'], dtype=np.float32).reshape(features.shape)
    feature_metric = compare_tensors(features, values, atol=1e-5, rtol=1e-4)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.encoder), sess_options=options,
                                 providers=['CPUExecutionProvider'])
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Unexpected actual provider')
    length = np.array([valid], dtype=np.int64)
    def encode(value):
        return dict(zip([item.name for item in session.get_outputs()],
            session.run(None, {'audio_signal': value, 'length': length})))
    baseline = encode(features)
    candidate = encode(values)
    encoder_reference = read_stage(sample, 'encoder', 'float32')
    length_reference = read_stage(sample, 'encoder_length', 'int64')
    comparisons = {'features': feature_metric,
        'official_features_encoder_vs_oracle': compare_tensors(encoder_reference, baseline['outputs'], atol=1e-5, rtol=1e-4),
        'production_features_encoder_vs_oracle': compare_tensors(encoder_reference, candidate['outputs'], atol=1e-5, rtol=1e-4),
        'production_vs_official_features_same_encoder': compare_tensors(baseline['outputs'], candidate['outputs'], atol=1e-5, rtol=1e-4),
        'baseline_length': compare_tensors(length_reference, baseline['encoded_lengths'], atol=0, rtol=0),
        'candidate_length': compare_tensors(length_reference, candidate['encoded_lengths'], atol=0, rtol=0)}
    if hashes != {str(path.resolve()): range_hash(path) for path in inputs}:
        raise ValueError('Inputs changed during sensitivity control')
    baseline_pass = all(comparisons[key]['allclose'] for key in
                        ('official_features_encoder_vs_oracle', 'baseline_length'))
    receipt = {'status': 'SENSITIVITY_MEASURED' if baseline_pass else 'BASELINE_FAILED',
        'scope': 'Unmodified built JS processor and own packed fp32 encoder, native CPU only; feature gate unchanged, decoder/browser/full ASR untested',
        'sample_id': args.sample_id, 'model_identity': meta,
        'sources_sha256': hashes, 'probe_sha256': hashlib.sha256(NODE.encode()).hexdigest(),
        'ort_version': ort.__version__, 'providers': session.get_providers(),
        'cpu_threads': 4, 'comparisons': comparisons, 'inputs_unchanged': True}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': receipt['status'], 'comparisons': comparisons}))
    if not baseline_pass:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
