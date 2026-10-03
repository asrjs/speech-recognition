"""Frozen FFT512 -> existing JS diagnostic features with a hash-bound NumPy bank.

No librosa/Torch import, model loading, production modification or browser claim.
Negative candidate comparisons are evidence, not accepted model ports.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from replay_packed_fft import (ROOT, digest, read_capture, read_stage, build_probe,
                              write_probe, validate_probe_output, encode_complex_boundary)
from diagnostic_mel_bank import slaney_bank
from tensor_parity import compare_tensors
from artifact_outputs import require_new_outputs, write_new_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--expected-capture-sha256', required=True)
    parser.add_argument('--expected-bank-sha256', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-output', type=Path, required=True)
    args = parser.parse_args()
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        'replay_packed_fft.py', 'diagnostic_mel_bank.py', 'compare_js_feature_boundaries.py',
        'complex_intervention.py', 'normalization_reduction.py', 'normalization-reduction.mjs',
        'packed-f32-source.mjs')], ROOT / 'dist/audio/js-mel.js',
        ROOT / 'tools/tensor_parity.py', ROOT / 'tools/artifact_outputs.py']
    require_new_outputs([args.output, args.probe_output], protected_inputs=[args.capture, *sources],
                        protected_directories=[ROOT / 'src', ROOT / 'models'])
    hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    bank = slaney_bank(sample_rate=16000, n_fft=512, n_mels=128,
                       expected_sha256=args.expected_bank_sha256)
    _, sample, pcm, oracle, frames, reference, valid = read_capture(
        args.capture, args.expected_capture_sha256, args.sample_id)
    graph = helper.make_graph([
        helper.make_node('DFT', ['frames'], ['spectrum'], axis=2, inverse=0, onesided=1)
    ], 'frozen-real512-dft-control', [
        helper.make_tensor_value_info('frames', TensorProto.FLOAT, [1, 'frames', 512, 1])
    ], [helper.make_tensor_value_info('spectrum', TensorProto.FLOAT, [1, 'frames', 257, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    graph_bytes = model.SerializeToString()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
    values = session.run(['spectrum'], {'frames': frames[..., None]})[0]
    if (values.shape != (1, frames.shape[1], 257, 2) or values.dtype != np.float32
            or not np.isfinite(values).all()):
        raise ValueError('ORT DFT output contract differs')
    candidate = np.empty(values.shape[:-1], dtype=np.complex64)
    candidate.real, candidate.imag = values[..., 0], values[..., 1]
    candidate = candidate.transpose(0, 2, 1)
    spectrum = compare_tensors(oracle.view(np.float32), candidate.copy().view(np.float32),
                               atol=1e-5, rtol=1e-4)
    spectra = {'official_complex_control': oracle, 'ort_dft_candidate': candidate}
    probe = build_probe()
    write_probe(args.probe_output, probe)
    payload = {'pcm': pcm.ravel().tolist(), 'bins': 128, 'referenceRaw': [0] * reference.size,
        'referenceWindow': read_stage(sample, 'feature_stft/window', 'float32').tolist(),
        'referenceFilterbank': bank.ravel().tolist(), 'physicalFrames': oracle.shape[2],
        'validLength': valid, 'coordinate': None,
        'spectra': {name: encode_complex_boundary(value) for name, value in spectra.items()}}
    process = subprocess.run(['node', '--input-type=module', '-e', probe], cwd=ROOT,
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, timeout=60)
    if process.returncode:
        raise RuntimeError('Node feature probe failed: ' + process.stderr[-1500:])
    features = validate_probe_output(json.loads(process.stdout), set(spectra), reference, valid, None)
    if not features['official_complex_control']['allclose']:
        raise ValueError('Official-complex downstream control failed')
    if hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}:
        raise ValueError('Diagnostic sources changed during execution')
    if digest(args.capture.read_bytes()) != args.expected_capture_sha256:
        raise ValueError('Capture changed during execution')
    receipt = {'status': 'DIAGNOSTIC_FROZEN_DFT_FEATURES', 'sample_id': args.sample_id,
        'capture_sha256': args.expected_capture_sha256, 'bank_sha256': args.expected_bank_sha256,
        'bank_recipe': {'sample_rate': 16000, 'n_fft': 512, 'n_mels': 128,
                        'fmin': 0, 'fmax': 8000, 'norm': 'slaney', 'dtype': 'float32'},
        'sources_sha256': hashes, 'frames_sha256_f32le': digest(frames.astype('<f4').tobytes()),
        'pcm_sha256_f32le': digest(pcm.astype('<f4').tobytes()), 'graph_sha256': digest(graph_bytes),
        'window_sha256_f32le': digest(read_stage(sample, 'feature_stft/window', 'float32').astype('<f4').tobytes()),
        'oracle_real_sha256_f32le': digest(oracle.real.astype('<f4').tobytes()),
        'oracle_imag_sha256_f32le': digest(oracle.imag.astype('<f4').tobytes()),
        'physical_frames': oracle.shape[2], 'valid_frames': valid,
        'frame_reconstruction': 'Empirical centered zero-padding with captured window, not independently captured FFT-input frames',
        'node': subprocess.check_output(['node', '--version'], text=True).strip(),
        'probe_sha256': digest(probe.encode()), 'providers': session.get_providers(),
        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
        'spectrum_comparison': spectrum, 'feature_comparisons': features,
        'scope': 'Native CPU DFT on reconstructed captured frames + existing diagnostic JS mel/normalization; not production feature, encoder, token/text, browser or performance acceptance'}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
