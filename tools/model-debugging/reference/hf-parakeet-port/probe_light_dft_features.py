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

DFT_TYPES = {'fp32': (np.float32, TensorProto.FLOAT), 'fp64': (np.float64, TensorProto.DOUBLE)}


def pocketfft_control(frames):
    """Installed native SciPy control, not a portable/browser implementation."""
    if (frames.ndim != 3 or frames.shape[0] != 1 or frames.shape[1] < 1
            or frames.shape[2] != 512 or frames.dtype != np.float32
            or not np.isfinite(frames).all()):
        raise ValueError('PocketFFT requires finite fp32 [1, frames, 512] input')
    import scipy
    from scipy.fft._pocketfft import basic, helper as pocket_helper
    paths = [Path(module.__file__) for module in (basic, pocket_helper, basic.pfft)]
    hashes = {str(path): digest(path.read_bytes()) for path in paths}
    values = basic.rfft(frames, axis=-1, norm='backward', workers=1)
    if (values.shape != (1, frames.shape[1], 257) or values.dtype != np.complex64
            or not np.isfinite(values).all()):
        raise ValueError('PocketFFT output contract differs')
    if hashes != {str(path): digest(path.read_bytes()) for path in paths}:
        raise ValueError('PocketFFT sources changed during execution')
    return values.transpose(0, 2, 1), {
        'scipy': scipy.__version__, 'sources_sha256': hashes,
        'entrypoint': 'scipy.fft._pocketfft.basic.rfft',
        'axis': -1, 'norm': 'backward', 'workers': 1,
        'scope': 'Installed native binding only; no browser/compiler equivalence claim'}


def build_dft_graph(precision):
    if precision not in DFT_TYPES:
        raise ValueError('Unsupported diagnostic DFT precision')
    _, tensor_type = DFT_TYPES[precision]
    graph = helper.make_graph([
        helper.make_node('DFT', ['frames'], ['spectrum'], axis=2, inverse=0, onesided=1)
    ], 'frozen-real512-dft-control', [
        helper.make_tensor_value_info('frames', tensor_type, [1, 'frames', 512, 1])
    ], [helper.make_tensor_value_info('spectrum', tensor_type, [1, 'frames', 257, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def build_power_probe(power):
    probe = build_probe()
    if power == 'fp32-rounded':
        return probe
    if power != 'library':
        raise ValueError('Unsupported injected-spectrum power arithmetic')
    for before, after in (
        ("referenceComplexPower !== 'fp32-rounded'", "referenceComplexPower !== 'library'"),
        ("referenceComplexPower: 'fp32-rounded'", "referenceComplexPower: 'library'"),
    ):
        if probe.count(before) != 1:
            raise ValueError('Re-inspect named spectrum probe power contract')
        probe = probe.replace(before, after)
    if "'official-complex-rounded'" not in probe:
        raise ValueError('Named spectrum probe is missing')
    return probe.replace("'official-complex-rounded'", "'official-complex-library'")


def checked_coordinate(coordinate, mel_count, valid_frames):
    if coordinate is None:
        return None
    if (len(coordinate) != 2 or any(type(value) is not int for value in coordinate)
            or not 0 <= coordinate[0] < mel_count or not 0 <= coordinate[1] < valid_frames):
        raise ValueError('Diagnostic coordinate is outside the valid feature prefix')
    return list(coordinate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--expected-capture-sha256', required=True)
    parser.add_argument('--expected-bank-sha256', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-output', type=Path, required=True)
    parser.add_argument('--power', choices=['fp32-rounded', 'library'], default='fp32-rounded')
    parser.add_argument('--coordinate', type=int, nargs=2, metavar=('MEL', 'FRAME'))
    parser.add_argument('--dft-precision', choices=list(DFT_TYPES), default='fp32')
    parser.add_argument('--fft-engine', choices=['ort-dft', 'scipy-pocketfft', 'pffft-native', 'torch-rfft'], default='ort-dft')
    parser.add_argument('--pffft-library', type=Path)
    parser.add_argument('--expected-pffft-sha256')
    args = parser.parse_args()
    if args.fft_engine == 'scipy-pocketfft' and args.dft_precision != 'fp32':
        parser.error('PocketFFT control supports fp32 only')
    if args.fft_engine == 'torch-rfft' and args.dft_precision != 'fp32':
        parser.error('Torch oracle reproduction supports fp32 only')
    if args.fft_engine == 'pffft-native' and (args.dft_precision != 'fp32'
            or args.pffft_library is None or args.expected_pffft_sha256 is None):
        parser.error('PFFFT requires fp32, explicit trusted library and expected SHA256')
    if args.fft_engine != 'pffft-native' and (args.pffft_library is not None
                                            or args.expected_pffft_sha256 is not None):
        parser.error('PFFFT options require pffft-native engine')
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        'replay_packed_fft.py', 'diagnostic_mel_bank.py', 'compare_js_feature_boundaries.py',
        'complex_intervention.py', 'normalization_reduction.py', 'normalization-reduction.mjs',
        'packed-f32-source.mjs')], ROOT / 'dist/audio/js-mel.js',
        ROOT / 'tools/tensor_parity.py', ROOT / 'tools/artifact_outputs.py']
    if args.fft_engine == 'pffft-native':
        sources.append(Path(__file__).with_name('pffft_native_control.py'))
    protected = [args.capture, *sources]
    if args.pffft_library is not None:
        protected.append(args.pffft_library)
    require_new_outputs([args.output, args.probe_output], protected_inputs=protected,
                        protected_directories=[ROOT / 'src', ROOT / 'models'])
    hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    bank = slaney_bank(sample_rate=16000, n_fft=512, n_mels=128,
                       expected_sha256=args.expected_bank_sha256)
    _, sample, pcm, oracle, frames, reference, valid = read_capture(
        args.capture, args.expected_capture_sha256, args.sample_id)
    coordinate = checked_coordinate(args.coordinate, bank.shape[0], valid)
    graph_bytes = build_dft_graph(args.dft_precision) if args.fft_engine == 'ort-dft' else None
    dft_dtype, _ = DFT_TYPES[args.dft_precision]
    dft_frames = frames.astype(dft_dtype, copy=False)
    engine_metadata = None
    session = None
    if args.fft_engine == 'torch-rfft':
        import torch
        torch.set_num_threads(4)
        candidate = torch.fft.rfft(torch.from_numpy(dft_frames), dim=-1,
                                    norm='backward').numpy().transpose(0, 2, 1).copy()
        if (candidate.shape != oracle.shape or candidate.dtype != np.complex64
                or not np.isfinite(candidate).all()):
            raise ValueError('Torch real FFT output contract differs')
        engine_metadata = {'torch': torch.__version__, 'cpu_threads': torch.get_num_threads(),
                           'build_config': torch.__config__.show(),
                           'mkl_available': torch.backends.mkl.is_available(),
                           'scope': 'Native oracle reproduction, not a portable implementation'}
        candidate_name = 'torch_rfft_control'
    elif args.fft_engine == 'pffft-native':
        from pffft_native_control import pffft_control
        candidate, engine_metadata = pffft_control(dft_frames, args.pffft_library,
                                                  args.expected_pffft_sha256)
        candidate_name = 'pffft_candidate'
    elif args.fft_engine == 'scipy-pocketfft':
        candidate, engine_metadata = pocketfft_control(dft_frames)
        candidate_name = 'pocketfft_candidate'
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
        values = session.run(['spectrum'], {'frames': dft_frames[..., None]})[0]
        if (values.shape != (1, frames.shape[1], 257, 2) or values.dtype != dft_dtype
                or not np.isfinite(values).all()):
            raise ValueError('ORT DFT output contract differs')
        candidate = np.empty(values.shape[:-1], dtype=np.complex64)
        candidate.real, candidate.imag = values[..., 0], values[..., 1]
        candidate = candidate.transpose(0, 2, 1)
        candidate_name = 'ort_dft_candidate'
    spectrum = compare_tensors(oracle.view(np.float32), candidate.copy().view(np.float32),
                               atol=1e-5, rtol=1e-4)
    spectra = {'official_complex_control': oracle, candidate_name: candidate}
    probe = build_power_probe(args.power)
    write_probe(args.probe_output, probe)
    payload = {'pcm': pcm.ravel().tolist(), 'bins': 128, 'referenceRaw': [0] * reference.size,
        'referenceWindow': read_stage(sample, 'feature_stft/window', 'float32').tolist(),
        'referenceFilterbank': bank.ravel().tolist(), 'physicalFrames': oracle.shape[2],
        'validLength': valid, 'coordinate': coordinate,
        'spectra': {name: encode_complex_boundary(value) for name, value in spectra.items()}}
    process = subprocess.run(['node', '--input-type=module', '-e', probe], cwd=ROOT,
        input=json.dumps(payload, allow_nan=False), text=True, capture_output=True, timeout=60)
    if process.returncode:
        raise RuntimeError('Node feature probe failed: ' + process.stderr[-1500:])
    features = validate_probe_output(json.loads(process.stdout), set(spectra), reference, valid, coordinate)
    if not features['official_complex_control']['allclose']:
        raise ValueError('Official-complex downstream control failed')
    if hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}:
        raise ValueError('Diagnostic sources changed during execution')
    if digest(args.capture.read_bytes()) != args.expected_capture_sha256:
        raise ValueError('Capture changed during execution')
    receipt = {'status': 'DIAGNOSTIC_FROZEN_DFT_FEATURES', 'sample_id': args.sample_id,
        'injected_spectrum_power': args.power,
        'fft_engine': args.fft_engine, 'engine_metadata': engine_metadata,
        'dft_precision': args.dft_precision, 'candidate_boundary_dtype': 'complex64',
        'dft_input_sha256': digest(dft_frames.astype(np.dtype(dft_dtype).newbyteorder('<')).tobytes()),
        'coordinate': coordinate,
        'capture_sha256': args.expected_capture_sha256, 'bank_sha256': args.expected_bank_sha256,
        'bank_recipe': {'sample_rate': 16000, 'n_fft': 512, 'n_mels': 128,
                        'fmin': 0, 'fmax': 8000, 'norm': 'slaney', 'dtype': 'float32'},
        'sources_sha256': hashes, 'frames_sha256_f32le': digest(frames.astype('<f4').tobytes()),
        'pcm_sha256_f32le': digest(pcm.astype('<f4').tobytes()),
        'graph_sha256': digest(graph_bytes) if graph_bytes is not None else None,
        'window_sha256_f32le': digest(read_stage(sample, 'feature_stft/window', 'float32').astype('<f4').tobytes()),
        'oracle_real_sha256_f32le': digest(oracle.real.astype('<f4').tobytes()),
        'oracle_imag_sha256_f32le': digest(oracle.imag.astype('<f4').tobytes()),
        'physical_frames': oracle.shape[2], 'valid_frames': valid,
        'frame_reconstruction': 'Empirical centered zero-padding with captured window, not independently captured FFT-input frames',
        'node': subprocess.check_output(['node', '--version'], text=True).strip(),
        'probe_sha256': digest(probe.encode()),
        'providers': session.get_providers() if session is not None else None,
        'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
        'spectrum_comparison': spectrum, 'feature_comparisons': features,
        'spectrum_capture_bytes_exact': candidate.tobytes() == oracle.tobytes(),
        'scope': 'Native CPU DFT on reconstructed captured frames + existing diagnostic JS mel/normalization; not production feature, encoder, token/text, browser or performance acceptance'}
    if coordinate is not None:
        mel, frame = coordinate
        receipt['coordinate_reference_feature'] = float(reference[0, mel, frame])
        receipt['coordinate_active_bins'] = [
            {'bin': int(index), 'weight': float(bank[mel, index]),
             'components': {name: {'real': float(value[0, index, frame].real),
                                  'imag': float(value[0, index, frame].imag)}
                            for name, value in spectra.items()}}
            for index in np.flatnonzero(bank[mel])]
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
