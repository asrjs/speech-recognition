"""Reconstruct the official NeMo processor from a pinned config, without weights.

Use a process-local writable NUMBA_CACHE_DIR in restricted Windows environments.
This is native processor evidence, not ONNX/browser/model-port acceptance.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from artifact_outputs import require_new_outputs, write_new_text


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--expected-capture-sha256', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--onnx-probe', action='store_true',
                        help='Compare explicit real-pair adapter and in-memory fp32 ONNX on CPU')
    parser.add_argument('--stage-probe', action='store_true',
                        help='Also observe intermediate outputs and verify instrumentation controls')
    parser.add_argument('--pairwise-normalization', action='store_true',
                        help='Test dynamic fp32 pairwise normalization on identical native log-mel')
    args = parser.parse_args()
    if args.stage_probe and not args.onnx_probe:
        parser.error('--stage-probe requires --onnx-probe')
    if args.pairwise_normalization and not args.stage_probe:
        parser.error('--pairwise-normalization requires --stage-probe')
    sources = [Path(__file__), ROOT / 'tools/artifact_outputs.py',
               ROOT / 'tools/tensor_parity.py']
    sources += [Path(__file__).with_name(name) for name in (
        'replay_packed_fft.py', 'compare_js_feature_boundaries.py',
        'complex_intervention.py', 'normalization_reduction.py')]
    if args.onnx_probe:
        sources.append(ROOT / 'tools/nemo-tdt-onnx-export/processor_real_pair.py')
    if args.stage_probe:
        sources.append(Path(__file__).with_name('processor_export_stages.py'))
    if args.pairwise_normalization:
        sources.append(ROOT / 'tools/nemo-tdt-onnx-export/processor_pairwise_normalization.py')
    require_new_outputs([args.output], protected_inputs=[args.config, args.capture, *sources],
                        protected_directories=[ROOT / 'src', ROOT / 'models'])
    if digest(args.config) != args.expected_config_sha256:
        raise ValueError('Config identity differs')
    if digest(args.capture) != args.expected_capture_sha256:
        raise ValueError('Capture identity differs')
    config = json.loads(args.config.read_text(encoding='utf-8'))['preprocessor'].copy()
    if config.pop('_target_') != 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor':
        raise ValueError('Unsupported processor target')
    hashes = {str(path.resolve()): digest(path) for path in sources}
    import numpy as np
    import torch
    from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
    from nemo.collections.asr.parts.preprocessing import features as nemo_features
    from replay_packed_fft import read_capture, read_stage
    from tensor_parity import compare_tensors
    installed_sources = [Path(inspect.getfile(AudioToMelSpectrogramPreprocessor)),
                         Path(nemo_features.__file__)]
    sources += installed_sources
    hashes.update({str(path.resolve()): digest(path) for path in installed_sources})
    torch.set_num_threads(4)
    processor = AudioToMelSpectrogramPreprocessor(**config).eval()
    _, captured_sample, pcm, oracle_spectrum, _, expected, valid = read_capture(
        args.capture, args.expected_capture_sha256, args.sample_id)
    with torch.no_grad():
        values, lengths = processor(input_signal=torch.from_numpy(pcm),
                                   length=torch.tensor([pcm.shape[1]], dtype=torch.long))
    actual = values.cpu().numpy()
    metrics = compare_tensors(expected, actual, atol=1e-5, rtol=1e-4)
    passed = metrics['allclose'] and int(lengths[0]) == valid
    export_control = None
    if args.onnx_probe:
        if not passed:
            raise ValueError('Official native processor must pass before export control')
        import io
        import onnx
        import onnxruntime as ort
        sys.path.insert(0, str(ROOT / 'tools/nemo-tdt-onnx-export'))
        from processor_real_pair import RealPairProcessor
        adapter = RealPairProcessor(processor).eval()
        waveform = torch.from_numpy(pcm)
        length = torch.tensor([pcm.shape[1]], dtype=torch.long)
        with torch.no_grad():
            pair_features, pair_length = adapter(waveform, length)
        pair_metrics = compare_tensors(expected, pair_features.numpy(), atol=1e-5, rtol=1e-4)
        if not pair_metrics['allclose'] or int(pair_length[0]) != valid:
            raise ValueError('Native real-pair adapter does not preserve the captured oracle')
        buffer = io.BytesIO()
        torch.onnx.export(adapter, (waveform[:, :1600], torch.tensor([1600], dtype=torch.long)),
            buffer, opset_version=17, dynamo=False, input_names=['waveforms', 'waveforms_lens'],
            output_names=['features', 'features_lens'],
            dynamic_axes={'waveforms': {1: 'samples'}, 'features': {2: 'frames'}})
        graph_bytes = buffer.getvalue()
        graph = onnx.load_model_from_string(graph_bytes)
        onnx.checker.check_model(graph)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(graph_bytes, options, providers=['CPUExecutionProvider'])
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Unexpected processor execution provider')
        ort_features, ort_length = session.run(None, {'waveforms': pcm, 'waveforms_lens': length.numpy()})
        ort_metrics = compare_tensors(expected, ort_features, atol=1e-5, rtol=1e-4)
        export_control = {'status': 'PASS' if ort_metrics['allclose'] and int(ort_length[0]) == valid else 'FAIL',
            'native_real_pair': pair_metrics,
            'native_real_pair_bytes_exact': bool(np.array_equal(expected.view(np.uint32), pair_features.numpy().view(np.uint32))),
            'onnx_features': ort_metrics, 'onnx_length': int(ort_length[0]),
            'graph_sha256': hashlib.sha256(graph_bytes).hexdigest(), 'graph_bytes': len(graph_bytes),
            'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'providers': session.get_providers(),
            'scope': 'In-memory real-pair fp32 processor, dynamic sample axis, batch1 native CPU only'}
        if args.stage_probe:
            from processor_export_stages import run_stage_control, run_stft_input_control
            export_control['stages'] = run_stage_control(processor, waveform, length,
                oracle_spectrum, expected, ort_features, ort_length, pairwise=args.pairwise_normalization)
            export_control['stages']['stft_input_control'] = run_stft_input_control(
                processor, waveform, length, read_stage(captured_sample, 'feature_stft/input', 'float32'),
                oracle_spectrum)
    if hashes != {str(path.resolve()): digest(path) for path in sources}:
        raise ValueError('Processor diagnostic sources changed during execution')
    if digest(args.config) != args.expected_config_sha256 or digest(args.capture) != args.expected_capture_sha256:
        raise ValueError('Processor inputs changed during execution')
    native_status = 'NATIVE_PROCESSOR_PASS' if passed else 'NATIVE_PROCESSOR_FAIL'
    status = ('PROCESSOR_EXPORT_' + export_control['status']) if export_control is not None else native_status
    receipt = {'status': status, 'native_status': native_status,
        'sample_id': args.sample_id, 'config_sha256': args.expected_config_sha256,
        'capture_sha256': args.expected_capture_sha256, 'sources_sha256': hashes,
        'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
        'numba_cache_dir': os.environ.get('NUMBA_CACHE_DIR'),
        'length_expected': valid, 'length_actual': int(lengths[0]), 'features': metrics,
        'onnx_export_control': export_control,
        'feature_bytes_exact': bool(np.array_equal(expected.view(np.uint32), actual.view(np.uint32))),
        'scope': 'Config-only native NeMo processor; no weights, ONNX, browser or ASR acceptance'}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))
    if not passed or (export_control is not None and export_control['status'] != 'PASS'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
