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
    args = parser.parse_args()
    sources = [Path(__file__), ROOT / 'tools/artifact_outputs.py',
               ROOT / 'tools/tensor_parity.py']
    sources += [Path(__file__).with_name(name) for name in (
        'replay_packed_fft.py', 'compare_js_feature_boundaries.py',
        'complex_intervention.py', 'normalization_reduction.py')]
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
    from replay_packed_fft import read_capture
    from tensor_parity import compare_tensors
    installed_sources = [Path(inspect.getfile(AudioToMelSpectrogramPreprocessor)),
                         Path(nemo_features.__file__)]
    sources += installed_sources
    hashes.update({str(path.resolve()): digest(path) for path in installed_sources})
    torch.set_num_threads(4)
    processor = AudioToMelSpectrogramPreprocessor(**config).eval()
    _, _, pcm, _, _, expected, valid = read_capture(
        args.capture, args.expected_capture_sha256, args.sample_id)
    with torch.no_grad():
        values, lengths = processor(input_signal=torch.from_numpy(pcm),
                                   length=torch.tensor([pcm.shape[1]], dtype=torch.long))
    actual = values.cpu().numpy()
    metrics = compare_tensors(expected, actual, atol=1e-5, rtol=1e-4)
    passed = metrics['allclose'] and int(lengths[0]) == valid
    if hashes != {str(path.resolve()): digest(path) for path in sources}:
        raise ValueError('Processor diagnostic sources changed during execution')
    if digest(args.config) != args.expected_config_sha256 or digest(args.capture) != args.expected_capture_sha256:
        raise ValueError('Processor inputs changed during execution')
    receipt = {'status': 'NATIVE_PROCESSOR_PASS' if passed else 'NATIVE_PROCESSOR_FAIL',
        'sample_id': args.sample_id, 'config_sha256': args.expected_config_sha256,
        'capture_sha256': args.expected_capture_sha256, 'sources_sha256': hashes,
        'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
        'numba_cache_dir': os.environ.get('NUMBA_CACHE_DIR'),
        'length_expected': valid, 'length_actual': int(lengths[0]), 'features': metrics,
        'feature_bytes_exact': bool(np.array_equal(expected.view(np.uint32), actual.view(np.uint32))),
        'scope': 'Config-only native NeMo processor; no weights, ONNX, browser or ASR acceptance'}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
